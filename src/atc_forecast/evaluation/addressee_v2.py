"""Who is the message for? Addressee resolution over the traffic on frequency.

    python -m atc_forecast.evaluation.addressee_v2 --sbm runs/v2/sbm.pt --ctp runs/v2/ctp.pt \
        [--hist-only runs/v2/ctp_none.pt] --n-aircraft 8 --n 1000 --out results/v2/addressee.json

Operational picture. A controller works one frequency with N aircraft on it, all of them
on ADS-B. Every transmission is for one of them (or, rarely, for someone we do not track).
The forecaster hears it once and must update all N forecasts:

  1. score     for every aircraft a on frequency, the SBM reads the transcript with a's
               callsign and history: r_a = q(status in {own motion, own non-motion}).
  2. evidence  r_a is a posterior under the training mix, in which 80 % of messages were
               for the aircraft (MessageStore.MIX). Dividing out those prior odds leaves
               the evidence the transcript gives for a: L_a = odds(r_a) / odds(0.8).
  3. compete   the message is for exactly one aircraft, each a priori as likely, or for
               someone off our list:  P(a) = L_a / (sum_b L_b + out_weight).
               A clipped callsign ("...descend and maintain 2000") gives every aircraft
               about the same L, so each gets about 1/N; a clear callsign gives one
               aircraft nearly all of it, and a confusable one splits it between the two.
  4. compose   aircraft a's forecast draws its interpretations with status "for a"
               with probability P(a) (the SBM's own split between motion and non-motion,
               and its motion slots given that), else status "other", which the CTP
               reads as "the radio told us nothing" (the history-only answer).

Conditions, all on the same scenes and seeds:
  independent  today: every aircraft reads the message alone, under the training prior
  prior        step 2 with the prior 1/N instead of 0.8, but no competition
  scene        steps 1-4
  known        the addressee is known (P = 1 for it, 0 for the rest): the bound for 1-3
  hist_only    the history-only CTP, if given
  scene_hist   scene, with every reading that carries no motion (status own non-motion or
               other) forecast by the history-only CTP instead of the CTP given NONE: the
               routed form of "the radio told us nothing" (needs --hist-only)

Scenes. Each test "own" message (an instruction to aircraft a, through the noisy channel)
is placed among the N - 1 aircraft whose windows are closest in time to a's on the same
day (real concurrent traffic in the same airspace, at most --max-dt seconds apart). All N
are scored; the addressee and up to --bystanders of the others that are not training
windows are forecast and evaluated. A frequency-weighted score, (addressee + (N - 1) x
bystander) / N, is what the forecaster would see on air.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..data import atoms as A
from ..models.ctp import load_ctp, seeded_noise
from ..models.sbm import load_sbm, window_generators
from ..training.common import pick_device
from ..training.data_v2 import MessageStore, WindowStore
from ..training.tracking import Tracker, add_args
from .compose_v2 import ctp_samples, fixed_messages, per_message
from .metrics_v2 import nested, positions_from_velocities

ADDRESSED_TRAIN = 1.0 - MessageStore.MIX["other"]    # share of training messages "for me"
CONDITIONS = ("independent", "prior", "scene", "known")


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def evidence(r, prior=ADDRESSED_TRAIN):
    """Likelihood ratio of 'addressed to this aircraft' from the SBM's posterior r."""
    return np.exp(logit(r) - logit(prior))


def resolve(r, out_weight=1.0, prior=ADDRESSED_TRAIN):
    """Posterior over the N aircraft on frequency [N] (sums to < 1; the rest is 'someone
    we do not track')."""
    L = evidence(np.asarray(r, float), prior)
    return L / (L.sum() + out_weight)


def prior_shift(r, n, prior=ADDRESSED_TRAIN):
    """Each aircraft alone, with the prior 1/n instead of the training prior."""
    L = evidence(np.asarray(r, float), prior)
    return L / (L + (n - 1))


def gated_status(q, p_addr):
    """Status distribution [B, 3] that puts P(for a) = p_addr, split between own motion
    and own non-motion as the SBM does."""
    q = np.asarray(q, float)
    own = q[:, :2] / q[:, :2].sum(1, keepdims=True).clip(1e-9)
    return np.concatenate([own * p_addr[:, None], (1 - p_addr)[:, None]], 1)


def scenes(ws: WindowStore, windows, n_aircraft, max_dt):
    """For each window, the other aircraft on frequency: windows of other aircraft on the
    same day, closest in t0 (one per aircraft), at most max_dt apart. Returns a list of
    arrays of window ids (may be shorter than n_aircraft - 1 in quiet traffic)."""
    m = ws.meta
    day, t0, ic = m["day"].to_numpy(), m["t0"].to_numpy(), m["icao24"].to_numpy()
    ok = ws.ok
    by_day = {}
    for d in np.unique(day[windows]):
        idx = np.where((day == d) & ok)[0]
        by_day[d] = idx[np.argsort(t0[idx])]
    out = []
    for w in windows:
        idx = by_day[day[w]]
        ts = t0[idx]
        lo, hi = np.searchsorted(ts, t0[w] - max_dt), np.searchsorted(ts, t0[w] + max_dt)
        cand = idx[lo:hi]
        cand = cand[np.argsort(np.abs(t0[cand] - t0[w]))]
        seen, keep = {ic[w]}, []
        for c in cand:
            if ic[c] not in seen:
                seen.add(ic[c])
                keep.append(c)
            if len(keep) == n_aircraft - 1:
                break
        out.append(np.array(keep, np.int64))
    return out


def pair_batch(ms: MessageStore, msg_rows, wids, z, device):
    """Message msg_rows[i] heard by the aircraft of window wids[i], with labels z [n, 4]."""
    b, _ = ms.windows.batch(wids, device, sort=False)
    t = lambda x, dt: torch.as_tensor(np.asarray(x), dtype=dt).to(device)
    b["tokens"] = t(ms.tokens[msg_rows].astype(np.int64), torch.long)
    b["kappa"] = t(ms.kappa[msg_rows].astype(np.float32) / 255.0, torch.float32)
    b["cs_tokens"] = t(ms.own_cs[wids].astype(np.int64), torch.long)
    b["z"] = t(z, torch.long)
    return b


def callsign_heard(ms: MessageStore, rows):
    """Whether any callsign word survived the channel: the heard text starts with an
    instruction word when the callsign was clipped."""
    first = ms.msgs["text"].to_numpy()[rows]
    verbs = {"turn", "descend", "climb", "fly", "increase", "reduce", "maintain", "left",
             "right", "heading", "expedite", "speed"}
    return np.array([bool(t) and t.split()[0] not in verbs for t in first])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/v2")
    ap.add_argument("--profile", default="jacktol")
    ap.add_argument("--sbm", required=True)
    ap.add_argument("--ctp", required=True)
    ap.add_argument("--hist-only")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=1000, help="own messages (fixed random subset)")
    ap.add_argument("--n-aircraft", type=int, default=8, help="aircraft on frequency")
    ap.add_argument("--max-dt", type=float, default=60.0)
    ap.add_argument("--bystanders", type=int, default=3, help="evaluated per message")
    ap.add_argument("--out-weight", type=float, default=1.0,
                    help="weight of 'for someone we do not track', in aircraft")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--s", type=int, default=6)
    ap.add_argument("--nfe", type=int, default=12)
    ap.add_argument("--method", default="euler", choices=("euler", "heun"))
    ap.add_argument("--batch", type=int, default=16, help="messages per batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    add_args(ap)
    a = ap.parse_args()

    dev = pick_device(a.device)
    # open the W&B run first, so progress and system usage show live
    tr = Tracker(a.wandb, a.wandb_project, a.run_name or Path(a.out).stem, vars(a),
                 group=a.wandb_group, job_type="eval")
    ws = WindowStore(a.data)
    ms = MessageStore(a.data, a.profile, ws)
    sbm, sbm_ck = load_sbm(a.sbm, dev)
    prior = sbm_ck.get("addressed_prior", ADDRESSED_TRAIN)   # the SBM's own training mix
    ctp = load_ctp(a.ctp, dev)[0]
    hist = load_ctp(a.hist_only, dev)[0] if a.hist_only else None
    split = ws.meta["split"].to_numpy()

    pool = ws.split(a.split)
    win = np.sort(np.random.default_rng(99).choice(pool, min(4 * a.n, len(pool)), replace=False))
    rows = fixed_messages(ms, win, variants=("own",))
    rows = np.sort(np.random.default_rng(7).choice(rows, min(a.n, len(rows)), replace=False))
    heard = callsign_heard(ms, rows)
    others = scenes(ws, ms.window_id[rows], a.n_aircraft, a.max_dt)
    none_z = np.array([A.STATUS.index("other"), A.NONE, A.NONE, A.NONE])

    rec = {k: [] for k in ("row", "n_on_freq", "heard", "p_addr", "p_addr_prior",
                           "p_addr_indep", "p_best_other", "addr_top1")}
    ev = []                                       # (role, condition, energy_h, ade_h)
    with torch.no_grad():
        for i in range(0, len(rows), a.batch):
            r = rows[i:i + a.batch]
            sc = others[i:i + a.batch]
            # 1. score every aircraft on frequency for every message
            msg = np.concatenate([[m] * (1 + len(o)) for m, o in zip(r, sc)])
            wid = np.concatenate([[ms.window_id[m], *o] for m, o in zip(r, sc)])
            q = np.concatenate([sbm.status_probs(pair_batch(ms, msg[j:j + 256], wid[j:j + 256],
                                                            np.zeros((len(msg[j:j + 256]), 4)), dev)
                                                 ).cpu().numpy() for j in range(0, len(msg), 256)])
            r_addr = q[:, :2].sum(1)
            # 2-3. evidence and competition, per scene
            p = {c: np.zeros(len(msg)) for c in CONDITIONS}
            role = np.zeros(len(msg), bool)               # True for the addressee
            off = 0
            for m, o in zip(r, sc):
                n = 1 + len(o)
                s = slice(off, off + n)
                p["independent"][s] = r_addr[s]
                p["prior"][s] = prior_shift(r_addr[s], a.n_aircraft, prior)
                p["scene"][s] = resolve(r_addr[s], a.out_weight, prior)
                p["known"][s] = np.r_[1.0, np.zeros(n - 1)]
                role[off] = True
                rec["row"].append(m)
                rec["n_on_freq"].append(n)
                rec["p_addr"].append(p["scene"][off])
                rec["p_addr_prior"].append(p["prior"][off])
                rec["p_addr_indep"].append(r_addr[off])
                rec["p_best_other"].append(p["scene"][off + 1:off + n].max() if n > 1 else 0.0)
                rec["addr_top1"].append(bool(np.argmax(p["scene"][s]) == 0))
                off += n
            # 4. forecast the addressee and some bystanders that are not training windows
            ev_idx, off = [], 0
            for m, o in zip(r, sc):
                n = 1 + len(o)
                ev_idx.append(off)
                by = [off + 1 + j for j, w in enumerate(o) if split[w] != "train"][:a.bystanders]
                ev_idx += by
                off += n
            ev_idx = np.array(ev_idx)
            z_true = np.where(role[ev_idx, None], ms.z[msg[ev_idx]], none_z[None])
            b = pair_batch(ms, msg[ev_idx], wid[ev_idx], z_true, dev)
            y = positions_from_velocities(b["fut_vel"].cpu().numpy())
            gens = lambda: window_generators(wid[ev_idx], a.seed)
            outs = {}
            for c in CONDITIONS:
                ps = None if c == "independent" else torch.as_tensor(
                    gated_status(q[ev_idx], p[c][ev_idx]), dtype=torch.float32)
                zs = sbm.sample(b, a.k, gens(), p_status=ps)
                outs[c] = ctp_samples(ctp, b, zs, a.k, a.s, a.seed, a.nfe, a.method)
                if c == "scene":
                    zs_scene = zs
            outs["oracle"] = ctp_samples(ctp, b, b["z"][:, None].expand(-1, a.k, -1), a.k, a.s,
                                         a.seed, a.nfe, a.method)
            if hist is not None:
                noise = seeded_noise(b["window_id"], a.k * a.s, hist.space.shape, a.seed)
                outs["hist_only"] = hist.sample(b, a.k * a.s, noise=noise, nfe=a.nfe,
                                                method=a.method)
                silent = (zs_scene[..., 1:] == A.NONE).all(-1)                # [B, K]
                m = silent.repeat_interleave(a.s, 1)[..., None, None]
                outs["scene_hist"] = torch.where(m, outs["hist_only"], outs["scene"])
            for c, V in outs.items():
                pm = per_message(positions_from_velocities(V.cpu().numpy()), y, nested(a.k, a.s))
                for j in range(len(ev_idx)):
                    ev.append((bool(role[ev_idx[j]]), c, float(pm["energy_h"][j]),
                               float(pm["ade_h"][j]), float(pm["energy_v"][j]),
                               bool(heard[np.searchsorted(rows, msg[ev_idx[j]])])))
            print(f"{min(i + a.batch, len(rows))}/{len(rows)} messages", flush=True)
            tr.log({"progress/messages": min(i + a.batch, len(rows)), "progress/fraction": min(i + a.batch, len(rows)) / len(rows)})

    rec = {k: np.asarray(v) for k, v in rec.items()}
    ev_role = np.array([e[0] for e in ev])
    ev_cond = np.array([e[1] for e in ev])
    ev_e, ev_ade, ev_ev = (np.array([e[i] for e in ev]) for i in (2, 3, 4))
    ev_heard = np.array([e[5] for e in ev])
    N = a.n_aircraft
    res = {"args": vars(a), "n_messages": int(len(rows)),
           "n_on_frequency_mean": float(rec["n_on_freq"].mean()),
           "callsign_heard": float(heard.mean()),
           "addressee": {
               "top1": float(rec["addr_top1"].mean()),
               "p_mean": {"scene": float(rec["p_addr"].mean()),
                          "prior": float(rec["p_addr_prior"].mean()),
                          "independent": float(rec["p_addr_indep"].mean())},
               "p_mean_heard": float(rec["p_addr"][heard].mean()),
               "p_mean_clipped": float(rec["p_addr"][~heard].mean()) if (~heard).any() else None,
               "p_best_other_mean": float(rec["p_best_other"].mean())},
           "by_condition": {}}
    for c in np.unique(ev_cond):
        d = {}
        for name, mk in (("addressee", ev_role), ("bystander", ~ev_role)):
            s = (ev_cond == c) & mk
            d[name] = {"n": int(s.sum()), "energy_h": float(ev_e[s].mean()),
                       "ade_h": float(ev_ade[s].mean()), "energy_v": float(ev_ev[s].mean()),
                       "energy_h_heard": float(ev_e[s & ev_heard].mean()),
                       "energy_h_clipped": float(ev_e[s & ~ev_heard].mean())
                       if (s & ~ev_heard).any() else None}
        d["on_air"] = {"energy_h": (d["addressee"]["energy_h"]
                                    + (N - 1) * d["bystander"]["energy_h"]) / N}
        res["by_condition"][c] = d
    if tr.run:
        tr.summary({f"{c}/{k}": v["energy_h"] for c, d in res["by_condition"].items()
                    for k, v in d.items()} | {"addressee_top1": res["addressee"]["top1"]})
        tr.finish()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("addressee", "by_condition")}, indent=1))


if __name__ == "__main__":
    main()
