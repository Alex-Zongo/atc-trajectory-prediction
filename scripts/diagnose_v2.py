"""Diagnose under-dispersion and check the SBM, one hypothesis per sub-command.

    python scripts/diagnose_v2.py nfe       --ctp runs/v2/ctp.pt          # H2 sampler
    python scripts/diagnose_v2.py overfit   --ctp runs/v2/ctp.pt          # H3
    python scripts/diagnose_v2.py time      --ctp a.pt b.pt               # H4 flow time
    python scripts/diagnose_v2.py hindsight --ctp runs/v2/ctp.pt          # H5
    python scripts/diagnose_v2.py noise     --ctp runs/v2/ctp.pt          # H1 noise floor
    python scripts/diagnose_v2.py sbm       --sbm runs/v2/sbm.pt          # order, beam
    python scripts/diagnose_v2.py callsign  --sbm runs/v2/sbm.pt [--tag x] # which cue decides
    python scripts/diagnose_v2.py turn      --sbm runs/v2/sbm.pt [--tag x] # unheard turn direction

Every command scores the oracle condition (true Z) on a fixed validation subset with
seeded noise, so differences come from the factor under test only. Results go to
results/v2/diagnose_<command>.json and a figure next to it.

  H1  noise floor   spread-skill against the smoothed truth and against raw ADS-B. If
                    the model is calibrated on smoothed targets but under-dispersed on
                    raw ones, the gap is measurement noise, which the targets exclude.
  H2  sampler       energy score and spread at 60 s for NFE 1 ... 64, Euler and Heun.
                    Spread that keeps growing with NFE means few steps collapse samples.
  H3  overfitting   the same scores on training and validation windows.
  H4  flow time     checkpoints trained with logit-normal and uniform flow time.
  H5  hindsight     scores by delay and by whether an unannounced maneuver follows. If
                    labels leak hindsight, short delays look far better than long ones.
  callsign          does the SBM decide "is it for me" from the callsign or from the
                    history? Test messages addressed to the aircraft, with the callsign
                    heard, scored three ways: A true callsign and history; B another
                    aircraft's callsign, same history (only the callsign says "not me");
                    C true callsign, another aircraft's history (only the history does).
                    A model that compares callsigns sends B to "other"; one that relies
                    on history consistency sends C there. Also D, the trained "other"
                    messages, and the same on swap messages when the profile has them.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from atc_forecast.evaluation.metrics_v2 import (
    at_raw_times,
    positions_from_velocities,
    spread_skill,
    summarize,
)
from atc_forecast.models.ctp import load_ctp, seeded_noise
from atc_forecast.training.common import pick_device
from atc_forecast.training.data_v2 import WindowStore
from atc_forecast.training.tracking import Tracker, add_args

BLUE, ORANGE, AQUA, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"


def oracle_batches(ws, idx, dev, n=128):
    for i in range(0, len(idx), n):
        j = idx[i:i + n]
        z = ws.z_true[j].copy()
        z[~ws.is_man[j], 0] = 1
        yield ws.batch(j, dev, z)[0]


@torch.no_grad()
def score(model, ws, idx, dev, S=32, nfe=12, method="euler", keep=False):
    recs, Xs, ys, raws = [], [], [], []
    for b in oracle_batches(ws, idx, dev):
        noise = seeded_noise(b["window_id"], S, model.space.shape, run_seed=77)
        V = model.sample(b, S, noise=noise, nfe=nfe, method=method).cpu().numpy()
        X = positions_from_velocities(V)
        y = positions_from_velocities(b["fut_vel"].cpu().numpy())
        sc = summarize(X, y)
        sc["spread60_h"] = float(np.linalg.norm(X[:, :, -1, :2].std(1), axis=-1).mean())
        recs.append((len(y), sc))
        if keep:
            Xs.append(X)
            ys.append(y)
            raws.append(ws.raw_future[ws.meta["raw_row"].to_numpy()[b["window_id"]]])
    n = sum(k for k, _ in recs)
    out = {key: sum(k * s[key] for k, s in recs) / n for key in recs[0][1]}
    return (out, np.concatenate(Xs), np.concatenate(ys), np.concatenate(raws)) if keep else out


def subset(ws, split, n, seed=5):
    pool = ws.split(split)
    return np.sort(np.random.default_rng(seed).choice(pool, min(n, len(pool)), replace=False))


def cmd_nfe(a, ws, dev, out):
    m = load_ctp(a.ctp[0], dev)[0]
    idx = subset(ws, "val", a.n)
    res = {}
    for method in ("euler", "heun"):
        for nfe in (1, 2, 4, 8, 12, 16, 32, 64):
            if method == "heun" and nfe == 1:
                continue
            res[f"{method}_{nfe}"] = score(m, ws, idx, dev, nfe=nfe, method=method)
            print(method, nfe, {k: round(res[f'{method}_{nfe}'][k], 2)
                                for k in ("energy_h", "spread60_h")}, flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    for method, col in (("euler", BLUE), ("heun", ORANGE)):
        ks = [k for k in res if k.startswith(method)]
        n = [int(k.split("_")[1]) for k in ks]
        ax[0].plot(n, [res[k]["energy_h"] for k in ks], color=col, marker="o", ms=4, label=method)
        ax[1].plot(n, [res[k]["spread60_h"] for k in ks], color=col, marker="o", ms=4)
    for x, t in zip(ax, ("energy score, horizontal (m)", "sample spread at 60 s (m)")):
        x.set_xscale("log", base=2)
        x.set_xlabel("NFE")
        x.set_title(t)
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=150)
    return res


def cmd_overfit(a, ws, dev, out):
    m = load_ctp(a.ctp[0], dev)[0]
    return {s: score(m, ws, subset(ws, s, a.n), dev) for s in ("train", "val")}


def cmd_time(a, ws, dev, out):
    idx = subset(ws, "val", a.n)
    res = {}
    for p in a.ctp:
        m, ck = load_ctp(p, dev)
        res[Path(p).stem] = {"time_sampling": ck["cfg"]["time_sampling"],
                             **score(m, ws, idx, dev)}
    return res


def cmd_hindsight(a, ws, dev, out):
    m = load_ctp(a.ctp[0], dev)[0]
    idx = subset(ws, "val", a.n * 2)
    meta = ws.meta.iloc[idx]
    groups = {
        "delay 2-5 s": idx[(meta["delta"] < 5).to_numpy()],
        "delay 5-8.5 s": idx[((meta["delta"] >= 5) & (meta["delta"] < 8.5)).to_numpy()],
        "delay 8.5-12 s": idx[(meta["delta"] >= 8.5).to_numpy()],
        "steady": idx[(meta["kind"] == "steady").to_numpy()],
        "no later onset": idx[(meta["next_onset_rel"] > 60).to_numpy()],
        "later onset": idx[(meta["next_onset_rel"] <= 60).to_numpy()],
    }
    return {k: {"n": len(v), **score(m, ws, v, dev)} for k, v in groups.items() if len(v)}


def cmd_noise(a, ws, dev, out):
    m = load_ctp(a.ctp[0], dev)[0]
    idx = subset(ws, "val", a.n)
    sc, X, y, raw = score(m, ws, idx, dev, keep=True)
    ss_smooth = spread_skill(X, y)                     # [60, 3]
    # against raw: per raw point, sample spread vs error of the mean
    Q = at_raw_times(X, raw)                           # [B, S, R, 3]
    ok = ~np.isnan(raw[..., 0])
    sd = Q.std(1)
    err = Q.mean(1) - raw[..., 1:]
    t = np.nan_to_num(raw[..., 0])
    bins = np.arange(0, 61, 5)
    ss_raw = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = ok & (t > lo) & (t <= hi)
        ss_raw.append([float(np.sqrt((sd[..., c][s] ** 2).mean() / (err[..., c][s] ** 2).mean()))
                       for c in range(3)])
    ss_raw = np.array(ss_raw)
    off = ws.meta.iloc[idx][["offset0_h_m", "offset0_v_m"]]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    c = (bins[:-1] + bins[1:]) / 2
    for k, name in enumerate(("along", "right", "vertical")):
        ax[k].plot(np.arange(1, 61), ss_smooth[:, k], color=BLUE, label="vs smoothed truth")
        ax[k].plot(c, ss_raw[:, k], color=ORANGE, marker="o", ms=4, label="vs raw ADS-B")
        ax[k].axhline(1, color=INK2, lw=0.8, ls=":")
        ax[k].set_title(f"spread / skill, {name}")
        ax[k].set_xlabel("lead time (s)")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=150)
    return {"scores": sc, "spread_skill_smoothed": ss_smooth.tolist(),
            "spread_skill_raw_5s_bins": ss_raw.tolist(),
            "offset0_h_m_median": float(off["offset0_h_m"].median()),
            "offset0_v_ft_median_abs": float(off["offset0_v_m"].abs().median() * 3.28084)}


def cmd_sbm(a, ws, dev, out):
    from atc_forecast.models.sbm import load_sbm
    from atc_forecast.training.data_v2 import MessageStore
    sbm = load_sbm(a.sbm, dev)[0]
    ms = MessageStore(a.data, a.profile, ws)
    rows = np.sort(np.random.default_rng(3).choice(ms.rows("val"), a.n, replace=False,
                                                   p=ms.weights(ms.rows("val"))))
    spreads, cert, exact = [], [], []
    for i in range(0, len(rows), 256):
        b = ms.batch(rows[i:i + 256], dev)
        if sbm.cfg.factorization == "anyorder":
            spreads.append(sbm.order_spread(b, b["z"]).cpu().numpy())
        zb, _, c = sbm.beam(b, width=16)
        cert.append(c.cpu().numpy())
        exact.append((zb[:, 0] == b["z"]).all(1).cpu().numpy())
    res = {"beam_certified": float(np.concatenate(cert).mean()),
           "mode_exact_match": float(np.concatenate(exact).mean())}
    if spreads:
        sp = np.concatenate(spreads)
        res.update(order_spread_median=float(np.median(sp)),
                   order_spread_p90=float(np.quantile(sp, 0.9)))
    return res


def cmd_callsign(a, ws, dev, out):
    from atc_forecast.evaluation.addressee_v2 import callsign_heard
    from atc_forecast.models.sbm import load_sbm
    from atc_forecast.training.data_v2 import MessageStore

    ms = MessageStore(a.data, a.profile, ws)
    sbm = load_sbm(a.sbm, dev)[0]
    rng = np.random.default_rng(0)
    test = np.zeros(len(ws.meta), bool)
    test[ws.split("test")] = True
    first = ms.msgs["draw"].to_numpy() == 0

    def pick(variant, n):
        r = np.where((ms.variant == variant) & first & test[ms.window_id])[0]
        if not len(r):
            return r
        r = r[callsign_heard(ms, r)]
        return np.sort(rng.choice(r, min(n, len(r)), replace=False)) if len(r) else r

    def p_for_me(b):
        with torch.no_grad():
            q = torch.cat([sbm.status_probs({k: (v[i:i + 128] if hasattr(v, "__len__") else v)
                                             for k, v in b.items()})
                           for i in range(0, len(b["z"]), 128)])
        return q[:, :2].sum(1).cpu().numpy()

    rows = pick("own", a.n)
    wid = ms.window_id[rows]
    icao = ws.meta["icao24"].to_numpy()
    pool = ws.split("test")
    donor = np.array([next(d for d in rng.choice(pool, 50) if icao[d] != icao[w]) for w in wid])
    b = ms.batch(rows, dev)
    res = {"n": len(rows)}
    probs = {"A": p_for_me(b)}
    bB = dict(b)
    bB["cs_tokens"] = torch.as_tensor(ms.own_cs[donor].astype(np.int64), device=dev)
    probs["B"] = p_for_me(bB)
    hb, _ = ws.batch(donor, dev, sort=False)
    bC = dict(b)
    bC.update({k: hb[k] for k in ("hist", "ctx", "wake", "engine", "n_engines", "state")})
    probs["C"] = p_for_me(bC)
    for v, key in (("other", "D"), ("swap", "E")):
        r = pick(v, a.n)
        if len(r):
            probs[key] = p_for_me(ms.batch(r, dev))
    names = {"A": "own message, true callsign and history", "B": "other aircraft's callsign",
             "C": "other aircraft's history", "D": "trained 'other' messages",
             "E": "swap messages (own instruction, other callsign)"}
    for k, v in probs.items():
        res[k] = {"what": names[k], "mean_p_for_me": float(v.mean()),
                  "share_for_me": float((v > 0.5).mean())}
    res["callsign_reliance"] = 1 - res["B"]["share_for_me"]   # 1 = a wrong callsign always rejects
    res["history_reliance"] = 1 - res["C"]["share_for_me"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bins = np.linspace(0, 1, 21)
    for k, v in probs.items():
        ax.hist(v, bins, histtype="step", lw=1.8, label=f"{k}: {names[k]} ({(v > .5).mean():.0%})")
    ax.set_yscale("log")
    ax.set_xlabel("SBM probability the message is for this aircraft")
    ax.set_ylabel("messages")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=130)
    return res


def cmd_turn(a, ws, dev, out):
    """Does the SBM use the aircraft's track to infer an unheard turn direction? Test
    messages with a heading instruction whose transcript lost "left"/"right": the SBM's
    direction (beam marginal over L vs R atoms) against the flown direction and the
    shorter turn from the present track. Then the same messages with the track mirrored
    about the instructed heading (sin/cos of 2H - psi in ctx), which swaps the shorter
    turn and leaves the track-frame history unchanged."""
    from atc_forecast.data import atoms as A
    from atc_forecast.models.sbm import load_sbm
    from atc_forecast.training.data_v2 import MessageStore

    ms = MessageStore(a.data, a.profile, ws)
    sbm = load_sbm(a.sbm, dev)[0]
    test = np.zeros(len(ws.meta), bool)
    test[ws.split("test")] = True
    lat = ms.msgs["atom_lateral"].to_numpy()
    desc = np.array([A.describe("lateral", int(x)) for x in range(lat.max() + 1)])
    turn = np.char.startswith(desc[lat], "L") | np.char.startswith(desc[lat], "R")
    text = ms.msgs["text"].str.lower()
    said = ms.msgs["said"].str.lower()
    heading = text.str.contains("heading") & said.str.contains(r"turn (?:left|right)")
    lost = ~text.str.contains(r"\b(?:left|right)\b")
    base = (ms.variant == "own") & test[ms.window_id] & turn & heading.to_numpy()
    sign = np.array([1 if d.startswith("R") else -1 if d.startswith("L") else 0 for d in desc])

    def p_right(rows, mirror=False):
        out_ = []
        for i in range(0, len(rows), 256):
            b = ms.batch(rows[i:i + 256], dev)
            if mirror:
                h = np.array([int(desc[x][1:]) for x in lat[rows[i:i + 256]]], float)
                psi = np.arctan2(b["ctx"][:, 1].cpu().numpy(), b["ctx"][:, 2].cpu().numpy())
                psi2 = np.deg2rad(2 * h) - psi
                b["ctx"] = b["ctx"].clone()
                b["ctx"][:, 1] = torch.as_tensor(np.sin(psi2), dtype=torch.float32, device=dev)
                b["ctx"][:, 2] = torch.as_tensor(np.cos(psi2), dtype=torch.float32, device=dev)
            with torch.no_grad():
                zb, lp, _ = sbm.beam(b, width=16)
            p = torch.exp(lp).cpu().numpy()
            s = sign[zb[:, :, 1].cpu().numpy()]
            r, l = (p * (s > 0)).sum(1), (p * (s < 0)).sum(1)
            out_.append(r / np.maximum(r + l, 1e-9))
        return np.concatenate(out_)

    res = {}
    for name, m in (("lost", base & lost.to_numpy()), ("heard", base & ~lost.to_numpy())):
        rows = np.where(m)[0]
        if name == "heard" and len(rows) > a.n:
            rows = np.sort(np.random.default_rng(0).choice(rows, a.n, replace=False))
        trk = ws.meta["trk_deg"].to_numpy()[ms.window_id[rows]]
        h = np.array([int(desc[x][1:]) for x in lat[rows]], float)
        d = (h - trk + 540) % 360 - 180                      # >0: the shorter turn is right
        flown = sign[lat[rows]] > 0
        pr = p_right(rows)
        r = {"n": len(rows), "flown_is_shorter": float(((d > 0) == flown).mean()),
             "sbm_matches_flown": float(((pr > 0.5) == flown).mean()),
             "sbm_matches_shorter": float(((pr > 0.5) == (d > 0)).mean()),
             "mean_p_flown": float(np.where(flown, pr, 1 - pr).mean())}
        if name == "lost":
            pm = p_right(rows, mirror=True)
            r.update(mirrored_matches_shorter=float(((pm > 0.5) == (d < 0)).mean()),
                     mirrored_direction_flips=float(((pm > 0.5) != (pr > 0.5)).mean()),
                     mean_abs_shift=float(np.abs(pm - pr).mean()))
            bins = [0, 30, 90, 150, 180]
            r["by_turn"] = {f"{lo}-{hi}": {
                "n": int(k.sum()), "flown_is_shorter": float(((d[k] > 0) == flown[k]).mean()),
                "sbm_matches_flown": float(((pr[k] > 0.5) == flown[k]).mean()),
                "sbm_matches_shorter": float(((pr[k] > 0.5) == (d[k] > 0)).mean()),
                "mirrored_direction_flips": float(((pm[k] > 0.5) != (pr[k] > 0.5)).mean())}
                for lo, hi in itertools.pairwise(bins) for k in [(np.abs(d) >= lo) & (np.abs(d) < hi)] if k.any()}
        res[name] = r
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("nfe", "overfit", "time", "hindsight", "noise", "sbm",
                                        "callsign", "turn"))
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--ctp", nargs="*", default=[])
    ap.add_argument("--sbm")
    ap.add_argument("--data", default="data/v2")
    ap.add_argument("--profile", default="jacktol")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out-dir", default="results/v2")
    ap.add_argument("--device", default=None)
    add_args(ap)
    a = ap.parse_args()
    dev = pick_device(a.device)
    ws = WindowStore(a.data)
    out = Path(a.out_dir) / f"diagnose_{a.command}{a.tag}"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = getattr(sys.modules[__name__], f"cmd_{a.command}")(a, ws, dev, out)
    out.with_suffix(".json").write_text(json.dumps(res, indent=2))
    tr = Tracker(a.wandb, a.wandb_project, a.run_name or f"diagnose_{a.command}", vars(a),
                 group=a.wandb_group, job_type="diagnose")
    if tr.run:
        flat = {}

        def walk(p, v):
            if isinstance(v, dict):
                for k, x in v.items():
                    walk(f"{p}/{k}", x)
            elif isinstance(v, (int, float)):
                flat[p] = v
        walk(f"diag/{a.command}", res)
        tr.summary(flat)
        if out.with_suffix(".png").exists():
            tr.log({f"diag/{a.command}/figure": tr.wandb.Image(str(out.with_suffix(".png")))})
        tr.finish()
    print(json.dumps(res, indent=1)[:3000])


if __name__ == "__main__":
    main()
