"""Evaluate the v2 models on real transmissions matched to real ADS-B.

    python -m atc_forecast.evaluation.real_v2 \
        --transmissions transmissions.jsonl --adsb "adsb/*.parquet" [--source readsb] \
        --sbm runs/v2/sbm_jacktol_anyorder_s0.pt --ctp runs/v2/ctp_basis_atoms_zfix_s0.pt \
        [--hist-only runs/v2/ctp_basis_none_s0.pt] [--scene-radius-nm 40] \
        --out results/real.json [--per-message results/real_messages.jsonl]

Input, one JSON object per line (asr/transcribe.py writes the word part):

    {"id": "kjfk_app_0001",
     "t_end": 1767225612.4,                     # or ISO 8601; the prediction time t0
     "words": [{"word": "united", "probability": 0.98}, {"word": "two", ...}, ...],
     "icao24": "a1b2c3",                        # the addressee, if known; else
     "callsign": "UAL2",                        # its ICAO callsign, matched in the ADS-B
     "group": "descend"}                        # optional: break the scores down by this

"text" may replace "words" when the recognizer gives no confidences (each word then gets
--default-conf). ADS-B: OpenSky state vectors (parquet/CSV) or adsb.lol / readsb trace
files (inference/adsb.py); the file must cover 40 s before to 64 s after each t_end.

What a matched set measures. The trajectory after a transmission is the real outcome of
whatever was said, so p(Y | O, X) is scored end to end without any instruction label:
the score is the energy score of the K x S composed futures against the recorded 60 s,
as on the synthetic test set. When an aircraft does not act on the instruction within
60 s, the forecast is penalized as in operation. No semantic accuracy is reported here,
since real transmissions have no interpretation label; supply "group" to see where the
model wins or loses.

Conditions (all on the same windows and seeds):

  composed   the main model: the SBM reads the message, K interpretations, S futures each
  known      as composed, with the status set to "for this aircraft" (the addressee is
             known from the matching, as in addressee_v2's "known")
  mode       the SBM's single most likely interpretation, K x S futures from it
  hist_only  the history-only CTP, if given: what the radio adds is composed - hist_only
  scene      with --scene-radius-nm: the message is scored against every airborne
             aircraft within that radius (up to --max-aircraft), they compete for it
             (addressee_v2.resolve), and the addressee's forecast is gated by its share.
             Reports how often the true addressee ranks first.

Report: counts and rejection reasons (no ADS-B match, gaps, on ground), the share of
transcript tokens outside the training vocabulary, the SBM's status split, the scores
per condition and per group, coverage, PIT and rank histograms. --per-message writes one
line per transmission with its top interpretations and per-condition scores.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..collect.aircraft import DEFAULT_CACHE
from ..data import atoms as A
from ..data.resample import WindowSpec
from ..training.tracking import Tracker, add_args
from ..inference.adsb import aircraft_near, icao_for_callsign, load_states
from ..inference.forecaster import Forecaster, describe
from ..inference.transcript import read_transmissions
from .addressee_v2 import gated_status, resolve
from .metrics_v2 import histogram, nested, pit, rank, raw_ade, summarize
from .compose_v2 import per_message

RAW_PAD = 100


def pad_raw(raw_list):
    out = np.full((len(raw_list), RAW_PAD, 4), np.nan, np.float32)
    for i, r in enumerate(raw_list):
        r = np.asarray(r)[:RAW_PAD]
        out[i, :len(r)] = r
    return out


def match(txs, states, window_s):
    """The addressee's icao24 per transmission, from icao24 or callsign."""
    out = []
    for tx in txs:
        ic = tx.icao24
        if ic is None and tx.callsign:
            ic = icao_for_callsign(states, tx.callsign, tx.t_end, window_s)
        out.append(ic)
    return out


def scene_status(fc, states, win, rows, msgs, txs, a):
    """Addressee resolution for rows: (p_addr [B], top1 [B], n_on_freq [B], q [B, 3])."""
    reqs, owner = [], []
    for j, (r, tx) in enumerate(zip(rows, txs)):
        near = aircraft_near(states, tx.t0, win.lat0[r], win.lon0[r], a.scene_radius_nm)
        near = [x for x in near if x != win.icao24[r]][:a.max_aircraft - 1]
        reqs += [(x, tx.t0) for x in near]
        owner += [j] * len(near)
    q_addr = fc.status(fc.batch(win, rows, msgs, seed_ids=win.request[rows]))
    r_other = {j: [] for j in range(len(rows))}
    if reqs:
        cw, _ = fc.windows(states, pd.DataFrame(reqs, columns=["icao24", "t0"]), future=False)
        if cw is not None:
            own = np.asarray(owner)[cw.request]
            for i in range(0, len(cw), 256):
                sl = np.arange(i, min(i + 256, len(cw)))
                q = fc.status(fc.batch(cw, sl, [msgs[own[k]] for k in sl]))
                for k, qq in zip(sl, q):
                    r_other[own[k]].append(qq[:2].sum())
    p, top, n = [], [], []
    for j in range(len(rows)):
        r = np.r_[q_addr[j, :2].sum(), r_other[j]]
        pa = resolve(r, a.out_weight, fc.addressed_prior)
        p.append(pa[0])
        top.append(bool(np.argmax(pa) == 0))
        n.append(len(r))
    return np.array(p), np.array(top), np.array(n), q_addr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transmissions", required=True)
    ap.add_argument("--adsb", nargs="+", required=True, help="files or globs")
    ap.add_argument("--source", default="auto", choices=("auto", "opensky", "readsb"))
    ap.add_argument("--sbm", required=True)
    ap.add_argument("--ctp", required=True)
    ap.add_argument("--hist-only")
    ap.add_argument("--k", type=int, default=12, help="interpretations")
    ap.add_argument("--s", type=int, default=6, help="futures per interpretation")
    ap.add_argument("--nfe", type=int, default=12)
    ap.add_argument("--method", default="euler", choices=("euler", "heun"))
    ap.add_argument("--t0-offset", type=float, default=0.0,
                    help="seconds added to t_end to get the prediction time")
    ap.add_argument("--default-conf", type=float, default=1.0,
                    help="confidence for words given without one")
    ap.add_argument("--match-window", type=float, default=300.0,
                    help="seconds around t_end to find a callsign in the ADS-B")
    ap.add_argument("--scene-radius-nm", type=float, default=0.0,
                    help="> 0: resolve the addressee among aircraft within this radius")
    ap.add_argument("--max-aircraft", type=int, default=16)
    ap.add_argument("--out-weight", type=float, default=1.0)
    ap.add_argument("--aircraft-db", default=str(DEFAULT_CACHE),
                    help="OpenSky aircraft database CSV (downloaded once if missing); "
                         "'none': every aircraft type unknown")
    ap.add_argument("--altitude", default="baro", choices=("baro", "geo"))
    ap.add_argument("--gap", type=float, default=4.0,
                    help="widest raw ADS-B gap (s) a window may contain; training used 4")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-message")
    ap.add_argument("--device", default=None)
    add_args(ap)
    a = ap.parse_args()

    tr = Tracker(a.wandb, a.wandb_project, a.run_name or Path(a.out).stem, vars(a),
                 group=a.wandb_group, job_type="eval_real")
    fc = Forecaster.load(a.sbm, a.ctp, a.hist_only, a.device,
                         aircraft_db=None if a.aircraft_db == "none" else a.aircraft_db,
                         spec=WindowSpec(altitude=a.altitude, gap_s=a.gap))
    states = load_states(a.adsb, a.source)
    txs = read_transmissions(a.transmissions)
    for tx in txs:
        tx.t0 = tx.t_end + a.t0_offset
    print(f"real - {len(txs)} transmissions, {len(states)} state vectors, "
          f"{states['icao24'].nunique()} aircraft", flush=True)

    icao = match(txs, states, a.match_window)
    reason = ["no_addressee" if ic is None else None for ic in icao]
    ok = [i for i, ic in enumerate(icao) if ic is not None]
    req = pd.DataFrame({"icao24": [icao[i] for i in ok], "t0": [txs[i].t0 for i in ok],
                        "tx": ok})
    win, report = fc.windows(states, req, future=True)
    for j, rsn in enumerate(report["reason"]):
        reason[ok[j]] = rsn if isinstance(rsn, str) else None     # accepted: None (or NaN)
    n_acc = 0 if win is None else len(win)
    print(f"real - {n_acc} windows; rejected {Counter(r for r in reason if r)}", flush=True)

    groups_ks = nested(a.k, a.s)
    scores, hist, recs, unk = {}, {}, [], [0, 0]
    status_sum, top1, p_addr, n_freq = np.zeros(3), [], [], []
    for i in range(0, n_acc, a.batch):
        rows = np.arange(i, min(i + a.batch, n_acc))
        bt = [txs[int(req["tx"].iloc[r])] for r in win.request[rows]]
        msgs = [tx.words if tx.words else (tx.text or "") for tx in bt]
        for m in msgs:
            ids, _, nu = fc.encode(m, a.default_conf)
            unk[0] += nu
            unk[1] += int((ids > 0).sum())
        b = fc.batch(win, rows, msgs, seed_ids=win.request[rows])
        y = np.cumsum(win.fut_vel[rows], axis=-2)
        raw = pad_raw([win.raw_future[r] for r in rows])
        q = fc.status(b)
        status_sum += q.sum(0)
        out, zs = {}, {}
        out["composed"], zs["composed"] = fc.compose(b, a.k, a.s, a.seed, a.nfe, a.method)
        out["known"], _ = fc.compose(b, a.k, a.s, a.seed, a.nfe, a.method,
                                     p_status=gated_status(q, np.ones(len(rows))))
        rd = fc.readings(b, top=3)
        zb = np.array([r_[0][2] for r_ in rd])
        zm = torch.as_tensor(np.repeat(zb[:, None], a.k, 1), device=fc.device)
        out["mode"] = fc.futures(b, zm, a.s, a.seed, a.nfe, a.method)
        if fc.hist_only is not None:
            out["hist_only"] = fc.history_only(b, a.k * a.s, a.seed, a.nfe, a.method)
        if a.scene_radius_nm > 0:
            pa, t1, nf, _ = scene_status(fc, states, win, rows, msgs, bt, a)
            out["scene"], _ = fc.compose(b, a.k, a.s, a.seed, a.nfe, a.method,
                                         p_status=gated_status(q, pa))
            top1 += t1.tolist()
            p_addr += pa.tolist()
            n_freq += nf.tolist()
        per = {}
        for name, V in out.items():
            X = np.cumsum(V, axis=-2)
            g = groups_ks if name in ("composed", "known", "scene") else None
            sc = summarize(X, y, groups=g)
            sc["raw_ade_h"], sc["raw_ade_v"] = raw_ade(X, raw)
            scores.setdefault(name, []).append((len(rows), sc))
            per[name] = per_message(X, y, g)
            for part, sl in (("h", slice(0, 2)), ("v", slice(2, 3))):
                hist.setdefault((name, f"pit_{part}"), []).append(pit(X[..., sl], y[..., sl], g).ravel())
                hist.setdefault((name, f"rank_{part}"), []).append(rank(X[..., sl], y[..., sl], g).ravel())
        for k, (r, tx) in enumerate(zip(rows, bt)):
            rec = {"id": tx.id, "icao24": win.icao24[r], "callsign": win.callsign[r], "t0": tx.t0,
                   "group": tx.group, "status": dict(zip(A.STATUS, np.round(q[k], 4).tolist())),
                   "readings": [(t, round(p, 4)) for t, p, _ in rd[k]],
                   "interpretations": Counter(describe(z) for z in zs["composed"][k]).most_common(4)}
            for name, pm in per.items():
                rec[name] = {m: round(float(v[k]), 1) for m, v in pm.items()}
            if a.scene_radius_nm > 0:
                rec["p_addressee"], rec["addressee_top1"] = round(p_addr[-len(rows) + k], 4), top1[-len(rows) + k]
            recs.append(rec)
        tr.log({"progress/messages": len(recs), "progress/fraction": len(recs) / max(n_acc, 1)})
        print(f"real - {len(recs)}/{n_acc}", flush=True)

    wmean = lambda rr, key: float(sum(n * s[key] for n, s in rr) / sum(n for n, _ in rr))
    res = {"n_transmissions": len(txs), "n_evaluated": n_acc,
           "rejected": dict(Counter(r for r in reason if r)),
           "unknown_token_share": unk[0] / max(unk[1], 1),
           "status_mean": dict(zip(A.STATUS, (status_sum / max(n_acc, 1)).round(4).tolist())),
           "args": vars(a), "by_condition": {}, "by_group": {}, "calibration": {}}
    for name, rr in scores.items():
        res["by_condition"][name] = {k: wmean(rr, k) for k in rr[0][1]}
    for (name, key), u in hist.items():
        res["calibration"].setdefault(name, {})[key] = histogram(np.concatenate(u)).tolist()
    if top1:
        res["addressee"] = {"top1": float(np.mean(top1)), "p_mean": float(np.mean(p_addr)),
                            "aircraft_on_frequency_mean": float(np.mean(n_freq))}
    grp = pd.DataFrame([{"group": r["group"], "cond": c, **r[c]} for r in recs
                        for c in scores if c in r])
    if len(grp) and grp["group"].notna().any():
        for (g, c), d in grp.groupby(["group", "cond"]):
            res["by_group"].setdefault(str(g), {"n": int(len(d))})[c] = {
                k: float(d[k].mean()) for k in d.columns if k not in ("group", "cond")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=str))
    if a.per_message:
        Path(a.per_message).write_text("\n".join(json.dumps(r, default=str) for r in recs) + "\n")
    if tr.run:
        tr.summary({f"{c}/{k}": v for c, s in res["by_condition"].items() for k, v in s.items()})
        tr.finish()
    print(json.dumps({k: res[k] for k in ("n_transmissions", "n_evaluated", "rejected",
                                          "unknown_token_share", "by_condition")}, indent=1))


if __name__ == "__main__":
    main()
