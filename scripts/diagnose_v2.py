"""Diagnose under-dispersion and check the SBM, one hypothesis per sub-command.

    python scripts/diagnose_v2.py nfe       --ctp runs/v2/ctp.pt          # H2 sampler
    python scripts/diagnose_v2.py overfit   --ctp runs/v2/ctp.pt          # H3
    python scripts/diagnose_v2.py time      --ctp a.pt b.pt               # H4 flow time
    python scripts/diagnose_v2.py hindsight --ctp runs/v2/ctp.pt          # H5
    python scripts/diagnose_v2.py noise     --ctp runs/v2/ctp.pt          # H1 noise floor
    python scripts/diagnose_v2.py sbm       --sbm runs/v2/sbm.pt          # order, beam

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
"""

import argparse
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("nfe", "overfit", "time", "hindsight", "noise", "sbm"))
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
    out = Path(a.out_dir) / f"diagnose_{a.command}"
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
