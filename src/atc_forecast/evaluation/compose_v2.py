"""Evaluate v2 models end to end on fixed test messages.

    python -m atc_forecast.evaluation.compose_v2 --sbm runs/v2/sbm.pt --ctp runs/v2/ctp.pt \
        [--hist-only runs/v2/ctp_none.pt] [--direct runs/v2/ctp_text.pt] \
        [--mdn runs/v2/mdn.pt] --split test --out results/v2/eval.json

Each window is paired with one fixed message (its first own-instruction draw for maneuver
windows, a no-motion or other-aircraft message otherwise, by variant), so every condition
sees the same transcripts. Conditions:

  oracle        CTP given the true Z: the label-conditioned bound
  composed      SBM samples K interpretations; CTP draws S futures for each (K x S total)
  mode          beam-search joint mode of the SBM; CTP draws K x S futures from it
  hist_only     separate history-only CTP
  direct        CTP reading the transcript directly
  kinematic     rule on the SBM mode, and on the true Z (single deterministic future)
  mdn / anchors same encoders, Gaussian-mixture head, on the true Z (_oracle) and on the
                SBM's interpretations (_composed)
  direct_init   a second transcript model (e.g. initialized from the SBM encoder)

--only restricts the conditions (for sweeps); --method picks the flow sampler.
--per-message writes one record per message (variant, SBM mode probability and margin,
whether the mode is right, distinct interpretations among the K draws, and each
condition's horizontal/vertical energy score and ADE), for the composition analysis.

Scores (metrics_v2.summarize) by condition and by message variant; per-lead coverage and
spread-skill for calibration plots; error against raw ADS-B for the main conditions.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..models.baselines import kinematic_rule, load_mdn
from ..models.ctp import load_ctp, seeded_noise
from ..models.sbm import load_sbm, window_generators
from ..training.common import pick_device
from ..training.data_v2 import MessageStore, WindowStore
from ..training.tracking import Tracker, add_args
from .metrics_v2 import (
    ade_fde,
    coverage,
    energy,
    histogram,
    nested,
    pit,
    positions_from_velocities,
    rank,
    raw_ade,
    spread_skill,
    summarize,
)


def fixed_messages(ms: MessageStore, windows, variants=("own", "maintain", "own_other", "other")):
    """One message row per (window, variant) where it exists: draw 0 of that variant."""
    m = ms.msgs
    m = m[m["window_id"].isin(windows) & m["variant"].isin(variants) & (m["draw"] == 0)]
    return m.index.to_numpy()


def ctp_samples(ctp, b, z, K, S, seed, nfe, method="euler"):
    """CTP futures for interpretations z [B, K, 4], seeded per (window, k, sample)."""
    wid = np.asarray(b["window_id"])
    shape = ctp.space.shape
    noise = torch.stack([seeded_noise(wid, S, shape, seed, k=k) for k in range(K)], 1)
    V = ctp.sample(b, S, noise=noise.reshape(-1, S, *shape), z=z, nfe=nfe, method=method)
    return V.reshape(V.shape[0], K * S, 60, 3)


@torch.no_grad()
def run_batch(models, b, a):
    """Future velocity samples [B, N, 60, 3] for every available condition."""
    out = {}
    K, S = a.k, a.s
    want = (lambda n: True) if not a.only else (lambda n: n in a.only)
    sbm, ctp = models.get("sbm"), models.get("ctp")
    zt = b["z"].clone()
    if ctp is not None and want("oracle"):
        out["oracle"] = ctp_samples(ctp, b, zt[:, None].expand(-1, K, -1), K, S, a.seed, a.nfe,
                                    a.method)
    zs = zb = None
    if sbm is not None:
        enc = sbm.encode(b)
        zs = sbm.sample(b, K, window_generators(b["window_id"], a.seed), enc=enc)
        if sbm.cfg.factorization == "mixture":     # no beam: most frequent of 4K draws
            zm = sbm.sample(b, 4 * K, window_generators(b["window_id"], a.seed, salt=2), enc=enc)
            zb = torch.stack([torch.as_tensor(max(set(map(tuple, x)), key=list(map(tuple, x)).count))
                              for x in zm.cpu().numpy()]).to(zt.device)[:, None]
            zb = torch.cat([zb, zb], 1)
            lp = torch.full(zb.shape[:2], float("nan"), device=zt.device)
            cert = torch.zeros(len(zb), dtype=torch.bool)
        else:
            zb, lp, cert = sbm.beam(b, width=a.beam, enc=enc)
        out["_certified"] = cert.cpu().numpy()
        out["_mode_exact"] = (zb[:, 0] == zt).all(1).cpu().numpy()
        out["_mode_logp"] = lp[:, 0].cpu().numpy()
        out["_margin"] = (lp[:, 0] - lp[:, 1]).cpu().numpy()
        out["_n_distinct"] = np.array([len({tuple(r) for r in x}) for x in zs.cpu().numpy()])
    if sbm is not None and ctp is not None:
        if want("composed"):
            out["composed"] = ctp_samples(ctp, b, zs, K, S, a.seed, a.nfe, a.method)
        if want("mode"):
            out["mode"] = ctp_samples(ctp, b, zb[:, :1].expand(-1, K, -1), K, S, a.seed, a.nfe,
                                      a.method)
    if sbm is not None:
        st = b["state"].cpu().numpy()
        ce = b["center"].cpu().numpy()
        if want("kinematic_mode"):
            out["kinematic_mode"] = torch.as_tensor(
                kinematic_rule(zb[:, 0].cpu().numpy(), st, ce))[:, None]
        if want("kinematic_true"):
            out["kinematic_true"] = torch.as_tensor(kinematic_rule(zt.cpu().numpy(), st, ce))[:, None]
    for name in ("hist_only", "direct", "direct_init"):
        m = models.get(name)
        if m is not None and want(name):
            noise = seeded_noise(b["window_id"], K * S, m.space.shape, a.seed)
            out[name] = m.sample(b, K * S, noise=noise, nfe=a.nfe, method=a.method)
    for name in ("mdn", "anchors"):
        m = models.get(name)
        if m is None:
            continue
        gens = lambda salt: [torch.Generator().manual_seed(hash((a.seed, int(w), salt)) & 0x7FFFFFFF)
                             for w in np.asarray(b["window_id"])]
        if want(f"{name}_oracle"):
            out[f"{name}_oracle"] = m.sample(b, K * S, generators=gens(0))
        if zs is not None and want(f"{name}_composed"):
            out[f"{name}_composed"] = torch.cat(
                [m.sample({**b, "z": zs[:, k]}, S, generators=gens(k + 1)) for k in range(K)], 1)
    return out


def per_message(X, y, groups=None):
    """Per-message horizontal (m) and vertical (ft) energy score and ADE. groups: the
    interpretation of each sample (metrics_v2.nested), so the energy score of a composed
    forecast pairs samples across interpretations only."""
    ft = 3.28084
    out = {}
    for name, sl, unit in (("h", slice(0, 2), 1.0), ("v", slice(2, 3), ft)):
        Xp, yp = X[..., sl], y[..., sl]
        out[f"ade_{name}"] = ade_fde(Xp, yp)[0] * unit
        if X.shape[1] > 1:
            out[f"energy_{name}"] = energy(Xp, yp, groups) * unit
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/v2")
    ap.add_argument("--profile", default="jacktol")
    ap.add_argument("--sbm")
    ap.add_argument("--ctp")
    ap.add_argument("--hist-only")
    ap.add_argument("--direct")
    ap.add_argument("--direct-init", help="second transcript model, e.g. SBM-initialized")
    ap.add_argument("--mdn")
    ap.add_argument("--anchors")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=5000, help="windows (fixed random subset)")
    ap.add_argument("--k", type=int, default=12, help="interpretations")
    ap.add_argument("--s", type=int, default=6, help="futures per interpretation")
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--nfe", type=int, default=12)
    ap.add_argument("--method", default="euler", choices=("euler", "heun"))
    ap.add_argument("--only", nargs="*", default=None, help="conditions to run (default all)")
    ap.add_argument("--per-message", help="write per-message records to this .npz")
    ap.add_argument("--batch", type=int, default=64)
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
    models = {}
    if a.sbm:
        models["sbm"] = load_sbm(a.sbm, dev)[0]
    for key in ("ctp", "hist_only", "direct", "direct_init"):
        if getattr(a, key):
            models[key] = load_ctp(getattr(a, key), dev)[0]
    for key in ("mdn", "anchors"):
        if getattr(a, key):
            models[key] = load_mdn(getattr(a, key), dev)[0]

    pool = ws.split(a.split)
    win = np.sort(np.random.default_rng(99).choice(pool, min(a.n, len(pool)), replace=False))
    rows = fixed_messages(ms, win)
    variant = ms.variant[rows]
    raw_rows = ws.meta["raw_row"].to_numpy()
    scores, cal, flags, pm, hist = {}, {}, {}, {}, {}
    done = 0
    wmean_ = lambda per: float(sum(n * s["energy_h"] for recs in per.values() for n, s in recs if "energy_h" in s)
                               / max(sum(n for recs in per.values() for n, s in recs if "energy_h" in s), 1))
    groups = nested(a.k, a.s)                 # interpretation of each of the K x S samples
    for v in np.unique(variant):
        rv = rows[variant == v]
        for i in range(0, len(rv), a.batch):
            r = rv[i:i + a.batch]
            b = ms.batch(r, dev)
            y = positions_from_velocities(b["fut_vel"].cpu().numpy())
            raw = ws.raw_future[raw_rows[ms.window_id[r]]]
            pm.setdefault("row", []).append(r)
            pm.setdefault("variant", []).append(np.full(len(r), v))
            pm.setdefault("window_id", []).append(ms.window_id[r])
            pm.setdefault("z_true", []).append(b["z"].cpu().numpy())
            for name, V in run_batch(models, b, a).items():
                if name.startswith("_"):
                    flags.setdefault(name, []).append(V)
                    continue
                X = positions_from_velocities(V.cpu().numpy())
                g = groups if X.shape[1] == len(groups) else None
                for k, x in per_message(X, y, g).items():
                    pm.setdefault(f"{name}/{k}", []).append(x)
                if X.shape[1] > 1:
                    sc = summarize(X, y, groups=g)
                    cal.setdefault(name, []).append((len(r), coverage(X, y, groups=g),
                                                     spread_skill(X, y)))
                    for part, sl in (("h", slice(0, 2)), ("v", slice(2, 3))):
                        hist.setdefault((name, v, f"pit_{part}"), []).append(pit(X[..., sl], y[..., sl], g).ravel())
                        hist.setdefault((name, v, f"rank_{part}"), []).append(rank(X[..., sl], y[..., sl], g).ravel())
                else:                      # deterministic rule: point errors only
                    sc = {k: x for k, x in summarize(np.repeat(X, 2, 1), y).items()
                          if k.startswith(("ade", "fde"))}
                sc["raw_ade_h"], sc["raw_ade_v"] = raw_ade(X, raw)
                scores.setdefault(name, {}).setdefault(v, []).append((len(r), sc))
            done += len(r)
            tr.log({"progress/messages": done, "progress/fraction": done / len(rows),
                    **{f"running/{c}_energy_h": wmean_(scores[c]) for c in scores if v in scores[c] and "energy_h" in scores[c][v][0][1]}})
        print(f"{v}: {len(rv)} messages done", flush=True)

    wmean = lambda recs, k: float(sum(n * s[k] for n, s in recs) / sum(n for n, _ in recs))
    res = {"n_windows": len(win), "n_messages": len(rows),
           "variants": {v: int((variant == v).sum()) for v in np.unique(variant)},
           "args": vars(a), "by_condition": {}, "calibration": {}}
    for name, per in scores.items():
        res["by_condition"][name] = {v: {k: wmean(recs, k) for k in recs[0][1]}
                                     for v, recs in per.items()}
    for name, recs in cal.items():
        n = np.array([x[0] for x in recs], float)[:, None, None]
        cov = sum(x[1] * n[i, :, :, None] for i, x in enumerate(recs)) / n.sum()
        ss = sum(x[2] * n[i] for i, x in enumerate(recs)) / n.sum()
        res["calibration"][name] = {"cov50_by_lead": cov[0].mean(-1).tolist(),
                                    "cov90_by_lead": cov[1].mean(-1).tolist(),
                                    "spread_skill_by_lead": ss.tolist()}
    # PIT (marginal, per lead and axis) and multivariate rank (whole trajectory) histograms,
    # 10 bins, each 0.1 if calibrated; per variant and pooled over variants
    for (name, v, key), u in hist.items():
        u = np.concatenate(u)
        res["calibration"].setdefault(name, {}).setdefault("hist", {}).setdefault(v, {})[key] = histogram(u).tolist()
    for name in {k[0] for k in hist}:
        for key in ("pit_h", "pit_v", "rank_h", "rank_v"):
            u = np.concatenate([np.concatenate(x) for (c, _, kk), x in hist.items() if c == name and kk == key])
            res["calibration"][name]["hist"].setdefault("all", {})[key] = histogram(u).tolist()
    if "_certified" in flags:
        res["beam_certified"] = float(np.concatenate(flags["_certified"]).mean())
        res["mode_exact_match"] = float(np.concatenate(flags["_mode_exact"]).mean())
    if a.per_message:
        recs = {k: np.concatenate(v) for k, v in pm.items()}
        recs.update({k.lstrip("_"): np.concatenate(v) for k, v in flags.items()})
        Path(a.per_message).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.per_message, **recs)
    if tr.run:
        cols = sorted({k for c in res["by_condition"].values() for v in c.values() for k in v})
        rows_t = [[c, v] + [res["by_condition"][c][v].get(k) for k in cols]
                  for c in res["by_condition"] for v in res["by_condition"][c]]
        tr.log({"eval/scores": tr.wandb.Table(columns=["condition", "variant"] + cols,
                                               data=rows_t)})
        for c, cal in res["calibration"].items():
            tr.curve(f"eval/coverage_{c}", [cal["cov50_by_lead"], cal["cov90_by_lead"]],
                     names=["50%", "90%"], title=f"coverage by lead, {c}")
        tr.summary({k: res[k] for k in ("beam_certified", "mode_exact_match") if k in res})
        tr.finish()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res["by_condition"][k] for k in res["by_condition"]}, indent=1)[:4000])


if __name__ == "__main__":
    main()
