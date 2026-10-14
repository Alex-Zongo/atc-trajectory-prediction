"""Evaluate and recalibrate the Stage-2 flow predictor.

Recalibration inflates the sample spread about the ensemble mean by a factor lambda, fitted
to match empirical coverage to nominal on held-out data.

Lambda must be PER LEAD TIME, not a scalar. A single factor per coordinate can only rescale
the whole path about its mean, so it cannot fix a forecast whose dispersion has the wrong
SHAPE over the horizon, and ours does: measured with a fitted scalar lambda applied,
coverage ran around 0.60 in the first ten seconds and 0.94 in the last ten, badly
over-confident early and over-dispersed late, averaging out to something that looks fine.
Fitting per bin and interpolating between bin centres fixes the calibration without
introducing a visible kink in the sampled trajectories, which a piecewise-constant
expansion does (a measured 4.5x jump in one second is not a plausible trajectory even when
per-step coverage is unaffected).

Fit on a DISJOINT split from the one used to report, or the correction is graded on its own
data, and fit at the SAME guidance it will be applied at, because guidance and lambda both
act on the spread.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..models.stage2 import (N_CTRL, WindowDataset, deadreckon, drop_aircraft,
                             last_valid_index)
from ..units import ALT_SCALE_FT, M_TO_NM, POS_SCALE_M
from .scores import COORDS, LEVELS, endpoint_metrics, energy_score, kde_logpdf, to_nm


@torch.no_grad()
def sample_cloud(model, batch, k, n_steps, guidance=1.0, lam=None):
    """Sample k trajectories and dead-reckon them to positions [B, k, H, 3].

    `lam` ([H, 3] or a scalar) inflates the spread about the ensemble mean.
    """
    H = batch["fut_ctrl"].shape[1]
    onset_alt = batch["past"][:, -1, 2] * ALT_SCALE_FT
    ctrl = model.sample(batch, n_steps=n_steps, k=k, guidance=guidance).view(-1, k, H, N_CTRL)
    B = ctrl.shape[0]
    pos = deadreckon(ctrl.reshape(B * k, H, N_CTRL),
                     onset_alt.repeat_interleave(k)).view(B, k, H, 3)
    if lam is None:
        return pos
    mean = pos.mean(1, keepdim=True)
    return mean + lam * (pos - mean)


@torch.no_grad()
def point_eval(model, loader, device, k=5, max_batches=8, n_steps=10) -> dict:
    """Point metrics plus the clearance and aircraft ablations.

    best-of-k is mode COVERAGE (is any sample close) while avg-of-k is typical SAMPLE
    quality, which penalises over-dispersion. Both are reported because neither alone
    distinguishes a good distribution from a lucky one.

    The clearance gain is also split by whether the clearance carries any clause, since the
    aggregate is diluted by steady windows where the past trivially determines the future.
    """
    model.eval()
    acc = {k_: 0.0 for k_ in ("ade", "fde", "vert", "avg_ade", "ade_nodir", "vert_nodir",
                              "ade_noac", "vert_noac")}
    acc["n"] = 0
    strat = dict(g_man=0.0, gv_man=0.0, g_std=0.0, n_man=0, n_std=0)

    def metrics(b, guidance):
        pos = sample_cloud(model, b, k, n_steps, guidance)
        last, _ = last_valid_index(b["fut_mask"])
        fm = b["fut_mask"]
        B = pos.shape[0]
        horiz = (torch.linalg.norm((pos[..., :2] - b["fut_pos"][:, None, :, :2]) * POS_SCALE_M,
                                   dim=-1) * M_TO_NM)
        vert = ((pos[..., 2] - b["fut_pos"][:, None, :, 2]).abs() * ALT_SCALE_FT)
        d = fm[:, None].sum(-1).clamp(min=1)
        ade = (horiz * fm[:, None]).sum(-1) / d
        vmae = (vert * fm[:, None]).sum(-1) / d
        best = ade.argmin(1)
        ar = torch.arange(B, device=ade.device)
        return {"min_ade": ade[ar, best], "avg_ade": ade.mean(1),
                "min_fde": horiz[ar, best, last], "vert": vmae[ar, best]}

    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        batch = {kk: v.to(device) for kk, v in batch.items()}
        c = metrics(batch, 1.0)
        n0 = metrics(batch, 0.0)
        na = metrics(drop_aircraft(batch), 1.0)
        acc["ade"] += float(c["min_ade"].mean())
        acc["fde"] += float(c["min_fde"].mean())
        acc["vert"] += float(c["vert"].mean())
        acc["avg_ade"] += float(c["avg_ade"].mean())
        acc["ade_nodir"] += float(n0["min_ade"].mean())
        acc["vert_nodir"] += float(n0["vert"].mean())
        acc["ade_noac"] += float(na["min_ade"].mean())
        acc["vert_noac"] += float(na["vert"].mean())
        acc["n"] += 1
        d = batch["directive"]
        man = (d[:, 0] > 0.5) | (d[:, 4] > 0.5) | (d[:, 7] > 0.5)
        gl = n0["min_ade"] - c["min_ade"]
        gv = n0["vert"] - c["vert"]
        strat["g_man"] += float(gl[man].sum())
        strat["gv_man"] += float(gv[man].sum())
        strat["g_std"] += float(gl[~man].sum())
        strat["n_man"] += int(man.sum())
        strat["n_std"] += int((~man).sum())

    if acc["n"] == 0:
        # No batches ran, which happens when the validation split is smaller than one
        # batch and the loader drops the remainder. Returning zeros would read as a
        # perfect forecast, so fail loudly instead.
        raise ValueError(
            "point_eval saw no batches. The validation loader yielded nothing, most likely "
            "because drop_last=True and the split is smaller than one batch. Reduce "
            "--batch-size or use more data.")
    n = acc["n"]
    nm, ns = max(strat["n_man"], 1), max(strat["n_std"], 1)
    return {"minADE_nm": acc["ade"] / n, "minFDE_nm": acc["fde"] / n,
            "vert_ft": acc["vert"] / n, "avgADE_nm": acc["avg_ade"] / n,
            "minADE_nodir_nm": acc["ade_nodir"] / n,
            "clearance_gain_nm": (acc["ade_nodir"] - acc["ade"]) / n,
            "clearance_gain_vert_ft": (acc["vert_nodir"] - acc["vert"]) / n,
            "aircraft_gain_nm": (acc["ade_noac"] - acc["ade"]) / n,
            "aircraft_gain_vert_ft": (acc["vert_noac"] - acc["vert"]) / n,
            "clearance_gain_maneuver_nm": strat["g_man"] / nm,
            "clearance_gain_maneuver_vert_ft": strat["gv_man"] / nm,
            "clearance_gain_steady_nm": strat["g_std"] / ns,
            "frac_maneuver": strat["n_man"] / max(strat["n_man"] + strat["n_std"], 1), "k": k}


@torch.no_grad()
def coverage_eval(model, loader, device, k=12, max_batches=4, n_steps=10) -> dict:
    """The uncertainty headline: is the truth inside the predictive distribution, and does
    the clearance make it fit better? Needs a larger k for a stable endpoint KDE, so it runs
    on fewer batches than the point metrics."""
    model.eval()
    acc = dict(e_dir=0.0, e_nod=0.0, k_dir=0.0, k_nod=0.0, n=0)
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        batch = {kk: v.to(device) for kk, v in batch.items()}
        for tag, guidance in (("dir", 1.0), ("nod", 0.0)):
            pos = sample_cloud(model, batch, k, n_steps, guidance)
            last, has = last_valid_index(batch["fut_mask"])
            B, K = pos.shape[:2]
            ep = to_nm(pos[torch.arange(B)[:, None], torch.arange(K)[None], last[:, None]])
            y = to_nm(batch["fut_pos"][torch.arange(B), last])
            acc[f"e_{tag}"] += float(energy_score(ep[has], y[has]).mean())
            acc[f"k_{tag}"] += kde_logpdf(ep[has], y[has])
        acc["n"] += 1
    if acc["n"] == 0:
        raise ValueError(
            "coverage_eval saw no batches; see point_eval for why and how to fix it.")
    n = acc["n"]
    return {"energy_dir": acc["e_dir"] / n, "energy_nodir": acc["e_nod"] / n,
            "energy_gain": (acc["e_nod"] - acc["e_dir"]) / n,
            "kde_dir": acc["k_dir"] / n, "kde_nodir": acc["k_nod"] / n,
            "kde_gain": (acc["k_dir"] - acc["k_nod"]) / n, "cov_k": k}


@torch.no_grad()
def fit_lambda(model, loader, device, k=16, max_batches=6, n_steps=10, grid=None,
               levels=None, guidance=1.0, n_bins=6) -> dict:
    """Fit the per-coordinate spread inflation that best matches coverage to nominal.

    `levels` defaults to the whole curve; pass a single level to target that coverage
    exactly, so a metric reported at that level lands on nominal. `n_bins` above 1 fits
    separately per lead-time bin and returns a profile, which is what should be used.

    The profile fit needs a wider ceiling than a scalar fit, because the earliest bin can
    pin at the top of the grid: even large inflation does not reach nominal coverage that
    early.
    """
    lv = LEVELS if levels is None else levels
    if grid is None:
        grid = np.linspace(1.0, 4.0, 31) if n_bins == 1 else np.linspace(1.0, 10.0, 91)
    model.eval()
    S, Y, M = [], [], []
    for b, batch in enumerate(loader):
        if b >= max_batches:
            break
        batch = {kk: v.to(device) for kk, v in batch.items()}
        pos = sample_cloud(model, batch, k, n_steps, guidance)
        scl = torch.tensor([POS_SCALE_M * M_TO_NM, POS_SCALE_M * M_TO_NM, ALT_SCALE_FT],
                           device=device)
        S.append((pos * scl).cpu())
        Y.append((batch["fut_pos"] * scl).cpu())
        M.append(batch["fut_mask"].bool().cpu())
    if not S:
        raise ValueError(
            "fit_lambda collected no samples. The loader yielded nothing, most likely "
            "because drop_last=True and the fit split is smaller than one batch.")
    S, Y, M = torch.cat(S), torch.cat(Y), torch.cat(M)
    H = S.shape[2]
    edges = np.linspace(0, H, n_bins + 1).astype(int)

    def best(s, y, m):
        if not bool(m.any()):
            return 1.0
        mean = s.mean(1, keepdim=True)
        dev = s - mean
        bl, be = 1.0, float("inf")
        for lam in grid:
            si = mean + float(lam) * dev
            err = 0.0
            for L in lv:
                lo = torch.quantile(si, (1 - L / 100) / 2, dim=1)
                hi = torch.quantile(si, (1 + L / 100) / 2, dim=1)
                err += abs(float(((y >= lo) & (y <= hi))[m].float().mean()) - L / 100)
            if err < be:
                bl, be = float(lam), err
        return bl

    out = {"inflate": {c: best(S[..., i], Y[..., i], M) for i, c in enumerate(COORDS)},
           "levels": list(lv), "guidance": guidance, "k": k}
    if n_bins > 1:
        out["inflate_profile"] = {
            c: [best(S[:, :, a:b, i], Y[:, a:b, i], M[:, a:b])
                for a, b in zip(edges[:-1], edges[1:])] for i, c in enumerate(COORDS)}
        out["profile_edges"] = edges.tolist()
    return out


def lam_per_step(recal: dict, H: int, device, smooth: bool = True) -> torch.Tensor:
    """Expand a fitted recalibration into the per-step [H, 3] inflation tensor.

    Accepts either a scalar fit, broadcast over the horizon, or a lead-time profile.
    `smooth` interpolates between bin CENTRES rather than expanding piecewise-constant,
    which keeps the same calibration and removes the step change at each bin edge.
    """
    if "inflate_profile" not in recal:
        v = torch.tensor([recal["inflate"][c] for c in COORDS], device=device)
        return v[None, :].expand(H, 3).contiguous()
    edges = recal["profile_edges"]
    centres = np.array([0.5 * (a + b) for a, b in zip(edges[:-1], edges[1:])], float)
    steps = np.arange(H, dtype=float)
    out = torch.ones(H, 3, device=device)
    for ci, c in enumerate(COORDS):
        vals = np.asarray(recal["inflate_profile"][c], float)
        if smooth:
            v = np.interp(steps, centres, vals)
        else:
            v = np.empty(H)
            for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
                v[a:b] = vals[j]
        out[:, ci] = torch.as_tensor(v, dtype=out.dtype, device=device)
    return out


def main():
    from ..models.stage2 import load_stage2
    from ..training.common import pick_device

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="window directory")
    ap.add_argument("--out", default=None, help="write the fitted lambda to this JSON")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n-bins", type=int, default=6, help="lead-time bins; 1 fits a scalar")
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--level", type=int, default=None,
                    help="target this coverage level exactly instead of the whole curve")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = pick_device()
    model = load_stage2(a.model, device)
    npz = np.load(Path(a.data) / "windows.npz")
    wm = pd.read_parquet(Path(a.data) / "windows_meta.parquet")
    # disjoint leg halves: fit on one, report on the other
    legs = np.asarray(wm["leg_id"].unique(), dtype=object)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(legs)
    fit_legs = set(legs[: len(legs) // 2])
    in_fit = wm["leg_id"].isin(fit_legs).values
    fit_idx, eval_idx = np.where(in_fit)[0], np.where(~in_fit)[0]
    if min(len(fit_idx), len(eval_idx)) == 0:
        raise SystemExit("the leg split left one half empty; this needs more legs")
    dl = torch.utils.data.DataLoader

    def loader(ix):
        # clamped, because with drop_last a split smaller than one batch yields nothing
        return dl(WindowDataset(npz, wm, ix), min(a.batch_size, len(ix)), shuffle=False,
                  drop_last=True)

    fit_loader, eval_loader = loader(fit_idx), loader(eval_idx)
    print(f"fit on {len(fit_idx)} windows, report on {len(eval_idx)} (disjoint legs)")

    recal = fit_lambda(model, fit_loader, device, k=a.k, max_batches=a.batches,
                       n_steps=a.n_steps, n_bins=a.n_bins,
                       levels=(a.level,) if a.level else None)
    print("lambda " + "  ".join(f"{c}={v:.2f}" for c, v in recal["inflate"].items()))
    if "inflate_profile" in recal:
        for c, vals in recal["inflate_profile"].items():
            print(f"  {c:<9} " + " ".join(f"{v:.2f}" for v in vals))

    H = npz["fut_ctrl"].shape[1]
    lam = lam_per_step(recal, H, device)
    print(f"\n{'':<12}{'cov80':>8} {'kde':>9} {'avgADE':>8}")
    for tag, L in (("raw", None), ("calibrated", lam)):
        rows = []
        for b, batch in enumerate(eval_loader):
            if b >= a.batches:
                break
            batch = {kk: v.to(device) for kk, v in batch.items()}
            cloud = sample_cloud(model, batch, a.k, a.n_steps, 1.0, L)
            rows.append(endpoint_metrics(cloud, batch["fut_pos"], batch["fut_mask"]))
        m = pd.DataFrame(rows).mean()
        print(f"  {tag:<10}{m['cov80']:>8.3f} {m['kde']:>+9.2f} {m['avgADE_nm']:>8.4f}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"model": a.model, "data": a.data, "fit": recal},
                                          indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
