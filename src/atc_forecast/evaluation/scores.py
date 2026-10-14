"""Proper scoring rules for predictive distributions.

Why not just a density. Endpoint KDE log-density is interpretable but it only looks at one
step, and a Gaussian KDE over the 3H dimensions of a full trajectory is degenerate with
realistic sample counts: a 180-dimensional density cannot be estimated from 72 points. The
fix is not a better density estimator but a scoring rule that needs none.

The ENERGY SCORE is strictly proper for multivariate predictive distributions, estimated
from samples alone, and valid in any dimension:

    ES(P, y) = E||X - y|| - 0.5 * E||X - X'||,    X, X' ~ P independent

LOWER is better. The first term rewards being close to the truth and the second penalises a
collapsed distribution, which is exactly the point-versus-belief trade being measured.
Reporting the two terms SEPARATELY matters: a rising energy score is ambiguous on its own,
since the samples may have moved away from the truth or the posterior may have sharpened,
and those call for opposite responses.

Point accuracy is reported too, but it cannot settle this comparison. A mean-over-samples
error penalises dispersion and an ensemble-mean error penalises multimodality, so both
favour a confident point forecast regardless of whether it is right.
"""

import numpy as np
import torch
from scipy.stats import gaussian_kde

from ..models.stage2 import last_valid_index
from ..units import ALT_SCALE_FT, FT_TO_NM, M_TO_NM, POS_SCALE_M

COV_LEVEL = 0.80
LEVELS = (50, 80, 90, 95)
Z = {50: 0.6745, 80: 1.2816, 90: 1.6449, 95: 1.9600}
COORDS = ["east_nm", "north_nm", "alt_ft"]


def to_nm(a):
    """East/north in nm and altitude in nm-equivalent, so the three axes are commensurable
    before any norm is taken. Without this the energy score is dominated by whichever axis
    has the largest normalised units, which is an arbitrary artefact of the scaling
    constants."""
    xy = a[..., :2] * POS_SCALE_M * M_TO_NM
    z = a[..., 2:3] * ALT_SCALE_FT * FT_TO_NM
    return torch.cat([xy, z], -1)


def energy_score(samples, truth):
    """Energy score per item. samples [B, K, D], truth [B, D]. Lower is better."""
    d_sy = torch.linalg.norm(samples - truth.unsqueeze(1), dim=-1).mean(1)
    d_ss = torch.cdist(samples, samples).mean((1, 2))
    return d_sy - 0.5 * d_ss


def energy_terms(samples, truth):
    """The two terms separately: (accuracy E||s-y||, dispersion E||s-s'||)."""
    return (float(torch.linalg.norm(samples - truth.unsqueeze(1), dim=-1).mean()),
            float(torch.cdist(samples, samples).mean()))


def energy_score_traj(cloud, fut_pos, fut_mask, max_pairs=64):
    """Energy score over the FULL trajectory, per window. cloud [B, K, H, 3].

    Only valid steps enter the norm, and the sum is divided by sqrt(n_valid) so windows
    with different effective horizons stay comparable, since the norm of a longer vector
    grows with its length. The repulsion term uses at most `max_pairs` sample pairs; it is
    an unbiased estimate either way and the full K^2 is wasteful for large K.
    """
    c, y = to_nm(cloud), to_nm(fut_pos)
    m = fut_mask.bool()[:, None, :, None]
    nv = fut_mask.sum(1).clamp(min=1).sqrt()[:, None]
    d_sy = (((c - y[:, None]) * m) ** 2).sum((2, 3)).sqrt() / nv
    K = c.shape[1]
    g = torch.Generator(device="cpu").manual_seed(0)
    n = min(max_pairs, K * K)
    i = torch.randint(K, (n,), generator=g).to(c.device)
    j = torch.randint(K, (n,), generator=g).to(c.device)
    d_ss = (((c[:, i] - c[:, j]) * m) ** 2).sum((2, 3)).sqrt() / nv
    return d_sy.mean(1) - 0.5 * d_ss.mean(1)


def kde_logpdf(points, truth, floor=-50.0):
    """Mean KDE log-density of the truth under the sample cloud, higher is better.

    Floored, because when the truth sits far outside a tight cloud the log-density goes to
    minus infinity and one such window would otherwise swamp the mean. Windows whose cloud
    is collapsed (singular covariance) are skipped.
    """
    p, y = points.cpu().numpy(), truth.cpu().numpy()
    lls = []
    for b in range(p.shape[0]):
        try:
            lls.append(max(float(gaussian_kde(p[b].T).logpdf(y[b])[0]), floor))
        except np.linalg.LinAlgError:
            pass
    return float(np.mean(lls)) if lls else float("nan")


def endpoint_metrics(cloud, fut_pos, fut_mask, level=COV_LEVEL):
    """avgADE (nm), vertical MAE (ft), coverage and endpoint KDE from a cloud [B, K, H, 3].

    Endpoints use `last_valid_index`, so windows with an internal coverage hole are scored
    at their real last valid step rather than at a fabricated interpolated one.
    """
    B, K, H, _ = cloud.shape
    last, has = last_valid_index(fut_mask)
    horiz = (torch.linalg.norm((cloud[..., :2] - fut_pos[:, None, :, :2]) * POS_SCALE_M, dim=-1)
             * M_TO_NM)
    denom = fut_mask[:, None].sum(-1).clamp(min=1)
    ade = (horiz * fut_mask[:, None]).sum(-1) / denom
    vert = ((cloud[..., 2] - fut_pos[:, None, :, 2]).abs() * ALT_SCALE_FT)
    vmae = (vert * fut_mask[:, None]).sum(-1) / denom
    eps = cloud[torch.arange(B)[:, None], torch.arange(K)[None], last[:, None]]
    y = fut_pos[torch.arange(B), last]
    lo = torch.quantile(eps, (1 - level) / 2, dim=1)
    hi = torch.quantile(eps, (1 + level) / 2, dim=1)
    return {"avgADE_nm": float(ade.mean()), "minADE_nm": float(ade.min(1).values.mean()),
            "vMAE_ft": float(vmae.mean()),
            f"cov{int(level * 100)}": float(((y >= lo) & (y <= hi)).float().mean()),
            "kde": kde_logpdf(to_nm(eps)[has], to_nm(y)[has]),
            "energy_endpoint": float(energy_score(to_nm(eps)[has], to_nm(y)[has]).mean()),
            "n": int(has.sum())}


def crps_profile(cloud, fut_pos, fut_mask, bins=6):
    """Per-lead-time horizontal CRPS (nm) and coverage, in equal slices of the horizon.

    This is the metric that shows WHERE a forecast goes wrong. An endpoint-only number
    cannot tell a forecast that is honest throughout from one that is over-confident early
    and over-dispersed late, which averages out to something that looks fine.
    """
    c, y = to_nm(cloud), to_nm(fut_pos)
    B, K, H, _ = c.shape
    err = torch.linalg.norm(c[..., :2] - y[:, None, :, :2], dim=-1)
    lo = torch.quantile(c, (1 - COV_LEVEL) / 2, dim=1)
    hi = torch.quantile(c, (1 + COV_LEVEL) / 2, dim=1)
    inside = ((y >= lo) & (y <= hi)).float().mean(-1)
    out, edges = [], np.linspace(0, H, bins + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        w = fut_mask[:, a:b]
        n = w.sum()
        if n == 0:
            continue
        e1 = (err[:, :, a:b] * w[:, None]).sum() / (n * K)
        any_valid = fut_mask[:, a:b].any(1)
        e2 = ((torch.cdist(c[:, :, a:b, :2].reshape(B, K, -1),
                           c[:, :, a:b, :2].reshape(B, K, -1)).mean((1, 2))
               * any_valid.float()).sum() / max(float(any_valid.sum()), 1))
        out.append({"lead_s": f"{a}-{b}",
                    "crps_nm": float(e1 - 0.5 * e2 / max(b - a, 1)),
                    "mae_nm": float(e1),
                    "cov80": float((inside[:, a:b] * w).sum() / n),
                    "n_steps": int(n)})
    return out


def pit_values(cloud, fut_pos, fut_mask, axis=2):
    """PIT values, the truth's rank among the samples in [0, 1], over every valid step.

    A calibrated forecast gives a UNIFORM histogram. A U shape means over-confident, with
    the truth too often in the tails, and a central hump means over-dispersed.
    """
    c, y = to_nm(cloud), to_nm(fut_pos)
    r = (c[..., axis] < y[:, None, :, axis]).float().mean(1)
    return r[fut_mask.bool()].detach().cpu().numpy()


def summarize(cloud, fut_pos, fut_mask):
    """Full-path energy score, endpoint metrics and the lead-time profile in one call."""
    es = energy_score_traj(cloud, fut_pos, fut_mask)
    _, has = last_valid_index(fut_mask)
    return {"energy_traj": float(es[has].mean()),
            **endpoint_metrics(cloud, fut_pos, fut_mask),
            "profile": crps_profile(cloud, fut_pos, fut_mask)}
