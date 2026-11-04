"""Scores for sampled futures, in positions (metres, track frame).

All functions take samples X [B, S, 60, 3] and truth y [B, 60, 3], positions relative to
the present. Horizontal and vertical are scored separately (their units and errors differ
by orders of magnitude).

Nested samples. A composed forecast draws K interpretations and S futures for each, so
its K x S samples are not independent: the S futures of one interpretation share it.
`groups` [S_total] gives each sample's interpretation (nested(K, S) for that layout).
With groups, the energy score takes pairs only across interpretations, and the
calibration checks use sub-ensembles of one future per interpretation; both are then
exact for the mixture. Pass the same groups for every K x S condition (samples that are
independent anyway lose nothing but a little precision), so all are scored alike.

  ade / fde          error of the sample mean, averaged over steps / at the last step
  min_ade            best of k samples (one per interpretation with groups)
  energy             energy score (Gneiting & Raftery, 2007), a proper score for the whole
                     trajectory, unbiased form
  variogram          variogram score of order 0.5 (Scheuerer & Hamill, 2015), sensitive to
                     the correlation between lead times that the energy score misses
  pit                randomized rank of the truth among the samples, per lead and axis
                     (uniform if marginally calibrated)
  coverage           share of truths inside the central 50/90 % interval, per lead and
                     axis, from the PIT, so its calibrated value is the nominal level for
                     any ensemble size
  coverage_quantile  the same from empirical sample quantiles (the earlier definition;
                     its calibrated value with S samples is level * (S - 1) / (S + 1))
  rank               multivariate rank of the whole trajectory: the truth's outlyingness
                     (mean distance to the ensemble) among truth + members, randomized;
                     uniform if jointly calibrated, mass near 1 if over-confident
                     (Thorarinsdottir, Scheuerer & Heinz, 2016)
  spread_skill       mean sample std over RMSE of the mean, per lead (1 if calibrated)
"""

import numpy as np

H_IDX, V_IDX = slice(0, 2), slice(2, 3)


def nested(K, S):
    """Interpretation index of each of K x S samples laid out k-major."""
    return np.repeat(np.arange(K), S)


def subensembles(n, groups=None):
    """Index arrays of independent sub-ensembles: member j of every interpretation."""
    if groups is None or len(np.unique(groups)) < 2:
        return [np.arange(n)]
    g = np.asarray(groups)
    members = [np.where(g == k)[0] for k in np.unique(g)]
    S = min(len(m) for m in members)
    return [np.array([m[j] for m in members]) for j in range(S)]


def _parts(X, y):
    return {"h": (X[..., H_IDX], y[..., H_IDX]), "v": (X[..., V_IDX], y[..., V_IDX])}


def ade_fde(X, y):
    m = X.mean(1)
    d = np.linalg.norm(m - y, axis=-1)
    return d.mean(1), d[:, -1]


def min_ade(X, y, k=6, groups=None):
    idx = subensembles(X.shape[1], groups)[0][:k]
    d = np.linalg.norm(X[:, idx] - y[:, None], axis=-1).mean(-1)
    return d.min(1)


def energy(X, y, groups=None):
    B, S = X.shape[:2]
    Xf, yf = X.reshape(B, S, -1), y.reshape(B, -1)
    t1 = np.linalg.norm(Xf - yf[:, None], axis=-1).mean(1)
    G = np.einsum("bsd,btd->bst", Xf, Xf)
    n = np.einsum("bss->bs", G)
    D = np.sqrt(np.clip(n[:, :, None] + n[:, None, :] - 2 * G, 0, None))
    if groups is None or len(np.unique(groups)) < 2:
        pair = ~np.eye(S, dtype=bool)
    else:                               # pairs across interpretations only
        g = np.asarray(groups)
        pair = g[:, None] != g[None, :]
    t2 = D[:, pair].mean(1)
    return t1 - 0.5 * t2


def variogram(X, y, p=0.5, every=5):
    """Over lead times every `every` s, each axis, summed over axes."""
    Xs, ys = X[:, :, every - 1::every], y[:, every - 1::every]
    out = 0.0
    for a in range(X.shape[-1]):
        vy = np.abs(ys[:, :, None, a] - ys[:, None, :, a]) ** p
        vx = (np.abs(Xs[:, :, :, None, a] - Xs[:, :, None, :, a]) ** p).mean(1)
        out = out + ((vy - vx) ** 2).sum((1, 2))
    return out


def pit(X, y, groups=None, seed=0):
    """Randomized PIT [J, B, 60, 3] in [0, 1], one per independent sub-ensemble."""
    rng = np.random.default_rng(seed)
    out = []
    for idx in subensembles(X.shape[1], groups):
        Xs = X[:, idx]
        below = (Xs < y[:, None]).sum(1)
        ties = (Xs == y[:, None]).sum(1)
        out.append((below + rng.uniform(0, 1, below.shape) * (ties + 1)) / (len(idx) + 1))
    return np.stack(out)


def coverage(X, y, levels=(0.5, 0.9), groups=None):
    """[len(levels), 60, 3] share of truths inside the central interval, from the PIT."""
    u = pit(X, y, groups)
    return np.array([(np.abs(u - 0.5) <= lv / 2).mean((0, 1)) for lv in levels])


def coverage_quantile(X, y, levels=(0.5, 0.9)):
    """[len(levels), 60, 3] from empirical quantiles of all samples (earlier definition)."""
    out = []
    for lv in levels:
        lo = np.quantile(X, (1 - lv) / 2, axis=1)
        hi = np.quantile(X, 1 - (1 - lv) / 2, axis=1)
        out.append(((y >= lo) & (y <= hi)).mean(0))
    return np.array(out)


def rank(X, y, groups=None, seed=0):
    """Multivariate rank [J, B] of the whole trajectory, randomized, in [0, 1]; 1 = the
    truth is more outlying than every member (average-distance pre-rank)."""
    rng = np.random.default_rng(seed + 1)
    B = X.shape[0]
    out = []
    for idx in subensembles(X.shape[1], groups):
        P = np.concatenate([y[:, None], X[:, idx]], 1).reshape(B, len(idx) + 1, -1)
        D = np.linalg.norm(P[:, :, None] - P[:, None], axis=-1)
        rho = D.sum(2) / len(idx)                       # mean distance to the others
        lt = (rho[:, 1:] < rho[:, :1]).sum(1)
        eq = (rho[:, 1:] == rho[:, :1]).sum(1)
        out.append((lt + rng.uniform(0, 1, B) * (eq + 1)) / (len(idx) + 1))
    return np.stack(out)


def histogram(u, bins=10):
    """Share of values per equal bin of [0, 1] (each 1 / bins if calibrated)."""
    h = np.histogram(np.asarray(u).ravel(), bins=bins, range=(0, 1))[0]
    return h / max(h.sum(), 1)


def spread_skill(X, y):
    """[60, 3]: RMS of the sample std over the RMSE of the sample mean."""
    sd = X.std(1, ddof=1)
    err = X.mean(1) - y
    return np.sqrt((sd ** 2).mean(0)) / np.sqrt((err ** 2).mean(0)).clip(1e-9)


def summarize(X, y, k=6, groups=None):
    """Mean scores per batch; horizontal in m, vertical in ft. Calibration: cov50 / cov90
    from the PIT (calibrated = 0.5 / 0.9), cov90_q from sample quantiles (earlier
    definition), rank_out10_h / _v: share of truths in the most outlying tenth of the
    multivariate rank (calibrated = 0.1)."""
    ft = 3.28084
    out = {}
    for name, (Xp, yp) in _parts(X, y).items():
        unit = ft if name == "v" else 1.0
        a, f = ade_fde(Xp, yp)
        out[f"ade_{name}"] = float(a.mean() * unit)
        out[f"fde_{name}"] = float(f.mean() * unit)
        out[f"minade{k}_{name}"] = float(min_ade(Xp, yp, k, groups).mean() * unit)
        out[f"energy_{name}"] = float(energy(Xp, yp, groups).mean() * unit)
        out[f"rank_out10_{name}"] = float((rank(Xp, yp, groups) > 0.9).mean())
    out["variogram"] = float(variogram(X, y).mean())
    cov = coverage(X, y, groups=groups)
    out["cov50"], out["cov90"] = float(cov[0].mean()), float(cov[1].mean())
    out["cov90_q"] = float(coverage_quantile(X, y)[1].mean())
    return out


def positions_from_velocities(v):
    return np.cumsum(v, axis=-2)


def at_raw_times(P, raw):
    """Predicted positions [B, S, 60, 3] read at raw future times; raw [B, R, 4] (t, x, y,
    z) NaN padded. Returns [B, S, R, 3], linear between the 1 Hz points, 0 at t0."""
    B, S = P.shape[:2]
    full = np.concatenate([np.zeros((B, S, 1, 3)), P], 2)
    t = np.nan_to_num(raw[..., 0], nan=0.0).clip(0, 60)
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, 60)
    w = (t - lo)[:, None, :, None]
    g = lambda j: np.take_along_axis(full, np.broadcast_to(j[:, None, :, None], (B, S, j.shape[1], 3)), 2)
    return g(lo) * (1 - w) + g(hi) * w


def raw_ade(P, raw):
    """Error of the sample mean against raw ADS-B points (m horizontal, ft vertical)."""
    Q = at_raw_times(P, raw).mean(1)
    ok = ~np.isnan(raw[..., 0])
    dh = np.linalg.norm(Q[..., :2] - raw[..., 1:3], axis=-1)
    dv = np.abs(Q[..., 2] - raw[..., 3]) * 3.28084
    return float(dh[ok].mean()), float(dv[ok].mean())
