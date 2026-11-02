"""Resampled windows to model features: the track frame, the centering, the targets.

Frame. Each window is rotated so the aircraft's current track (from the history side, at
t0) points along +x ("along"), with +y to the right ("right", positive for a right turn)
and z up. The same turn then looks the same on any heading; the absolute track enters as
context. Positions are relative to the present p0, so nothing absolute leaks in except
through the context vector, on purpose.

Targets. The future is 60 one-second velocities v_k = p_k - p_{k-1} (p_0 = 0, the present).
Defined as increments, they integrate back to the positions exactly: p_k = sum_{j<=k} v_j.
The model predicts residuals against the centering velocity c = (gs_0, 0, vz_0), the
current velocity in the track frame:

    R_k = v_k - c,      k = 1 ... 60,   R in R^{60 x 3}

so "keep flying as now" is R = 0, and every model shares the same c.

Scales. Inputs are divided by fixed physical constants, following units.py: dataset
statistics would move when data is added and would not transfer to another airspace. The
one exception is the per-step target scale s_k used for the flow's loss, which is a
property of the training targets and is saved with the dataset (manifest.json) so a model
always carries the scale it was trained with.
"""

import numpy as np

from ..units import M_TO_FT, MS_TO_FPM, MS_TO_KT

POS_M = 1000.0        # history position (m)
VEL_MS = 100.0        # horizontal velocity (m/s), about 194 kt
VZ_MS = 10.0          # vertical velocity (m/s), about 1,970 fpm
ACC_MS2 = 3.0         # acceleration (m/s^2)
DZ_M = 300.0          # relative altitude (m), about 1,000 ft
ALT_M = 10_000.0      # absolute altitude (m), about 33,000 ft

HIST_FEATURES = ["x", "y", "dz", "vx", "vy", "vz", "ax", "ay", "az"]
CTX_FEATURES = ["alt", "sin_trk", "cos_trk", "gs", "vz"]


def rotation(psi):
    """ENU (east, north) to track frame (along, right) for track angle psi (rad, from
    north, clockwise). Returns [B, 2, 2]."""
    s, c = np.sin(psi), np.cos(psi)
    return np.stack([np.stack([s, c], -1), np.stack([c, -s], -1)], -2)


def to_track(xyz, psi):
    """Rotate [B, T, 3] ENU vectors into the track frame of each window."""
    R = rotation(psi)
    h = np.einsum("bij,btj->bti", R, xyz[..., :2])
    return np.concatenate([h, xyz[..., 2:3]], -1)


def make_features(w: dict) -> dict:
    """Model inputs and targets from resample() output.

    hist     [B, 30, 9]  x, y, dz (track frame, relative to p0), velocities, accelerations
    ctx      [B, 5]      absolute altitude, sin/cos track, ground speed, vertical rate
    center   [B, 3]      c = (gs_0, 0, vz_0) in m/s
    fut_vel  [B, 60, 3]  v_k in the track frame (m/s); targets are fut_vel - center
    state    dict of the present in aviation units, for the instruction tokens
    """
    v0 = w["hist_vel"][:, -1]
    psi = np.arctan2(v0[:, 0], v0[:, 1])
    gs0 = np.hypot(v0[:, 0], v0[:, 1])
    hp, hv, ha = (to_track(w[k], psi) for k in ("hist_pos", "hist_vel", "hist_acc"))
    fp = to_track(w["fut_pos"], psi)
    fv = np.diff(np.concatenate([np.zeros_like(fp[:, :1]), fp], 1), axis=1)  # increments
    hist = np.concatenate([hp[..., :2] / POS_M, hp[..., 2:] / DZ_M,
                           hv[..., :2] / VEL_MS, hv[..., 2:] / VZ_MS, ha / ACC_MS2], -1)
    ctx = np.stack([w["z0"] / ALT_M, np.sin(psi), np.cos(psi), gs0 / VEL_MS,
                    v0[:, 2] / VZ_MS], 1)
    center = np.stack([gs0, np.zeros_like(gs0), v0[:, 2]], 1)
    raw = []
    for k, r in enumerate(w["raw_future"]):
        rr = to_track(r[None, :, 1:], psi[k:k + 1])[0]
        raw.append(np.concatenate([r[:, :1], rr], 1))
    return {
        "hist": hist.astype(np.float32), "ctx": ctx.astype(np.float32),
        "center": center.astype(np.float32), "fut_vel": fv.astype(np.float32),
        "raw_future": raw,
        "state": {"trk_deg": np.degrees(psi) % 360, "alt_ft": w["z0"] * M_TO_FT,
                  "gs_kt": gs0 * MS_TO_KT, "vr_fpm": v0[:, 2] * MS_TO_FPM},
    }


def positions(fut_vel):
    """Integrate 1 s velocity increments to positions relative to the present."""
    return np.cumsum(fut_vel, axis=-2)


def residuals(fut_vel, center):
    return fut_vel - center[:, None, :]


def cumulative_turn(fut_vel):
    """Signed cumulative track change (deg) along the future, right positive, unwrapped so
    a 270 deg left turn reads -270 and not +90."""
    ang = np.arctan2(fut_vel[..., 1], fut_vel[..., 0])
    return np.degrees(np.unwrap(ang, axis=-1))


def step_scale(R, floor=0.05):
    """Per-step, per-axis std of the residual targets over a training set, [60, 3]. The
    floor keeps the first steps, where residuals are nearly zero, from blowing up."""
    return np.maximum(R.std(0), floor).astype(np.float32)


def pca_basis(R, scale, k=(4, 4, 8)):
    """Per-axis PCA of the scaled residual targets: the basis for the coefficient models.

    Returns mean [60, 3] and a list of three [k_a, 60] component matrices, and the fraction
    of variance each axis keeps.
    """
    Rs = R / scale[None]
    mean = Rs.mean(0)
    comps, kept = [], []
    for a, ka in enumerate(k):
        X = Rs[..., a] - mean[None, :, a]
        _, s, vt = np.linalg.svd(X, full_matrices=False)
        comps.append(vt[:ka].astype(np.float32))
        kept.append(float((s[:ka] ** 2).sum() / (s ** 2).sum()))
    return mean.astype(np.float32), comps, kept
