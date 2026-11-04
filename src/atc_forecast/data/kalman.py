"""Batched constant-acceleration Kalman filter and RTS smoother on irregular samples.

ADS-B arrives irregularly (median spacing 1.0 s, with drop-outs), and raw 1 s position
differences scatter by about 16 kt around the reported speed. Linear interpolation onto a
grid keeps that noise and draws straight lines through gaps. A Kalman filter does both
jobs at once: it fuses the position and the separately broadcast velocity, and its motion
model bridges a short gap with a smooth arc instead of a kink.

One model per axis, state x = [position, velocity, acceleration], white-jerk process noise.
Many independent sequences are processed together: sequence b is a padded row of raw
samples, and padding steps have dt = 0 and no measurement, which makes them exact no-ops in
both passes.

    filter_batch   forward pass; the estimate at sample i uses samples <= i only (causal)
    smooth_batch   forward + Rauch-Tung-Striebel backward pass; uses every sample
    eval_causal    state at query times from the last sample at or before each time
    eval_hermite   C1 cubic between smoothed knots, for targets
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AxisNoise:
    """Noise for one axis, in SI units.

    q      white-jerk spectral density (m^2/s^5): how fast acceleration may change
    sig_p  position measurement std (m)
    sig_v  velocity measurement std (m/s)
    sig_a0 prior std of the initial acceleration (m/s^2)
    """
    q: float
    sig_p: float
    sig_v: float
    sig_a0: float = 2.0


# Horizontal: lat/lon carry 5 decimals (about 1 m) and ADS-B position error of a few m;
# velocity is broadcast in 1 kt steps from a different message, a fraction of a second off
# the position time. Vertical: altitude in 25 ft steps (std 2.2 m), vertical rate in 64 fpm
# steps (0.33 m/s) and noisier than that in practice. q is set so a 25 deg bank turn
# (lateral acceleration 4.6 m/s^2) can build up in about 3 s; see scripts/inspect_v2.py for
# the innovation check that validates these numbers.
HORIZONTAL = AxisNoise(q=0.6, sig_p=6.0, sig_v=1.0)
VERTICAL = AxisNoise(q=0.15, sig_p=4.0, sig_v=0.8, sig_a0=1.0)


def _F(dt):
    """Transition matrices for a vector of dt, shape [..., 3, 3]."""
    F = np.zeros(dt.shape + (3, 3))
    F[..., 0, 0] = F[..., 1, 1] = F[..., 2, 2] = 1.0
    F[..., 0, 1] = F[..., 1, 2] = dt
    F[..., 0, 2] = 0.5 * dt * dt
    return F


def _Q(dt, q):
    """White-jerk process noise for a vector of dt and per-sequence q."""
    d2, d3 = dt * dt, dt * dt * dt
    d4, d5 = d3 * dt, d3 * dt * dt
    Q = np.empty(dt.shape + (3, 3))
    Q[..., 0, 0] = d5 / 20
    Q[..., 0, 1] = Q[..., 1, 0] = d4 / 8
    Q[..., 0, 2] = Q[..., 2, 0] = d3 / 6
    Q[..., 1, 1] = d3 / 3
    Q[..., 1, 2] = Q[..., 2, 1] = d2 / 2
    Q[..., 2, 2] = dt
    return Q * q[..., None, None]


def filter_batch(t, zp, zv, n, noise: AxisNoise | list, keep=False):
    """Forward Kalman filter over B padded sequences.

    t   [B, T] sample times (s), increasing within the first n[b] entries
    zp  [B, T] position measurements
    zv  [B, T] velocity measurements, NaN where absent
    n   [B]    number of real samples per row (>= 1)
    noise      one AxisNoise for all rows, or a list of B

    Returns the filtered state xf [B, T, 3] and covariance Pf [B, T, 3, 3]; with keep=True
    also the one-step predictions (xp, Pp, F) the smoother needs. Rows past n[b] repeat the
    last real state.
    """
    t = np.asarray(t, float)
    B, T = t.shape
    if isinstance(noise, AxisNoise):
        noise = [noise] * B
    q = np.array([z.q for z in noise])
    rp = np.array([z.sig_p for z in noise]) ** 2
    rv = np.array([z.sig_v for z in noise]) ** 2
    ra = np.array([z.sig_a0 for z in noise]) ** 2

    valid = np.arange(T)[None, :] < np.asarray(n)[:, None]
    dt = np.zeros((B, T))
    dt[:, 1:] = np.where(valid[:, 1:], t[:, 1:] - t[:, :-1], 0.0)
    Fs, Qs = _F(dt), _Q(dt, q[:, None])

    x = np.zeros((B, 3))
    P = np.zeros((B, 3, 3))
    v0 = np.where(np.isnan(zv[:, 0]), 0.0, zv[:, 0])
    x[:, 0], x[:, 1] = zp[:, 0], v0
    P[:, 0, 0], P[:, 2, 2] = rp, ra
    P[:, 1, 1] = np.where(np.isnan(zv[:, 0]), 100.0 ** 2, rv)

    xf = np.empty((B, T, 3))
    Pf = np.empty((B, T, 3, 3))
    xf[:, 0], Pf[:, 0] = x, P
    if keep:
        xp = np.empty((B, T, 3))
        Pp = np.empty((B, T, 3, 3))
        xp[:, 0], Pp[:, 0] = x, P
    eye = np.eye(3)
    for k in range(1, T):
        F, m = Fs[:, k], valid[:, k]
        x = np.einsum("bij,bj->bi", F, x)
        P = F @ P @ F.transpose(0, 2, 1) + Qs[:, k]
        if keep:
            xp[:, k], Pp[:, k] = x, P
        # Update with position and, where present, velocity. A missing component gets
        # infinite noise, which removes it from the update without branching.
        has_v = ~np.isnan(zv[:, k])
        R = np.zeros((B, 2, 2))
        R[:, 0, 0] = rp
        R[:, 1, 1] = np.where(has_v, rv, 1e12)
        S = P[:, :2, :2] + R
        K = P[:, :, :2] @ np.linalg.inv(S)
        y = np.stack([zp[:, k] - x[:, 0], np.where(has_v, zv[:, k], x[:, 1]) - x[:, 1]], 1)
        xu = x + np.einsum("bij,bj->bi", K, y)
        IKH = eye[None] - K @ np.eye(3)[:2][None]
        Pu = IKH @ P @ IKH.transpose(0, 2, 1) + K @ R @ K.transpose(0, 2, 1)   # Joseph form
        x = np.where(m[:, None], xu, x)
        P = np.where(m[:, None, None], Pu, P)
        xf[:, k], Pf[:, k] = x, P
    if keep:
        return xf, Pf, xp, Pp, Fs
    return xf, Pf


def smooth_batch(t, zp, zv, n, noise: AxisNoise | list):
    """Forward filter plus RTS backward pass. Returns smoothed states xs [B, T, 3]."""
    xf, Pf, xp, Pp, Fs = filter_batch(t, zp, zv, n, noise, keep=True)
    T = xf.shape[1]
    xs = xf.copy()
    for k in range(T - 2, -1, -1):
        # C = Pf[k] F[k+1]^T Pp[k+1]^-1; on a padding step F = I, Q = 0 so C = I
        C = Pf[:, k] @ Fs[:, k + 1].transpose(0, 2, 1) @ np.linalg.inv(Pp[:, k + 1])
        xs[:, k] = xf[:, k] + np.einsum("bij,bj->bi", C, xs[:, k + 1] - xp[:, k + 1])
    return xs


def eval_causal(t, xf, n, tq):
    """State at query times tq [B, Q] from the filtered state of the last sample at or
    before each query, propagated by the motion model. Uses nothing after the query time,
    so a history built this way cannot see the future. Queries before the first sample
    are extrapolated backwards from it; callers keep those out by construction.
    """
    B, Q = tq.shape
    idx = np.empty((B, Q), int)
    for b in range(B):
        idx[b] = np.clip(np.searchsorted(t[b, :n[b]], tq[b], side="right") - 1, 0, n[b] - 1)
    x0 = np.take_along_axis(xf, idx[..., None], 1)
    dt = tq - np.take_along_axis(t, idx, 1)
    return np.einsum("bqij,bqj->bqi", _F(dt), x0)


def eval_hermite(t, xs, n, tq):
    """Position and velocity at query times by cubic Hermite interpolation between the
    smoothed knots that bracket each query. Position is C1 across knots, so a short gap is
    bridged by a smooth arc. Queries outside the knots use the motion model from the
    nearest knot.
    """
    B, Q = tq.shape
    out = np.empty((B, Q, 2))
    for b in range(B):
        tb, pb, vb = t[b, :n[b]], xs[b, :n[b], 0], xs[b, :n[b], 1]
        ab = xs[b, :n[b], 2]
        q = tq[b]
        i = np.clip(np.searchsorted(tb, q, side="right") - 1, 0, max(n[b] - 2, 0))
        if n[b] == 1:
            d = q - tb[0]
            out[b, :, 0] = pb[0] + vb[0] * d + 0.5 * ab[0] * d * d
            out[b, :, 1] = vb[0] + ab[0] * d
            continue
        h = tb[i + 1] - tb[i]
        s = (q - tb[i]) / h
        h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
        h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
        p = h00 * pb[i] + h10 * h * vb[i] + h01 * pb[i + 1] + h11 * h * vb[i + 1]
        dh00, dh10 = (6 * s**2 - 6 * s) / h, 3 * s**2 - 4 * s + 1
        dh01, dh11 = (-6 * s**2 + 6 * s) / h, 3 * s**2 - 2 * s
        v = dh00 * pb[i] + dh10 * vb[i] + dh01 * pb[i + 1] + dh11 * vb[i + 1]
        # outside the knots: motion model from the nearest one
        for side, j in ((q < tb[0], 0), (q > tb[-1], n[b] - 1)):
            if side.any():
                d = q[side] - tb[j]
                p[side] = pb[j] + vb[j] * d + 0.5 * ab[j] * d * d
                v[side] = vb[j] + ab[j] * d
        out[b, :, 0], out[b, :, 1] = p, v
    return out
