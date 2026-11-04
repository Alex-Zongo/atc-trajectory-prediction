"""Maneuver primitives: a few physical numbers per axis that a kinematic rule turns into a
path (exploratory; scripts/explore_primitives_v2.py, scripts/explore_kinds_v2.py).

Per axis the rate of change (turn rate, vertical rate, acceleration) follows

    present rate v0 until the delay d,
    a linear ramp of duration tau to the new rate r   (acceleration |r - v0| / tau),
    hold, then a ramp of the same duration back to 0 starting at e (capture of the target).

For given (d, tau, e) the cumulative change is v0 G0(t) + r G1(t), linear in r, so a fit is
a grid over (d, tau, e) with r in closed form.
"""
import numpy as np

T = np.arange(1, 61)
TAUS = (0, 3, 6, 10, 15, 20)


def ramps(d, tau, e, t=T):
    """Rate weights: g0 multiplies the present rate, g1 the new rate."""
    tc = t - 0.5                                         # rate over step k is centred at k - 1/2
    up = np.clip((tc - d) / tau, 0, 1) if tau > 0 else (tc >= d).astype(float)
    down = np.clip((tc - e) / tau, 0, 1) if tau > 0 else (tc >= e).astype(float)
    return 1 - up, up * (1 - down)


def fit_smooth(y, v0, d_grid=np.arange(0, 31, 1.0), taus=TAUS, e_step=3):
    """Least-squares fit of the smooth primitive to cumulative changes y [N, 60] with present
    rates v0 [N]. Returns params [N, 4] = (d, r, tau, e; e = 61 when no capture within the
    horizon), the RMS error [N] and the fitted curves [N, 60]."""
    N = len(y)
    best = np.full(N, np.inf)
    P = np.zeros((N, 4))
    Yh = np.zeros_like(y, dtype=float)
    for tau in taus:
        for d in d_grid:
            for e in list(np.arange(d + tau + e_step, 61, e_step)) + [1e9]:
                g0, g1 = ramps(d, tau, e)
                G0, G1 = np.cumsum(g0), np.cumsum(g1)
                gg = (G1 * G1).sum()
                if gg == 0:
                    continue
                r = ((y - v0[:, None] * G0[None]) @ G1) / gg
                yh = v0[:, None] * G0[None] + r[:, None] * G1[None]
                err = ((y - yh) ** 2).mean(1)
                w = err < best
                best[w] = err[w]
                Yh[w] = yh[w]
                P[w] = np.c_[np.full(w.sum(), d), r[w], np.full(w.sum(), tau),
                             np.full(w.sum(), min(e, 61))]
    return P, np.sqrt(best), Yh


def kinds(P, v0, thresh):
    """Kind per window and axis: 0 keeps its present rate, 1 changes rate and captures within
    the horizon, 2 changes rate and is still changing at the horizon."""
    change = np.abs(P[:, 1] - v0) > thresh
    return np.where(~change, 0, np.where(P[:, 3] < 61, 1, 2))


def rate_curve(v0, d, r, tau, e, t=T):
    """Rate per step implied by the parameters (one window)."""
    g0, g1 = ramps(d, tau, e, t)
    return v0 * g0 + r * g1


# ---------------------------------------------------------------- design A targets
def grid(d_grid=np.arange(0, 31, 1.0), taus=TAUS, e_step=3):
    """All (d, tau, e) combinations with their cumulative weights G0, G1 [C, 60]."""
    combos, G0, G1 = [], [], []
    for tau in taus:
        for d in d_grid:
            for e in list(np.arange(d + tau + e_step, 61, e_step)) + [61.0]:
                g0, g1 = ramps(d, tau, e if e < 61 else 1e9)
                if g1.sum() == 0:
                    continue
                combos.append((d, tau, e))
                G0.append(np.cumsum(g0))
                G1.append(np.cumsum(g1))
    return np.array(combos), np.array(G0), np.array(G1)


def fit_free(y, chunk=20000, grid_=None):
    """Fit with the present rate v0 free as well (no look at the future's first step): for
    each (d, tau, e) the pair (v0, r) is a 2x2 least-squares solve, done for every grid point
    at once from Y @ [G0, G1]. Returns params [N, 5] = (v0, d, r, tau, e) and RMS [N]."""
    combos, G0, G1 = grid_ if grid_ is not None else grid()
    a, b, c = (G0 * G0).sum(1), (G0 * G1).sum(1), (G1 * G1).sum(1)          # [C]
    det = a * c - b * b
    ok = det > 1e-9
    out, rms = np.zeros((len(y), 5), np.float32), np.zeros(len(y), np.float32)
    for i in range(0, len(y), chunk):
        Y = np.asarray(y[i:i + chunk], np.float64)
        p, q = Y @ G0.T, Y @ G1.T                                           # [n, C]
        # singular grid points: d = 0 (no present-rate phase) -> r alone; no new rate -> v0 alone
        only_r, only_v = ~ok & (a < 1e-9), ~ok & (a >= 1e-9)
        safe = lambda x: np.where(x > 1e-9, x, 1.0)
        v0 = np.where(ok, (c * p - b * q) / safe(det), np.where(only_v, p / safe(a), 0.0))
        r = np.where(ok, (a * q - b * p) / safe(det), np.where(only_r, q / safe(c), 0.0))
        err = (Y * Y).sum(1)[:, None] - 2 * (v0 * p + r * q) + v0 ** 2 * a + 2 * v0 * r * b + r ** 2 * c
        j = err.argmin(1)
        n = np.arange(len(Y))
        out[i:i + chunk] = np.c_[v0[n, j], combos[j, 0], r[n, j], combos[j, 1], combos[j, 2]]
        rms[i:i + chunk] = np.sqrt(np.maximum(err[n, j], 0) / Y.shape[1])
    return out, rms


def curve(P, t=T):
    """Cumulative change [N, 60] from params [N, 5] (v0, d, r, tau, e); vectorized ramps."""
    v0, d, r, tau, e = (P[:, i:i + 1].astype(np.float64) for i in range(5))
    tc = (t - 0.5)[None]
    tau_ = np.maximum(tau, 1e-3)
    up = np.where(tau > 0, np.clip((tc - d) / tau_, 0, 1), (tc >= d) * 1.0)
    down = np.where(e >= 61, 0.0, np.where(tau > 0, np.clip((tc - e) / tau_, 0, 1), (tc >= e) * 1.0))
    rate = v0 * (1 - up) + r * up * (1 - down)
    return np.cumsum(rate, 1)


KT, FT = 1.94384, 3.28084


def signals(fut_vel, center):
    """Per-axis cumulative signals of a future: heading change (deg), altitude change (ft),
    ground-speed change (kt, against the present speed)."""
    psi = np.degrees(np.unwrap(np.arctan2(fut_vel[..., 1], fut_vel[..., 0]), axis=1))
    z = np.cumsum(fut_vel[..., 2], 1) * FT
    gs = np.hypot(fut_vel[..., 0], fut_vel[..., 1]) * KT - center[:, 0:1] * KT
    return psi, z, gs


def velocities(psi, z, gs, center):
    """Inverse of signals: track-frame velocities [N, 60, 3] (m/s)."""
    V = center[:, 0:1] + gs / KT
    h = np.radians(psi)
    vz = np.diff(np.concatenate([np.zeros((len(z), 1)), z], 1), axis=1) / FT
    return np.stack([V * np.cos(h), V * np.sin(h), vz], -1)


G = 9.81


def rates(P, t=T):
    """Rate per step [N, 60] from params [N, 5]."""
    return np.diff(np.concatenate([np.zeros((len(P), 1)), curve(P, t)], 1), axis=1)


def to_bank(P, center):
    """Turn primitive in bank angle (deg): phi = atan(omega V / g), with V the ground speed of
    the speed primitive at the start of each phase (present rate: t = 0; new rate: t = d)."""
    P = P.copy()
    V = center[:, 0:1] + curve(P[:, 2]) / KT                       # m/s [N, 60]
    k = np.clip(P[:, 0, 1].astype(int), 0, 59)
    Vd = V[np.arange(len(P)), k]
    P[:, 0, 0] = np.degrees(np.arctan(np.radians(P[:, 0, 0]) * center[:, 0] / G))
    P[:, 0, 2] = np.degrees(np.arctan(np.radians(P[:, 0, 2]) * Vd / G))
    return P


def bank_velocities(P, center):
    """Velocities [N, 60, 3] from a bank-angle turn primitive: the bank profile follows the
    primitive, the turn rate is g tan(phi) / V(t) with V(t) from the speed primitive."""
    V = center[:, 0:1] + curve(P[:, 2]) / KT
    phi = rates(P[:, 0])                                             # bank (deg) per step
    psi = np.cumsum(np.degrees(G * np.tan(np.radians(phi)) / np.maximum(V, 1.0)), 1)
    return velocities(psi, curve(P[:, 1]), curve(P[:, 2]), center)
