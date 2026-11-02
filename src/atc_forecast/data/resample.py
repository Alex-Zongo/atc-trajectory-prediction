"""Raw ADS-B to 1 Hz history and future around a prediction time t0.

This is the single entry point from surveillance data to model inputs. The training build
uses it, and so does any external evaluation set: give it raw state vectors and the times
at which to predict, and it returns the same windows the models were trained on.

The order of operations is the leak guard:

  1. clean       one aircraft's raw rows: drop stale repeats and ground rows, gate jumps
  2. split       raw samples at or before t0 form the HISTORY side, samples after t0 the
                 FUTURE side. Nothing below ever mixes them.
  3. check gaps  a window is kept only if neither side has a raw gap wider than gap_s
                 (4 s), the history ends within max_stale_s of t0, and the aircraft stays
                 above min_gs_kt (no ground movement, no hovering). Rejected windows are
                 counted by reason, never patched.
  4. resample    history: a FORWARD Kalman filter, read at t0-29 ... t0 by propagating the
                 last sample at or before each grid time, so every history value uses
                 only data up to its own time. Future: forward filter plus RTS smoother on
                 the future samples only, read at t0+1 ... t0+60 by C1 Hermite
                 interpolation between smoothed samples.
  5. origin      the history is relative to its own estimate of the present, p0; the
                 future relative to the future smoother's estimate of the present. The
                 gap between the two (offset0, a few metres) is measurement noise and is
                 kept out of the targets. Raw future points are relative to p0, which is
                 where a forecast starts.

Units are SI throughout (m, m/s), in a local east-north-up frame at the window's last
history sample. Conversion to model features happens in frames.py.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..units import DEG_TO_M, KT_TO_MS
from .kalman import HORIZONTAL, VERTICAL, eval_causal, eval_hermite, filter_batch, smooth_batch


@dataclass(frozen=True)
class WindowSpec:
    hist_s: int = 30          # history grid t0-29 ... t0 (30 points, the last is the present)
    fut_s: int = 60           # future grid t0+1 ... t0+60
    burn_in_s: float = 10.0   # raw history before the grid, so the filter has converged
    gap_s: float = 4.0        # widest raw gap a side may contain
    max_stale_s: float = 2.0  # the newest history sample is at most this old at t0
    end_slack_s: float = 1.0  # the future may extrapolate this far past its last sample
    min_gs_kt: float = 50.0   # below this anywhere in the window: ground or hover, dropped
    altitude: str = "baro"    # baro: what ATC assigns; geo: GNSS height


REJECT_REASONS = ("no_track", "history_short", "history_gap", "history_stale",
                  "future_short", "future_gap", "slow")


def prepare_track(g: pd.DataFrame, altitude: str = "baro") -> dict | None:
    """One aircraft's raw rows to clean arrays, sorted by position time.

    OpenSky emits a state vector about once a second even when no new position arrived;
    about 22 percent of rows repeat the previous time_position and are dropped. Ground rows
    are removed, which leaves a gap that the window check then refuses to bridge. Position
    and altitude jumps faster than any aircraft can move are outliers and dropped.
    """
    alt_col = "baro_altitude" if altitude == "baro" else "geo_altitude"
    cols = ["time_position", "latitude", "longitude", alt_col, "velocity", "true_track",
            "vertical_rate"]
    g = g.copy()
    for c in cols:
        g[c] = pd.to_numeric(g[c], errors="coerce").astype("float64")
    if "on_ground" in g.columns:
        og = g["on_ground"].astype(str).str.lower().isin(["true", "1"])
        g = g[~og]
    g = (g.dropna(subset=["time_position", "latitude", "longitude", alt_col])
          .sort_values("time_position").drop_duplicates("time_position"))
    if len(g) < 5:
        return None
    t = g["time_position"].to_numpy()
    lat, lon = g["latitude"].to_numpy(), g["longitude"].to_numpy()
    z = g[alt_col].to_numpy()
    keep = np.ones(len(t), bool)
    # jump gate against the previous KEPT sample: 400 m/s horizontal, 150 m/s vertical
    j = 0
    for i in range(1, len(t)):
        dt = max(t[i] - t[j], 1e-3)
        dn = (lat[i] - lat[j]) * DEG_TO_M
        de = (lon[i] - lon[j]) * DEG_TO_M * np.cos(np.radians(lat[j]))
        if np.hypot(de, dn) > 400.0 * dt + 50.0 or abs(z[i] - z[j]) > 150.0 * dt + 30.0:
            keep[i] = False
        else:
            j = i
    g = g[keep]
    gs = g["velocity"].to_numpy()
    trk = np.radians(g["true_track"].to_numpy())
    return {
        "t": g["time_position"].to_numpy(), "lat": g["latitude"].to_numpy(),
        "lon": g["longitude"].to_numpy(), "z": g[alt_col].to_numpy(),
        "ve": gs * np.sin(trk), "vn": gs * np.cos(trk),
        "vz": g["vertical_rate"].to_numpy(),
    }


def _check_side(ts, lo, hi, gap, lead, tail):
    """Gaps of the sequence [lo, ts..., hi]: lead = allowed lo->first, tail = last->hi."""
    if len(ts) == 0:
        return "short"
    if ts[0] - lo > lead:
        return "short"
    if len(ts) > 1 and np.max(np.diff(ts)) > gap:
        return "gap"
    if hi - ts[-1] > tail:
        return "stale"
    return None


def _pad(rows, width, fill=np.nan):
    out = np.full((len(rows), width), fill)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def gather(track: dict, t0s, spec: WindowSpec = WindowSpec()):
    """Cut the raw samples around each t0 into a history side (t <= t0) and a future side
    (t > t0), and check the gap rules on each. Cheap; the filtering happens in resample().

    Returns (items, reasons): one dict of raw arrays per accepted t0, and a reason per
    requested t0 (None when accepted).
    """
    t0s = np.asarray(t0s, float)
    if track is None:
        return [], ["no_track"] * len(t0s)
    t = track["t"]
    H, F = spec.hist_s, spec.fut_s
    items, reasons = [], [None] * len(t0s)
    for i, t0 in enumerate(t0s):
        lo = t0 - (H - 1) - spec.burn_in_s
        a, b = np.searchsorted(t, lo, "left"), np.searchsorted(t, t0, "right")
        r = _check_side(t[a:b], lo, t0, spec.gap_s, spec.gap_s, spec.max_stale_s)
        if r is not None:
            reasons[i] = "history_" + r
            continue
        d = np.searchsorted(t, t0 + F + spec.gap_s, "right")
        r = _check_side(t[b:d], t0, t0 + F, spec.gap_s, spec.gap_s, spec.end_slack_s)
        if r is not None:
            reasons[i] = "future_" + ("short" if r == "stale" else r)
            continue
        # local frame at the last history sample; east uses each sample's own latitude,
        # which keeps the east scale right over the 20 km a window can span
        lat0, lon0 = track["lat"][b - 1], track["lon"][b - 1]
        sl = slice(a, d)
        e = (track["lon"][sl] - lon0) * DEG_TO_M * np.cos(np.radians(track["lat"][sl]))
        n = (track["lat"][sl] - lat0) * DEG_TO_M
        items.append({"i": i, "t0": t0, "lat0": lat0, "lon0": lon0, "nh": b - a,
                      "t": t[sl], "p": np.stack([e, n, track["z"][sl]], 1),
                      "v": np.stack([track["ve"][sl], track["vn"][sl], track["vz"][sl]], 1)})
    return items, reasons


def _run(items, side, smooth):
    """Kalman pass over one side of every item, three axes as 3B independent rows."""
    rows = [(it["t"][:it["nh"]], it["p"][:it["nh"]], it["v"][:it["nh"]]) if side == "h"
            else (it["t"][it["nh"]:], it["p"][it["nh"]:], it["v"][it["nh"]:]) for it in items]
    n = np.array([len(r[0]) for r in rows])
    T = n.max()
    tt = _pad([r[0] for r in rows], T)
    tt = np.where(np.isnan(tt), np.nanmax(tt, 1, keepdims=True), tt)
    zp = np.concatenate([_pad([r[1][:, a] for r in rows], T, 0.0) for a in range(3)])
    zv = np.concatenate([_pad([r[2][:, a] for r in rows], T) for a in range(3)])
    t3, n3 = np.concatenate([tt] * 3), np.concatenate([n] * 3)
    noise = [HORIZONTAL] * (2 * len(n)) + [VERTICAL] * len(n)
    x = smooth_batch(t3, zp, zv, n3, noise) if smooth else filter_batch(t3, zp, zv, n3, noise)[0]
    return t3, x, n3


def resample(items, spec: WindowSpec = WindowSpec()):
    """Filter the history side, smooth the future side, read both on the 1 Hz grid.

    Returns a dict of arrays over the items that also pass the ground-speed gate, with
    `keep` the positions kept (in items order) and `slow` those dropped.
    """
    if not items:
        return None
    B, H, F = len(items), spec.hist_s, spec.fut_s
    t0a = np.array([it["t0"] for it in items])
    th3, xh, nh3 = _run(items, "h", smooth=False)
    tf3, xs, nf3 = _run(items, "f", smooth=True)
    hg = t0a[:, None] + np.arange(-(H - 1), 1)[None, :]
    fg = t0a[:, None] + np.arange(0, F + 1)[None, :]                    # t0 and t0+1..t0+F
    hist = eval_causal(th3, xh, nh3, np.concatenate([hg] * 3))          # [3B, H, 3]
    fut = eval_hermite(tf3, xs, nf3, np.concatenate([fg] * 3))          # [3B, F+1, 2]
    split = lambda a: np.stack([a[:B], a[B:2 * B], a[2 * B:]], -1)       # -> [B, ..., 3]
    hp, hv, ha = split(hist[..., 0]), split(hist[..., 1]), split(hist[..., 2])
    fp, fv = split(fut[..., 0])[:, 1:], split(fut[..., 1])[:, 1:]

    # Two estimates of the present: p0 from the history (what a forecaster knows) and the
    # future smoother's own value at t0. Their difference is measurement noise, not
    # motion, so the future is anchored at its own estimate: targets then begin smoothly
    # and still use future samples only. The offset is kept for diagnostics; scoring
    # against raw positions (anchored at p0) still contains it.
    p0 = hp[:, -1].copy()
    anchor = split(fut[..., 0])[:, 0]
    hp, fp = hp - p0[:, None], fp - anchor[:, None]
    raw = []
    for k, it in enumerate(items):
        tf, pf = it["t"][it["nh"]:], it["p"][it["nh"]:]
        m = tf <= it["t0"] + F
        raw.append(np.concatenate([(tf[m] - it["t0"])[:, None], pf[m] - p0[k]], 1))

    gs_min = np.minimum(np.hypot(hv[..., 0], hv[..., 1]).min(1),
                        np.hypot(fv[..., 0], fv[..., 1]).min(1))
    fast = gs_min >= spec.min_gs_kt * KT_TO_MS
    keep = np.where(fast)[0]
    return {
        "keep": keep, "slow": np.where(~fast)[0],
        "t0": t0a[keep], "hist_pos": hp[keep], "hist_vel": hv[keep], "hist_acc": ha[keep],
        "fut_pos": fp[keep], "fut_vel": fv[keep], "z0": p0[keep, 2],
        "offset0": (anchor - p0)[keep],                                 # [B, 3] m
        "lat0": np.array([it["lat0"] for it in items])[keep],
        "lon0": np.array([it["lon0"] for it in items])[keep],
        "n_hist_raw": np.array([it["nh"] for it in items])[keep],
        "raw_future": [raw[k] for k in keep],                           # (t-t0, e, n, z) - p0
    }


def extract_windows(track: dict, t0s, spec: WindowSpec = WindowSpec()):
    """Windows for one aircraft at the requested prediction times.

    Returns (windows, reasons): windows holds the accepted t0s (`index` is their position
    in t0s), reasons one entry per requested t0, None when accepted.
    """
    items, reasons = gather(track, t0s, spec)
    w = resample(items, spec)
    if w is None:
        return None, reasons
    idx = np.array([it["i"] for it in items])
    for k in w.pop("slow"):
        reasons[idx[k]] = "slow"
    w["index"] = idx[w.pop("keep")]
    return (w if len(w["index"]) else None), reasons


def history_windows(track: dict, t0s, spec: WindowSpec = WindowSpec()):
    """Windows for live prediction: the history side only, no recorded future needed.

    The two sides never mix (steps 2-5 above), so the history, the present and the
    features built from them are the same as extract_windows gives; only the future is
    absent (fut_pos is NaN, raw_future empty) and the ground-speed gate reads the history
    alone. Returns (windows, reasons) like extract_windows.
    """
    t0s = np.asarray(t0s, float)
    if track is None:
        return None, ["no_track"] * len(t0s)
    t, H = track["t"], spec.hist_s
    items, reasons = [], [None] * len(t0s)
    for i, t0 in enumerate(t0s):
        lo = t0 - (H - 1) - spec.burn_in_s
        a, b = np.searchsorted(t, lo, "left"), np.searchsorted(t, t0, "right")
        r = _check_side(t[a:b], lo, t0, spec.gap_s, spec.gap_s, spec.max_stale_s)
        if r is not None:
            reasons[i] = "history_" + r
            continue
        lat0, lon0 = track["lat"][b - 1], track["lon"][b - 1]
        sl = slice(a, b)
        e = (track["lon"][sl] - lon0) * DEG_TO_M * np.cos(np.radians(track["lat"][sl]))
        n = (track["lat"][sl] - lat0) * DEG_TO_M
        items.append({"i": i, "t0": t0, "lat0": lat0, "lon0": lon0, "nh": b - a, "t": t[sl],
                      "p": np.stack([e, n, track["z"][sl]], 1),
                      "v": np.stack([track["ve"][sl], track["vn"][sl], track["vz"][sl]], 1)})
    if not items:
        return None, reasons
    B, F = len(items), spec.fut_s
    t0a = np.array([it["t0"] for it in items])
    th3, xh, nh3 = _run(items, "h", smooth=False)
    hg = t0a[:, None] + np.arange(-(H - 1), 1)[None, :]
    hist = eval_causal(th3, xh, nh3, np.concatenate([hg] * 3))
    split = lambda a: np.stack([a[:B], a[B:2 * B], a[2 * B:]], -1)
    hp, hv, ha = split(hist[..., 0]), split(hist[..., 1]), split(hist[..., 2])
    p0 = hp[:, -1].copy()
    hp = hp - p0[:, None]
    fast = np.hypot(hv[..., 0], hv[..., 1]).min(1) >= spec.min_gs_kt * KT_TO_MS
    idx = np.array([it["i"] for it in items])
    for k in np.where(~fast)[0]:
        reasons[idx[k]] = "slow"
    keep = np.where(fast)[0]
    if not len(keep):
        return None, reasons
    w = {"t0": t0a[keep], "hist_pos": hp[keep], "hist_vel": hv[keep], "hist_acc": ha[keep],
         "fut_pos": np.full((len(keep), F, 3), np.nan), "z0": p0[keep, 2],
         "offset0": np.full((len(keep), 3), np.nan),
         "lat0": np.array([it["lat0"] for it in items])[keep],
         "lon0": np.array([it["lon0"] for it in items])[keep],
         "n_hist_raw": np.array([it["nh"] for it in items])[keep],
         "raw_future": [np.zeros((0, 4)) for _ in keep], "index": idx[keep]}
    return w, reasons


def windows_from_adsb(raw: pd.DataFrame, requests: pd.DataFrame,
                      spec: WindowSpec = WindowSpec(), future: bool = True):
    """External interface: raw OpenSky-style state vectors plus a table of (icao24, t0)
    prediction requests, to model-ready windows.

    Returns (windows, report): the windows of the accepted requests (with `request_index`)
    and a per-request table of the rejection reason. Feature construction then goes through
    frames.make_features, exactly as in training. future=False: live prediction, history
    only (history_windows).
    """
    raw = raw.assign(icao24=raw["icao24"].astype(str))
    groups = dict(tuple(raw.groupby("icao24", sort=False)))
    parts, rep = [], np.full(len(requests), None, dtype=object)
    req = requests.reset_index(drop=True)
    cut = extract_windows if future else history_windows
    for ic, r in req.groupby(req["icao24"].astype(str)):
        tr = prepare_track(groups[ic], spec.altitude) if ic in groups else None
        w, reasons = cut(tr, r["t0"].to_numpy(), spec)
        rep[r.index.to_numpy()] = reasons
        if w is not None:
            w["request_index"] = r.index.to_numpy()[w.pop("index")]
            parts.append(w)
    return concat_windows(parts), pd.DataFrame({"reason": rep}).assign(
        accepted=lambda d: d["reason"].isna())


def concat_windows(parts):
    if not parts:
        return None
    out = {}
    for k in parts[0]:
        if k == "raw_future":
            out[k] = [r for p in parts for r in p[k]]
        else:
            out[k] = np.concatenate([p[k] for p in parts])
    return out
