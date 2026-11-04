"""Raw ADS-B point clouds to maneuver segments.

An ATC instruction corresponds to the ONSET of a maneuver; the trajectory segment is its
execution. So each track is segmented by change points on the three channels a controller
actually commands, taken from the smoothed signals:

    lateral   turns          from true_track
    vertical  climb/descent  from geo_altitude
    speed     accel/decel    from velocity

A segment is a maximal run over which the (lateral, vertical, speed) state is constant,
after merging sub-threshold runs to suppress ADS-B jitter.

Pipeline: label_phases -> split on coverage gaps -> smooth -> classify -> merge short runs
-> emit segment records. OpenSky gives metres and m/s; segment outputs are converted to
aviation units for phraseology.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ..units import FL180_M, MS_TO_FPM, MS_TO_KT, M_TO_FT

TURN_RATE_TH = 1.0
CUM_TURN_TH = 30.0
VS_TH = 2.5
ACCEL_TH = 0.50
MIN_SEG_S = 15.0
MAX_GAP_S = 30.0
MIN_POINTS = 15
SMOOTH_WIN = 11

NUM_COLS = ["longitude", "latitude", "geo_altitude", "velocity",
            "true_track", "vertical_rate", "time_position"]


@dataclass(frozen=True)
class SegParams:
    """Detection thresholds. Pass a modified copy to re-segment without mutating module state.

    `cum_turn_th` catches GRADUAL turns whose rate stays below `turn_rate_th` but whose net
    heading change is large; without it 10 to 22 percent of such segments are mislabelled
    straight. `accel_th` is deliberately high because `velocity` is groundspeed, which
    conflates commanded speed changes with wind and with the natural accel of a climb: at
    0.25 m/s^2 speed came out as the most common clearance, inverted against real ATC.
    """
    turn_rate_th: float = TURN_RATE_TH
    cum_turn_th: float = CUM_TURN_TH
    vs_th: float = VS_TH
    accel_th: float = ACCEL_TH
    min_seg_s: float = MIN_SEG_S
    max_gap_s: float = MAX_GAP_S
    min_points: int = MIN_POINTS
    smooth_win: int = SMOOTH_WIN


def to_numpy_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to plain numpy dtypes.

    Parquet preserves pyarrow-backed nullable dtypes, whose comparisons produce nullable
    booleans containing pd.NA, which numpy cannot use as an index mask. CSV hides this
    because read_csv yields float64, so normalise once up front.
    """
    df = df.copy()
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    if "on_ground" in df.columns:
        df["on_ground"] = df["on_ground"].astype(str).str.lower().isin(["true", "1"])
    if "callsign" in df.columns:
        df["callsign"] = df["callsign"].astype("object")
    return df


def label_phases(df: pd.DataFrame) -> pd.DataFrame:
    """Add a kinematic `phase`: ground / climb / descent / cruise / level_low, where
    level_low is terminal level flight below FL180."""
    df = to_numpy_dtypes(df)
    alt = df["geo_altitude"].to_numpy("float64")
    vs = df["vertical_rate"].to_numpy("float64")
    og = df["on_ground"].to_numpy(bool)
    phase = np.full(len(df), "level_low", dtype=object)
    with np.errstate(invalid="ignore"):
        phase[vs > VS_TH] = "climb"
        phase[vs < -VS_TH] = "descent"
        phase[(np.abs(vs) <= VS_TH) & (alt >= FL180_M)] = "cruise"
        phase[og | (alt < 30)] = "ground"
    df["phase"] = phase
    return df


def _smooth(x, win=SMOOTH_WIN, poly=2):
    x = np.asarray(x, dtype=float)
    if len(x) < 5:
        return x
    w = min(win, len(x))
    w -= 1 - w % 2
    return x if w <= poly else savgol_filter(x, w, poly)


def _smooth_track(track_deg, win=SMOOTH_WIN):
    """Unwrap heading before smoothing so 359 to 0 is not read as a 359 degree swing."""
    return _smooth(np.degrees(np.unwrap(np.radians(np.asarray(track_deg, dtype=float)))), win)


def _runs(state):
    if len(state) == 0:
        return []
    change = np.where(state[1:] != state[:-1])[0] + 1
    bounds = [0, *change.tolist(), len(state)]
    return list(zip(bounds[:-1], bounds[1:]))


def _merge_short(state, t, min_s):
    state = state.copy()
    while True:
        runs = _runs(state)
        if len(runs) <= 1:
            break
        short = [(t[e - 1] - t[s], s, e) for s, e in runs if (t[e - 1] - t[s]) < min_s]
        if not short:
            break
        short.sort()
        _, s, e = short[0]
        idx = next(i for i, (rs, re_) in enumerate(runs) if rs == s and re_ == e)
        state[s:e] = state[runs[idx - 1][0]] if idx > 0 else state[runs[idx + 1][0]]
    return state


def track_signals(sub: pd.DataFrame, params: SegParams = SegParams()) -> dict:
    """Every intermediate signal the segmenter derives from one contiguous track.

    The smoothed channels, their rates, the per-point labels before merging, and the merged
    state. `_segment_track` cuts segments from exactly this, so anything that displays the
    segmentation (the app's dataset walkthrough) shows the real computation.
    """
    t = sub["time_position"].to_numpy(float)
    track_sm = _smooth_track(sub["true_track"].to_numpy(float), params.smooth_win)
    alt_sm = _smooth(sub["geo_altitude"].to_numpy(float), params.smooth_win)
    vel_sm = _smooth(sub["velocity"].to_numpy(float), params.smooth_win)

    turn_rate = np.gradient(track_sm, t)
    vspeed = np.gradient(alt_sm, t)
    accel = np.gradient(vel_sm, t)

    lateral = np.where(turn_rate > params.turn_rate_th, "R",
                       np.where(turn_rate < -params.turn_rate_th, "L", "S"))
    vertical = np.where(vspeed > params.vs_th, "climb",
                        np.where(vspeed < -params.vs_th, "descend", "level"))
    speed = np.where(accel > params.accel_th, "accel",
                     np.where(accel < -params.accel_th, "decel", "const"))
    raw_state = np.array([f"{a}|{b}|{c}" for a, b, c in zip(lateral, vertical, speed)])
    return {"t": t, "track_sm": track_sm, "alt_sm": alt_sm, "vel_sm": vel_sm,
            "turn_rate": turn_rate, "vspeed": vspeed, "accel": accel,
            "lateral": lateral, "vertical": vertical, "speed": speed,
            "raw_state": raw_state, "state": _merge_short(raw_state, t, params.min_seg_s)}


def _segment_track(sub: pd.DataFrame, params: SegParams) -> list[dict]:
    track = sub["true_track"].to_numpy(float)
    lat = sub["latitude"].to_numpy(float)
    lon = sub["longitude"].to_numpy(float)
    phase = sub["phase"].to_numpy() if "phase" in sub else np.full(len(sub), "")

    sig = track_signals(sub, params)
    t, track_sm, alt_sm, vel_sm = sig["t"], sig["track_sm"], sig["alt_sm"], sig["vel_sm"]
    state = sig["state"]

    icao = sub["icao24"].iloc[0]
    callsign = str(sub["callsign"].iloc[0]).strip() if "callsign" in sub else icao

    out: list[dict] = []
    for s, e in _runs(state):
        lat_state, vert_state, spd_state = state[s].split("|")
        j = e - 1
        turn_deg = float(track_sm[j] - track_sm[s])
        if lat_state == "S" and abs(turn_deg) >= params.cum_turn_th:
            lat_state = "R" if turn_deg > 0 else "L"
        alt_entry, alt_exit = float(alt_sm[s]) * M_TO_FT, float(alt_sm[j]) * M_TO_FT
        spd_entry, spd_exit = float(vel_sm[s]) * MS_TO_KT, float(vel_sm[j]) * MS_TO_KT
        vals, counts = np.unique(phase[s:e], return_counts=True)
        seg_t = t[s:e]
        if len(seg_t) > 1:
            d = np.diff(seg_t)
            median_dt, max_dt = float(np.median(d)), float(d.max())
        else:
            median_dt = max_dt = float("nan")
        duration = float(t[j] - t[s])
        out.append({
            "icao24": icao, "callsign": callsign,
            "t_start": float(t[s]), "t_end": float(t[j]),
            "duration_s": round(duration, 1), "n_points": int(e - s),
            "lateral": lat_state, "vertical": vert_state, "speed": spd_state,
            "phase": vals[counts.argmax()] if len(vals) else "",
            "is_maneuver": not (lat_state == "S" and vert_state == "level"
                                and spd_state == "const"),
            "hdg_entry": round(float(track[s]) % 360, 1),
            "hdg_exit": round(float(track[j]) % 360, 1),
            "turn_deg": round(turn_deg, 1),
            "alt_entry_ft": round(alt_entry), "alt_exit_ft": round(alt_exit),
            "alt_change_ft": round(alt_exit - alt_entry),
            "spd_entry_kt": round(spd_entry), "spd_exit_kt": round(spd_exit),
            "spd_change_kt": round(spd_exit - spd_entry),
            "mean_alt_ft": round((alt_entry + alt_exit) / 2),
            "median_dt_s": round(median_dt, 1), "max_dt_s": round(max_dt, 1),
            "coverage": round((e - s) / duration if duration > 0 else 0.0, 2),
            "lat_start": round(float(lat[s]), 5), "lon_start": round(float(lon[s]), 5),
            "lat_end": round(float(lat[j]), 5), "lon_end": round(float(lon[j]), 5),
        })
    return out


def resolve_callsign(g: pd.DataFrame, icao: str) -> str:
    """The most frequent non-null callsign for an aircraft.

    ADS-B drops the callsign on a small fraction of points, and taking the first point's
    value poisons every segment of that aircraft. Falls back to the icao24 hex, so the
    callsign is never NaN (which would otherwise be spelled out as "november alpha
    november").
    """
    if "callsign" not in g.columns:
        return str(icao)
    cs = g["callsign"].dropna().astype(str).str.strip()
    cs = cs[(cs != "") & (cs.str.lower() != "nan")]
    return str(cs.mode().iloc[0]) if not cs.empty else str(icao)


def segment_aircraft(g: pd.DataFrame, params: SegParams = SegParams()) -> list[dict]:
    g = g.sort_values("time_position").drop_duplicates("time_position").reset_index(drop=True)
    if len(g) < params.min_points:
        return []
    g = g.copy()
    g["callsign"] = resolve_callsign(g, g["icao24"].iloc[0])
    t = g["time_position"].to_numpy(float)
    splits = np.where(np.diff(t) > params.max_gap_s)[0] + 1
    out: list[dict] = []
    for idx in np.split(np.arange(len(g)), splits):
        if len(idx) >= params.min_points:
            out.extend(_segment_track(g.iloc[idx], params))
    return out


MIN_COVERAGE = 0.4
MAX_MEDIAN_DT = 4.0
MIN_DURATION = 10.0


def filter_segments(segs: pd.DataFrame, min_coverage=MIN_COVERAGE,
                    max_median_dt_s=MAX_MEDIAN_DT, min_duration_s=MIN_DURATION,
                    min_alt_ft=None, keep_only=True) -> pd.DataFrame:
    """Quality gate: keep densely sampled segments that are long enough.

    `min_alt_ft` is optional and off by default. Coverage is the better discriminator,
    since low finals at a busy airport are well covered while low GA at the box edge is
    not, and that difference is the real signal.
    """
    if segs.empty:
        return segs
    ok = ((segs["coverage"] >= min_coverage)
          & (segs["median_dt_s"] <= max_median_dt_s)
          & (segs["duration_s"] >= min_duration_s))
    if min_alt_ft is not None:
        ok &= segs["mean_alt_ft"] >= min_alt_ft
    segs = segs.copy()
    segs["quality_ok"] = ok
    return segs[ok].reset_index(drop=True) if keep_only else segs


def segment_trajectories(df: pd.DataFrame, airborne_only=True, maneuvers_only=False,
                         quality_filter=False, min_alt_ft=None,
                         params: SegParams = SegParams()) -> pd.DataFrame:
    """Segment every aircraft in `df`, one row per segment."""
    df = label_phases(df)
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[c for c in NUM_COLS if c in df.columns])
    if airborne_only:
        df = df[df["phase"] != "ground"]

    segs: list[dict] = []
    for _, g in df.groupby("icao24", sort=False):
        segs.extend(segment_aircraft(g, params))
    result = pd.DataFrame(segs)
    if result.empty:
        return result
    if maneuvers_only:
        result = result[result["is_maneuver"]].reset_index(drop=True)
    if quality_filter:
        result = filter_segments(result, min_alt_ft=min_alt_ft)
    return result


def describe_segment(seg) -> str:
    """Short human-readable summary of a segment."""
    parts = []
    if seg["lateral"] in ("L", "R"):
        side = "left" if seg["lateral"] == "L" else "right"
        parts.append(f"turn {side} {abs(seg['turn_deg']):.0f}deg "
                     f"({seg['hdg_entry']:.0f}->{seg['hdg_exit']:.0f})")
    if seg["vertical"] in ("climb", "descend"):
        parts.append(f"{seg['vertical']} {seg['alt_entry_ft']:.0f}->{seg['alt_exit_ft']:.0f} ft")
    if seg["speed"] in ("accel", "decel"):
        parts.append(f"{seg['speed']} {seg['spd_entry_kt']:.0f}->{seg['spd_exit_kt']:.0f} kt")
    if not parts:
        parts.append(f"steady {seg['hdg_entry']:.0f}deg {seg['alt_entry_ft']:.0f} ft")
    return f"{seg['callsign']}: " + ", ".join(parts) + f" [{seg['duration_s']:.0f}s]"
