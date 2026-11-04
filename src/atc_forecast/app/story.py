"""Rebuild one leg of the dataset from its raw ADS-B, keeping every intermediate.

This runs the same functions `scripts/build_dataset.py` runs, on one aircraft, in the same
order: phase labelling, the airborne filter, coverage-gap splitting, `track_signals`
(smoothing, rates, thresholded labels, short-run merging), segment emission, the quality
flag, phraseology, and leg assignment. The window shown at the end is the one stored in
`windows.npz`, not a rebuild of it, and the transcripts come from the observation channel.
"""

import numpy as np
import pandas as pd

from ..data.legs import LEG_GAP_S, assign_legs
from ..data.phraseology import add_instructions
from ..data.segmentation import (NUM_COLS, SegParams, _segment_track, filter_segments,
                                 label_phases, resolve_callsign, track_signals)
from ..data.text import tokenize
from ..data.windows import GAP_TH, H_STEPS, MAX_OBS, PAST_WINDOW_S
from ..units import ALT_SCALE_FT, FL180_M, M_TO_FT, MS_TO_FPM, MS_TO_KT
from .geo import rel_to_latlon
from .observe import Observation, observe


def _r(x, nd=3):
    a = np.asarray(x, dtype=np.float64)
    return np.where(np.isfinite(a), np.round(a, nd), np.nan).tolist()


def _nan_none(v):
    return [None if (isinstance(x, float) and np.isnan(x)) else x for x in v]


def build_story(raw: pd.DataFrame, day: str, leg_id: str, onset: float, window: dict,
                obs: Observation, leg_type=None, n_draws: int = 4,
                params: SegParams = SegParams()) -> dict:
    """The full construction of `leg_id`, focused on the clearance at `onset`.

    `window` is the stored Stage-2 window for that clearance: past (raw tokens), past_time,
    past_mask, fut_pos, fut_mask, directive, all as numpy arrays.
    """
    # 1-2. phases, numeric coercion and the airborne filter, as segment_trajectories does
    df = label_phases(raw)
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    complete = df.dropna(subset=[c for c in NUM_COLS if c in df.columns])
    air = complete[complete["phase"] != "ground"]

    # 3. per-aircraft: dedupe, callsign, split on coverage gaps, as segment_aircraft does
    g = air.sort_values("time_position").drop_duplicates("time_position").reset_index(drop=True)
    g["callsign"] = resolve_callsign(g, g["icao24"].iloc[0])
    t_all = g["time_position"].to_numpy(float)
    splits = np.where(np.diff(t_all) > params.max_gap_s)[0] + 1
    chunks, segs = [], []
    for ci, idx in enumerate(np.split(np.arange(len(g)), splits)):
        if len(idx) < params.min_points:
            chunks.append({"idx": idx, "used": False})
            continue
        sub = g.iloc[idx]
        chunks.append({"idx": idx, "used": True, "sig": track_signals(sub, params)})
        segs.extend(_segment_track(sub, params))
    seg_df = pd.DataFrame(segs)
    seg_df = filter_segments(seg_df, keep_only=False)
    seg_df = add_instructions(seg_df)
    seg_df["icao24"] = seg_df["icao24"].astype(str)
    seg_df = assign_legs(seg_df, LEG_GAP_S)
    seg_df["leg_id"] = day + "_" + seg_df["leg_id"].astype(str)

    legs = (seg_df.groupby("leg_id").agg(t0=("t_start", "min"), t1=("t_end", "max"),
                                         n=("seg_idx", "size"),
                                         n_man=("is_maneuver", "sum"))
            .reset_index().sort_values("t0"))
    leg = seg_df[seg_df["leg_id"] == leg_id].sort_values("seg_idx")
    if leg.empty:
        raise ValueError(f"{leg_id} was not reproduced from the raw ADS-B")
    t0, t1 = float(leg["t_start"].min()), float(leg["t_end"].max())
    lo, hi = t0 - 90.0, t1 + 90.0

    # every raw sample in the leg's span, flagged by why it would be dropped
    span = (df[(df["time_position"] >= lo) & (df["time_position"] <= hi)]
            .sort_values("time_position"))
    is_complete = span[[c for c in NUM_COLS if c in span.columns]].notna().all(axis=1)
    points = {
        "t": _r(span["time_position"] - t0, 2),
        "lat": _r(span["latitude"], 5), "lon": _r(span["longitude"], 5),
        "alt_ft": _r(span["geo_altitude"] * M_TO_FT, 0),
        "gs_kt": _r(span["velocity"] * MS_TO_KT, 1),
        "trk": _r(span["true_track"], 1),
        "vr_fpm": _r(span["vertical_rate"] * MS_TO_FPM, 0),
        "phase": span["phase"].tolist(),
        "status": np.where(~is_complete.to_numpy(), "incomplete",
                           np.where(span["phase"].to_numpy() == "ground", "ground",
                                    "kept")).tolist(),
    }

    sig_out = []
    for c in chunks:
        if not c["used"]:
            continue
        s = c["sig"]
        m = (s["t"] >= lo) & (s["t"] <= hi)
        if not m.any():
            continue
        sub = g.iloc[c["idx"]]
        sig_out.append({
            "t": _r(s["t"][m] - t0, 2),
            "lat": _r(sub["latitude"].to_numpy()[m], 5),
            "lon": _r(sub["longitude"].to_numpy()[m], 5),
            "trk_raw": _r(np.degrees(np.unwrap(
                np.radians(sub["true_track"].to_numpy(float))))[m], 2),
            "alt_raw": _r(sub["geo_altitude"].to_numpy(float)[m] * M_TO_FT, 0),
            "vel_raw": _r(sub["velocity"].to_numpy(float)[m] * MS_TO_KT, 1),
            "trk_sm": _r(s["track_sm"][m], 2), "alt_sm": _r(s["alt_sm"][m] * M_TO_FT, 0),
            "vel_sm": _r(s["vel_sm"][m] * MS_TO_KT, 1),
            "turn_rate": _r(s["turn_rate"][m], 3), "vspeed": _r(s["vspeed"][m], 3),
            "accel": _r(s["accel"][m], 3),
            "lateral": s["lateral"][m].tolist(), "vertical": s["vertical"][m].tolist(),
            "speed": s["speed"][m].tolist(),
            "raw_state": s["raw_state"][m].tolist(), "state": s["state"][m].tolist(),
        })

    seg_rows = []
    for r in seg_df[(seg_df["t_end"] >= lo) & (seg_df["t_start"] <= hi)].itertuples():
        seg_rows.append({
            "leg_id": r.leg_id, "seg_idx": int(r.seg_idx), "in_leg": r.leg_id == leg_id,
            "t0": round(r.t_start - t0, 2), "t1": round(r.t_end - t0, 2),
            "lateral": r.lateral, "vertical": r.vertical, "speed": r.speed,
            "is_maneuver": bool(r.is_maneuver), "quality_ok": bool(r.quality_ok),
            "hdg_entry": r.hdg_entry, "hdg_exit": r.hdg_exit, "turn_deg": r.turn_deg,
            "alt_entry_ft": r.alt_entry_ft, "alt_exit_ft": r.alt_exit_ft,
            "spd_entry_kt": r.spd_entry_kt, "spd_exit_kt": r.spd_exit_kt,
            "duration_s": r.duration_s, "coverage": r.coverage,
            "lat0": r.lat_start, "lon0": r.lon_start, "lat1": r.lat_end, "lon1": r.lon_end,
            "readable": r.atc_readable, "directive": r.atc_directive,
            "spoken": r.atc_directive_spoken, "instruction": r.atc_instruction,
            "is_focus": abs(r.t_start - onset) < 1e-3 and r.leg_id == leg_id,
        })

    # the stored window, placed on the map
    trk = g[["time_position", "latitude", "longitude"]]
    lat0 = float(np.interp(onset, trk["time_position"], trk["latitude"]))
    lon0 = float(np.interp(onset, trk["time_position"], trk["longitude"]))
    pm = window["past_mask"] > 0.5
    plat, plon = rel_to_latlon(window["past"][pm, 0], window["past"][pm, 1], lat0, lon0)
    flat, flon = rel_to_latlon(window["fut_pos"][:, 0], window["fut_pos"][:, 1], lat0, lon0)
    focus = next((s for s in seg_rows if s["is_focus"]), None)
    nxt = leg[(leg["t_start"] > onset + 1e-3) & leg["is_maneuver"]]
    win = {
        "onset": round(onset - t0, 2), "lat0": lat0, "lon0": lon0,
        "next_onset": round(float(nxt["t_start"].min()) - t0, 2) if len(nxt) else None,
        "past_t": _r(-window["past_time"][pm], 2),
        "past_lat": _r(plat, 5), "past_lon": _r(plon, 5),
        "past_alt": _r(window["past"][pm, 2] * ALT_SCALE_FT, 0),
        "fut_lat": _r(flat, 5), "fut_lon": _r(flon, 5),
        "fut_alt": _r(window["fut_pos"][:, 2] * ALT_SCALE_FT, 0),
        "fut_mask": (window["fut_mask"] > 0.5).tolist(),
        "directive": _r(window["directive"], 4),
        "P_window_s": PAST_WINDOW_S, "max_obs": MAX_OBS, "H": H_STEPS, "gap_th": GAP_TH,
    }

    draws = []
    if focus is not None:
        seg = {"lateral": focus["lateral"], "vertical": focus["vertical"],
               "speed": focus["speed"], "hdg_exit": focus["hdg_exit"],
               "alt_exit_ft": focus["alt_exit_ft"], "spd_exit_kt": focus["spd_exit_kt"]}
        rng = np.random.default_rng(obs.seed)
        for _ in range(n_draws):
            o = observe(seg, obs, leg_type=leg_type, phase=None, rng=rng)
            draws.append({"text": o["text"], "core": o["core"], "confidence": o["confidence"],
                          "corrupted": o["corrupted"], "tokens": tokenize(o["text"])})

    return {
        "day": day, "leg_id": leg_id, "icao24": str(g["icao24"].iloc[0]),
        "callsign": str(g["callsign"].iloc[0]), "t0_epoch": t0, "leg_type": leg_type,
        "obs_mode": obs.mode,
        "params": {"turn_rate_th": params.turn_rate_th, "cum_turn_th": params.cum_turn_th,
                   "vs_th": params.vs_th, "accel_th": params.accel_th,
                   "min_seg_s": params.min_seg_s, "max_gap_s": params.max_gap_s,
                   "min_points": params.min_points, "smooth_win": params.smooth_win,
                   "leg_gap_s": LEG_GAP_S, "fl180_ft": FL180_M * M_TO_FT},
        "counts": {"raw": int(len(df)), "complete": int(len(complete)),
                   "airborne": int(len(air)), "chunks": len(chunks),
                   "segments": int(len(seg_df)), "maneuvers": int(seg_df["is_maneuver"].sum()),
                   "legs": int(len(legs))},
        "legs": [{"leg_id": r.leg_id, "t0": round(r.t0 - t0, 1), "t1": round(r.t1 - t0, 1),
                  "n": int(r.n), "n_man": int(r.n_man)} for r in legs.itertuples()],
        "points": points, "signals": sig_out, "segments": seg_rows,
        "window": win, "draws": draws,
    }
