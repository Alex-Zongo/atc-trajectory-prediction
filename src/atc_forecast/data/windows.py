"""Slice flight legs into (past, clearance, aircraft, future) windows for Stage 2.

Three leakage cuts are baked in:

  1. split by leg      a leg is atomic, so no trajectory point can appear on both sides of
                       a train/val split.
  2. past before onset the past window is [onset - P, onset); the aircraft has not yet
                       begun the maneuver, so the clearance cannot be read off it.
  3. future truncated  each window's future ends at the NEXT clearance onset, so it is
     at the next onset  explained by exactly one clearance. A later clearance would be an
                       unconditioned confound. Padded to H with a validity mask.

Features (normalised by the fixed physical constants in units.py):

  past        [P x 7]  [dx, dy, alt, gs, hdg_sin, hdg_cos, vrate] relative to onset,
                       or [P x 8] with a trailing "observed" flag in grid mode
  fut_ctrl    [H x 4]  [gs, hdg_sin, hdg_cos, vrate]   <- the flow's target
  fut_pos     [H x 3]  [dx, dy, alt]                   true positions, for scoring only
  directive   [10]     the clean structured clearance
  aircraft             (wake, engine_class, n_engines)

The future is CONTROLS, which is what a clearance commands; positions are dead-reckoned
from them for the distance metrics.

Two past representations are available. The default resamples onto a uniform 1 Hz grid,
which linearly interpolates every slot and so smooths turns and level-offs across gaps up
to GAP_TH. With `raw_past=True` the raw ADS-B samples are kept as a variable-length token
set, each carrying its own time_to_onset in seconds, so spacing and decimation are encoded
from the actual timestamps rather than assumed uniform.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..units import (ALT_SCALE_FT, DEG_TO_M, GS_SCALE_KT, MS_TO_FPM, MS_TO_KT, M_TO_FT,
                     POS_SCALE_M, VR_SCALE_FPM)
from .phraseology import round_heading, round_speed

P_STEPS = 30
H_STEPS = 60
DT = 1.0
# A grid step interpolated across a raw-sample gap wider than this is FABRICATED, because
# real ADS-B drops out. Such steps are excluded from the future (no fake targets) and
# masked out of the past.
GAP_TH = 12.0
PAST_WINDOW_S = 30.0
MAX_OBS = 30

NEEDED = ["time_position", "latitude", "longitude", "geo_altitude", "velocity",
          "true_track", "vertical_rate"]


def add_next_onset(meta: pd.DataFrame) -> pd.DataFrame:
    """Per leg, the next segment's onset: the future-truncation boundary."""
    meta = meta.sort_values(["leg_id", "seg_idx"]).copy()
    meta["next_onset"] = meta.groupby("leg_id")["t_start"].shift(-1)
    return meta


def directive_vector(row) -> np.ndarray:
    """The clean 10-dim structured clearance, quantised exactly as it would be spoken.

    Quantising here (and not at the raw trajectory value) matters: otherwise phraseology
    rounding, 4,351 ft becoming "4,000", would later look like recognition error.
    """
    ht = row.lateral in ("L", "R")
    hv = row.vertical in ("climb", "descend")
    hs = row.speed in ("accel", "decel")
    ih = math.radians(round_heading(float(row.hdg_exit))) if ht else 0.0
    ialt = (max(round(float(row.alt_exit_ft) / 1000.0), 1) * 1000) if hv else 0.0
    ikt = round_speed(float(row.spd_exit_kt)) if hs else 0.0
    return np.array([
        float(ht), float(row.lateral == "R"), math.sin(ih) * ht, math.cos(ih) * ht,
        float(hv), float(row.vertical == "climb"), ialt / ALT_SCALE_FT,
        float(hs), float(row.speed == "accel"), ikt / GS_SCALE_KT,
    ], dtype=np.float32)


def _build_day(meta_day, raw, p_steps=P_STEPS, h_steps=H_STEPS, raw_past=False,
               past_window_s=PAST_WINDOW_S, max_obs=MAX_OBS):
    raw = raw.dropna(subset=NEEDED).sort_values(["icao24", "time_position"])
    groups = {ic: g for ic, g in raw.groupby("icao24", sort=False)}
    keys = ("past", "past_mask", "past_time", "fut_ctrl", "fut_mask", "fut_pos", "directive",
            "n_engines", "leg_id", "icao24", "wake", "engine_class", "day", "seg_idx", "onset")
    out = {k: [] for k in keys}

    for row in meta_day.itertuples():
        g = groups.get(row.icao24)
        if g is None:
            continue
        t = g["time_position"].to_numpy(float)
        onset = float(row.t_start)
        if onset < t.min() or onset > t.max():
            continue

        lat = g["latitude"].to_numpy(float)
        lon = g["longitude"].to_numpy(float)
        alt = g["geo_altitude"].to_numpy(float) * M_TO_FT
        gs = g["velocity"].to_numpy(float) * MS_TO_KT
        trk = np.radians(g["true_track"].to_numpy(float))
        vr = g["vertical_rate"].to_numpy(float) * MS_TO_FPM
        lat0, lon0 = float(np.interp(onset, t, lat)), float(np.interp(onset, t, lon))
        cosl = math.cos(math.radians(lat0))

        def feats(grid):
            dx = (np.interp(grid, t, lon) - lon0) * DEG_TO_M * cosl
            dy = (np.interp(grid, t, lat) - lat0) * DEG_TO_M
            hs_ = np.interp(grid, t, np.sin(trk))
            hc_ = np.interp(grid, t, np.cos(trk))
            n = np.hypot(hs_, hc_)
            n[n == 0] = 1.0
            return (dx, dy, np.interp(grid, t, alt), np.interp(grid, t, gs),
                    hs_ / n, hc_ / n, np.interp(grid, t, vr))

        def real_cover(grid):
            """True where a grid time is real coverage: in bounds AND not interpolated
            across a raw-sample gap wider than GAP_TH."""
            i = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
            span = t[i] - t[i - 1]
            return (grid >= t.min()) & (grid <= t.max()) & (span <= GAP_TH)

        fg = onset + np.arange(h_steps) * DT
        fdx, fdy, fa, fs, fhs, fhc, fv = feats(fg)
        end = min(onset + h_steps * DT, t.max() + 1.0,
                  float(row.next_onset) if not pd.isna(row.next_onset) else np.inf)
        fm = (((fg >= onset) & (fg < end)) & real_cover(fg)).astype(np.float32)
        if fm.sum() < 5:
            continue

        if raw_past:
            # Raw observations in [onset - window, onset), no interpolation, each token
            # keeping its own time_to_onset. Padded to max_obs and BACK-filled so the LAST
            # row is the most recent real sample: dead reckoning reads the onset altitude
            # as past[:, -1, 2], so front-filling would put a zero-padding row there and
            # start the altitude integral from zero.
            sel = np.where((t >= onset - past_window_s) & (t < onset))[0]
            if len(sel) < 5:
                continue
            sel = sel[-max_obs:]
            n = len(sel)
            feat = np.stack([
                (lon[sel] - lon0) * DEG_TO_M * cosl / POS_SCALE_M,
                (lat[sel] - lat0) * DEG_TO_M / POS_SCALE_M,
                alt[sel] / ALT_SCALE_FT, gs[sel] / GS_SCALE_KT,
                np.sin(trk[sel]), np.cos(trk[sel]), vr[sel] / VR_SCALE_FPM], 1).astype(np.float32)
            past = np.zeros((max_obs, 7), np.float32)
            pm = np.zeros(max_obs, np.float32)
            ptime = np.zeros(max_obs, np.float32)
            past[max_obs - n:] = feat
            pm[max_obs - n:] = 1.0
            ptime[max_obs - n:] = (onset - t[sel]).astype(np.float32)
            out["past"].append(past)
            out["past_time"].append(ptime)
        else:
            pg = onset - np.arange(p_steps, 0, -1) * DT
            pdx, pdy, pa, ps, phs, phc, pv = feats(pg)
            pcov = real_cover(pg)
            pm = pcov.astype(np.float32)
            if pm.sum() < 5:
                continue
            out["past"].append(np.stack([
                pdx / POS_SCALE_M, pdy / POS_SCALE_M, pa / ALT_SCALE_FT, ps / GS_SCALE_KT,
                phs, phc, pv / VR_SCALE_FPM, pm], 1).astype(np.float32))

        out["fut_ctrl"].append(np.stack([fs / GS_SCALE_KT, fhs, fhc,
                                         fv / VR_SCALE_FPM], 1).astype(np.float32))
        out["fut_pos"].append(np.stack([fdx / POS_SCALE_M, fdy / POS_SCALE_M,
                                        fa / ALT_SCALE_FT], 1).astype(np.float32))
        out["past_mask"].append(pm)
        out["fut_mask"].append(fm)
        out["directive"].append(directive_vector(row))
        out["n_engines"].append(float(row.n_engines) if not pd.isna(row.n_engines) else 1.0)
        for k in ("leg_id", "icao24", "wake", "engine_class", "day", "seg_idx"):
            out[k].append(getattr(row, k))
        out["onset"].append(onset)

    return out if out["past"] else None


def build_windows(day_files, meta: pd.DataFrame, out_dir, p_steps=P_STEPS, h_steps=H_STEPS,
                  raw_past=False, past_window_s=PAST_WINDOW_S, max_obs=MAX_OBS):
    """Build and save the Stage-2 windows, one day at a time so memory stays flat.

    Write each variant (different horizon, or raw versus grid past) to a SEPARATE out_dir
    so they can coexist for a sweep.
    """
    meta = add_next_onset(meta)
    acc = None
    for p in day_files:
        p = Path(p)
        day = p.stem.replace("adsb_", "")
        md = meta[meta["day"] == day]
        if md.empty:
            continue
        raw = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        d = _build_day(md, raw, p_steps=p_steps, h_steps=h_steps, raw_past=raw_past,
                       past_window_s=past_window_s, max_obs=max_obs)
        print(f"windows - {day}: {0 if d is None else len(d['past'])}", flush=True)
        if d is None:
            continue
        if acc is None:
            acc = {k: list(v) for k, v in d.items()}
        else:
            for k in acc:
                acc[k].extend(d[k])
        del raw

    if acc is None:
        raise SystemExit("no windows built; check that meta days match the ADS-B files")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["past", "past_mask", "fut_ctrl", "fut_mask", "fut_pos", "directive"]
    if raw_past:
        keys.append("past_time")
    arrays = {k: np.stack(acc[k]) for k in keys}
    arrays["n_engines"] = np.asarray(acc["n_engines"], np.float32)
    np.savez_compressed(out_dir / "windows.npz", **arrays)
    pd.DataFrame({k: acc[k] for k in ("leg_id", "icao24", "wake", "engine_class", "day",
                                      "seg_idx", "onset")}
                 ).to_parquet(out_dir / "windows_meta.parquet")
    print(f"windows - saved {len(arrays['past'])} -> {out_dir}\n"
          f"  past {arrays['past'].shape}  fut_ctrl {arrays['fut_ctrl'].shape}")
    return out_dir


def attach_onset_state(meta: pd.DataFrame, window_dir) -> pd.DataFrame:
    """Join the aircraft's state at clearance onset into the Stage-1 meta.

    meta holds only the clearance TARGETS, so the current state has to come from the past
    trajectory, where onset is the last past row by the back-fill convention. This is an
    inner join, so only segments that have a trajectory window survive; train the grounded
    and ungrounded variants on this same subset for a fair ablation.
    """
    state_cols = ["alt_onset_ft", "vrate_onset_fpm", "gs_onset_kt",
                  "hdg_onset_sin", "hdg_onset_cos"]
    if all(c in meta.columns for c in state_cols):
        # Already joined. Re-merging would suffix the columns to _x/_y, after which the
        # dataset no longer finds "alt_onset_ft" and grounding silently turns itself off.
        print(f"onset state already present on {len(meta)} segments, skipping the join")
        return meta
    wdir = Path(window_dir)
    npz = np.load(wdir / "windows.npz")
    wm = pd.read_parquet(wdir / "windows_meta.parquet")
    past = npz["past"]
    tbl = pd.DataFrame({
        "leg_id": wm["leg_id"].values, "seg_idx": wm["seg_idx"].values,
        "alt_onset_ft": past[:, -1, 2] * ALT_SCALE_FT,
        "vrate_onset_fpm": past[:, -1, 6] * VR_SCALE_FPM,
        "gs_onset_kt": past[:, -1, 3] * GS_SCALE_KT,
        "hdg_onset_sin": past[:, -1, 4],
        "hdg_onset_cos": past[:, -1, 5],
    }).drop_duplicates(["leg_id", "seg_idx"])
    out = meta.merge(tbl, on=["leg_id", "seg_idx"], how="inner")
    print(f"onset state joined onto {len(out)}/{len(meta)} segments from {window_dir}")
    return out
