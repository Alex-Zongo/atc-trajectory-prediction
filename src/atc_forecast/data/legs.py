"""Assemble the (segment, clearance) dataset, grouped into flight legs.

One icao24 is not one trajectory: an aircraft can fly several legs in one observation
window, separated by ground time or long coverage dropouts. The trajectory unit is
therefore a LEG, a maximal run of temporally contiguous segments. A leg never spans an
unobserved gap, which would pair clearances across state that was never seen.

Two tables come out:

  segment level  one row per segment, with leg_id, seg_idx, geometry, the clearance text
                 and quality flags. Steady segments are KEPT with an empty clearance, so
                 each leg is a complete ordered trajectory with clearances at maneuver
                 onsets and nothing between.
  leg level      one row per leg: type, extent, counts, concatenated clearance transcript.

Quality is flagged rather than dropped, so leg sequences stay intact and the caller
filters as needed.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .phraseology import add_instructions
from .segmentation import filter_segments, segment_trajectories

LEG_GAP_S = 300.0

TEXT_COLS = ["atc_instruction", "atc_readable", "atc_directive",
             "atc_directive_spoken", "callsign"]

DATASET_COLS = [
    "leg_id", "leg_type", "seg_idx", "icao24", "callsign", "ac_class", "callsign_known",
    "typecode", "engine_class", "wake", "n_engines", "is_vfr", "squawk",
    "t_start", "t_end", "duration_s",
    "atc_directive", "atc_directive_spoken", "atc_instruction", "atc_readable",
    "is_maneuver", "lateral", "vertical", "speed", "phase",
    "hdg_entry", "hdg_exit", "turn_deg",
    "alt_entry_ft", "alt_exit_ft", "alt_change_ft",
    "spd_entry_kt", "spd_exit_kt", "spd_change_kt",
    "lat_start", "lon_start", "lat_end", "lon_end", "mean_alt_ft",
    "coverage", "median_dt_s", "quality_ok",
]


def assign_legs(segs: pd.DataFrame, leg_gap_s: float = LEG_GAP_S) -> pd.DataFrame:
    """Add leg_id and within-leg seg_idx by splitting on inter-segment gaps."""
    if segs.empty:
        return segs.assign(leg_id=pd.Series(dtype=str), seg_idx=pd.Series(dtype=int))
    segs = segs.sort_values(["icao24", "t_start"]).reset_index(drop=True)
    gap = segs["t_start"] - segs.groupby("icao24")["t_end"].shift()
    new_leg = gap.isna() | (gap > leg_gap_s)
    leg_num = new_leg.groupby(segs["icao24"]).cumsum().astype(int)
    segs["leg_id"] = segs["icao24"].astype(str) + "_L" + leg_num.astype(str)
    segs["seg_idx"] = segs.groupby("leg_id").cumcount()
    return segs


def _aircraft_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-aircraft performance metadata plus whether it is under ATC control.

    Performance (typecode, engine_class, wake) must be an explicit conditioning variable:
    the same clearance yields a two- to fourfold different trajectory for a light single
    versus a narrowbody, and without it that variance masquerades as ATC uncertainty.

    `is_vfr` uses the US convention that VFR aircraft squawk 1200 while IFR aircraft get a
    discrete code. A VFR aircraft is self-navigating, so a synthesised clearance for it is
    fiction. It is flagged rather than dropped, since squawk is often missing and
    "unknown" is an honest third level.
    """
    from ..collect.aircraft import annotate_aircraft

    per_ac = annotate_aircraft(pd.DataFrame({"icao24": df["icao24"].astype(str).unique()}))
    if "squawk" in df.columns:
        sq = (df.dropna(subset=["squawk"])
                .assign(squawk=lambda x: x["squawk"].astype(str).str.strip()
                        .str.replace(r"\.0$", "", regex=True))
                .groupby("icao24")["squawk"]
                .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None)
                .rename("squawk").reset_index())
        sq["icao24"] = sq["icao24"].astype(str)
        per_ac = per_ac.merge(sq, on="icao24", how="left")
        per_ac["is_vfr"] = np.where(per_ac["squawk"].isna(), "unknown",
                                    np.where(per_ac["squawk"] == "1200", "vfr", "ifr"))
    else:
        per_ac["squawk"], per_ac["is_vfr"] = None, "unknown"
    return per_ac


def build_segment_dataset(df: pd.DataFrame, leg_gap_s: float = LEG_GAP_S) -> pd.DataFrame:
    """segment -> quality flag -> clearances -> aircraft -> legs."""
    ac = _aircraft_features(df)
    segs = segment_trajectories(df, airborne_only=True)
    if segs.empty:
        return segs
    segs = filter_segments(segs, keep_only=False)
    segs = add_instructions(segs)
    segs["icao24"] = segs["icao24"].astype(str)
    segs = segs.merge(ac, on="icao24", how="left")
    segs = assign_legs(segs, leg_gap_s)
    cols = [c for c in DATASET_COLS if c in segs.columns]
    return segs[cols].sort_values(["leg_id", "seg_idx"]).reset_index(drop=True)


def build_leg_summary(ds: pd.DataFrame) -> pd.DataFrame:
    """One row per leg: type, extent, counts, clearance transcript."""
    if ds.empty:
        return ds
    g = ds.sort_values(["leg_id", "seg_idx"]).groupby("leg_id", sort=False)
    s = g.agg(icao24=("icao24", "first"), callsign=("callsign", "first"),
              n_segments=("seg_idx", "size"), n_maneuvers=("is_maneuver", "sum"),
              t_start=("t_start", "min"), t_end=("t_end", "max"),
              alt_start_ft=("alt_entry_ft", "first"), alt_end_ft=("alt_exit_ft", "last"),
              _amax_e=("alt_entry_ft", "max"), _amax_x=("alt_exit_ft", "max"),
              _amin_e=("alt_entry_ft", "min"), _amin_x=("alt_exit_ft", "min"),
              lat_start=("lat_start", "first"), lon_start=("lon_start", "first"),
              lat_end=("lat_end", "last"), lon_end=("lon_end", "last"),
              mean_coverage=("coverage", "mean")).reset_index()
    s["max_alt_ft"] = s[["_amax_e", "_amax_x"]].max(axis=1)
    s["min_alt_ft"] = s[["_amin_e", "_amin_x"]].min(axis=1)
    s = s.drop(columns=["_amax_e", "_amax_x", "_amin_e", "_amin_x"])
    s["duration_min"] = ((s["t_end"] - s["t_start"]) / 60).round(1)
    s["net_alt_ft"] = s["alt_end_ft"] - s["alt_start_ft"]
    # A real flight reaches altitude while ground movement stays at field elevation, so
    # classify by peak altitude first and only then by the climb/descent profile.
    s["leg_type"] = np.select(
        [s["max_alt_ft"] < 1000, s["net_alt_ft"] < -3000, s["net_alt_ft"] > 3000],
        ["ground", "arrival", "departure"], default="overflight")
    tr = (ds[ds["atc_instruction"] != ""].groupby("leg_id")["atc_readable"]
          .apply(lambda x: "  |  ".join(x)))
    return s.merge(tr.rename("transcript"), on="leg_id", how="left")


def build_dataset(df: pd.DataFrame, leg_gap_s: float = LEG_GAP_S):
    """Both tables, with each segment tagged with its leg_type."""
    ds = build_segment_dataset(df, leg_gap_s)
    summary = build_leg_summary(ds)
    if not ds.empty:
        ds = ds.merge(summary[["leg_id", "leg_type"]], on="leg_id", how="left")
        ds = (ds[[c for c in DATASET_COLS if c in ds.columns]]
              .sort_values(["leg_id", "seg_idx"]).reset_index(drop=True))
    return ds, summary


def _read_any(path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def build_dataset_from_files(paths, leg_gap_s: float = LEG_GAP_S):
    """Build from several day files ONE DAY AT A TIME.

    Raw ADS-B is large (millions of rows per day) but segments are not, so segmenting each
    day separately and concatenating only the segment tables keeps memory flat as days are
    added. leg_id is prefixed with the day to stay unique, since the same aircraft flies on
    many days.
    """
    frames = []
    for p in paths:
        p = Path(p)
        day = p.stem.replace("adsb_", "")
        df = _read_any(p)
        print(f"legs - {day}: {len(df)} rows, {df['icao24'].nunique()} aircraft", flush=True)
        ds = build_segment_dataset(df, leg_gap_s)
        if ds.empty:
            continue
        ds["leg_id"] = day + "_" + ds["leg_id"].astype(str)
        ds["day"] = day
        frames.append(ds)
        del df
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    ds = pd.concat(frames, ignore_index=True)
    summary = build_leg_summary(ds)
    summary = summary.merge(ds[["leg_id", "day"]].drop_duplicates(), on="leg_id", how="left")
    ds = ds.merge(summary[["leg_id", "leg_type"]], on="leg_id", how="left")
    cols = [c for c in DATASET_COLS if c in ds.columns] + ["day"]
    return ds[cols].sort_values(["leg_id", "seg_idx"]).reset_index(drop=True), summary


def save_dataset(ds: pd.DataFrame, summary: pd.DataFrame, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(out_dir / "segments.parquet", index=False)
    summary.to_parquet(out_dir / "legs.parquet", index=False)
    # the Stage-1 trainer reads meta.parquet; it is the maneuver subset of the segments
    man = ds[ds["is_maneuver"]].reset_index(drop=True)
    man.to_parquet(out_dir / "meta.parquet", index=False)
    print(f"legs - {len(ds)} segments ({len(man)} maneuvers) in "
          f"{summary['leg_id'].nunique()} legs -> {out_dir}")


def load_dataset(out_dir):
    """Load back with text columns normalised to "" rather than NaN.

    An empty clearance means "no instruction was issued here", not missing data, so always
    load through this helper.
    """
    out_dir = Path(out_dir)
    ds = pd.read_parquet(out_dir / "segments.parquet")
    summary = pd.read_parquet(out_dir / "legs.parquet")
    for c in TEXT_COLS:
        if c in ds.columns:
            ds[c] = ds[c].fillna("")
    if "transcript" in summary.columns:
        summary["transcript"] = summary["transcript"].fillna("")
    return ds, summary
