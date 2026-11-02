"""ADS-B from OpenSky or adsb.lol to the state-vector table the window builder reads.

Everything downstream (data/resample.py) reads OpenSky-style state vectors in SI units:

    icao24, callsign, time_position (unix s), latitude, longitude, baro_altitude (m),
    geo_altitude (m), velocity (m/s), true_track (deg), vertical_rate (m/s), on_ground

Sources:

  opensky   parquet or CSV with those columns (collect/opensky.py writes them), or with
            pyopensky's raw history columns (lat, lon, baroaltitude, heading, vertrate,
            lastposupdate, ...), which are renamed and converted
  readsb    readsb / tar1090 trace files, the format of adsb.lol's daily history archives
            (globe_history: traces/<xx>/trace_full_<icao>.json, gzip-compressed with or
            without a .gz suffix). Each trace row is
              [dt, lat, lon, alt_baro_ft | "ground", gs_kt, track_deg, flags, vrate_fpm,
               details | null, source, alt_geom_ft, ...]
            relative to the file's "timestamp". Rows flagged stale (flags & 1) are dropped,
            as OpenSky's stale repeats are; feet, knots and ft/min become SI. The vertical
            rate is the barometric rate unless flags & 4 marks it geometric; either is
            only used as a filter measurement, the altitude itself is pressure altitude.

    states = load_states(["traces/**/*.json"], source="readsb")

The aircraft type features (wake, engine class) come from the OpenSky aircraft database
by icao24 (collect/aircraft.py), whatever the surveillance source.
"""

import glob
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..collect.opensky import OUT_COLS, _postprocess

FT_TO_M = 0.3048
KT_TO_MS = 0.514444
FPM_TO_MS = FT_TO_M / 60.0
COLS = ["icao24", "callsign", "time_position", "latitude", "longitude", "baro_altitude",
        "geo_altitude", "velocity", "true_track", "vertical_rate", "on_ground"]


def _expand(paths) -> list[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out = []
    for p in paths:
        hits = sorted(glob.glob(str(p), recursive=True))
        out += [Path(h) for h in hits] if hits else [Path(p)]
    return [p for p in out if p.is_file()]


def _read_json(path: Path) -> dict:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def read_opensky(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "time_position" not in df.columns or "latitude" not in df.columns:
        df = _postprocess(df)                      # pyopensky raw columns
    if "time_position" not in df.columns and "time" in df.columns:
        df["time_position"] = df["time"]
    for c in COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[COLS]


def read_readsb(path: Path) -> pd.DataFrame:
    d = _read_json(path)
    rows = d.get("trace") or []
    if not rows:
        return pd.DataFrame(columns=COLS)
    t0 = float(d["timestamp"])
    n = len(rows)
    col = lambda i: [r[i] if len(r) > i else None for r in rows]
    alt = col(3)
    ground = np.array([a == "ground" for a in alt])
    num = lambda xs: pd.to_numeric(pd.Series(xs, dtype=object), errors="coerce").to_numpy(float)
    flags = num(col(6))
    flags = np.where(np.isnan(flags), 0, flags).astype(int)
    callsign, cur = [], None
    for det in col(8):                             # details appear on some rows only
        if isinstance(det, dict) and det.get("flight"):
            cur = str(det["flight"]).strip()
        callsign.append(cur)
    df = pd.DataFrame({
        "icao24": str(d.get("icao", path.stem.split("_")[-1])).lower().lstrip("~"),
        "callsign": callsign,
        "time_position": t0 + num(col(0)),
        "latitude": num(col(1)), "longitude": num(col(2)),
        "baro_altitude": np.where(ground, np.nan, num(alt)) * FT_TO_M,
        "geo_altitude": num(col(10)) * FT_TO_M,
        "velocity": num(col(4)) * KT_TO_MS, "true_track": num(col(5)),
        "vertical_rate": num(col(7)) * FPM_TO_MS, "on_ground": ground,
    })
    return df[(flags & 1) == 0].reset_index(drop=True)


def load_states(paths, source: str = "auto") -> pd.DataFrame:
    """State vectors from one or more files (globs allowed), in OpenSky columns and SI."""
    files = _expand(paths)
    if not files:
        raise FileNotFoundError(f"no ADS-B files match {paths}")
    parts = []
    for f in files:
        src = source
        if src == "auto":
            src = "readsb" if (".json" in f.suffixes or f.name.startswith("trace")) else "opensky"
        parts.append(read_readsb(f) if src == "readsb" else read_opensky(f))
    df = pd.concat(parts, ignore_index=True)
    df["icao24"] = df["icao24"].astype(str).str.lower().str.strip()
    df["callsign"] = df["callsign"].astype("string").str.strip()
    return df.sort_values(["icao24", "time_position"]).reset_index(drop=True)


def callsign_at(states: pd.DataFrame, icao24: str, t0: float, window_s: float = 600.0):
    """The callsign an aircraft broadcast last at or before t0 (within window_s), else the
    first one after it. An airframe flies several flight numbers a day, so the nearest
    broadcast, not the most frequent one, is its callsign at t0."""
    s = states[(states["icao24"] == icao24) & ((states["time_position"] - t0).abs() <= window_s)]
    s = s[s["callsign"].notna() & (s["callsign"].str.len() > 0)]
    if s.empty:
        return None
    before = s[s["time_position"] <= t0]
    row = before.iloc[before["time_position"].argmax()] if len(before) else \
        s.iloc[s["time_position"].argmin()]
    return str(row["callsign"])


def icao_for_callsign(states: pd.DataFrame, callsign: str, t0: float, window_s: float = 300.0):
    """The aircraft broadcasting `callsign` (ICAO form, e.g. UAL123) closest to t0."""
    cs = str(callsign).strip().upper().replace(" ", "")
    s = states[(states["callsign"].str.upper().str.replace(" ", "") == cs)
               & ((states["time_position"] - t0).abs() <= window_s)]
    if s.empty:
        return None
    return str(s.iloc[(s["time_position"] - t0).abs().argmin()]["icao24"])


def aircraft_near(states: pd.DataFrame, t0: float, lat: float, lon: float, radius_nm: float,
                  max_age_s: float = 10.0) -> list[str]:
    """Aircraft airborne within radius_nm of (lat, lon) at t0, nearest first: the traffic
    that could share the frequency, for addressee resolution."""
    s = states[(states["time_position"] <= t0) & (states["time_position"] >= t0 - max_age_s)
               & ~states["on_ground"].fillna(False).astype(bool)]
    s = s.sort_values("time_position").groupby("icao24").tail(1)
    dn = (s["latitude"] - lat) * 60.0
    de = (s["longitude"] - lon) * 60.0 * np.cos(np.radians(lat))
    d = np.hypot(dn, de)
    s = s.assign(d=d)[d <= radius_nm].sort_values("d")
    return s["icao24"].tolist()


__all__ = ["load_states", "read_opensky", "read_readsb", "callsign_at", "icao_for_callsign",
           "aircraft_near", "OUT_COLS"]
