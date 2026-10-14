"""Historical ADS-B state vectors from OpenSky's Trino backend.

Historical access is a separate grant from a standard OpenSky account; request it from the
account dashboard. `pyopensky` reads credentials, in order of precedence, from the
environment (OPENSKY_USERNAME / OPENSKY_PASSWORD, or OPENSKY_CLIENT_ID /
OPENSKY_CLIENT_SECRET) and then from its own settings.conf. Passing username and password
to `fetch_area_history` sets the environment variables without touching settings.conf.

Queries are sliced hourly and run sequentially. OpenSky's fair-use policy discourages bulk
ad-hoc multi-day pulls, and a small targeted bounding box taken one day at a time is the
polite way to collect at scale.

    from datetime import datetime, timezone
    from atc_forecast.collect.opensky import fetch_days
    fetch_days(["2026-01-15"], out_dir="data/adsb")
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# Wide enough (~60 nm) that approach and departure traffic on the recorded audio
# frequencies has trajectory coverage. Replace with your own airspace.
BAY_AREA_BBOX = dict(lat_min=36.5, lat_max=39.0, lon_min=-123.5, lon_max=-120.8)

RENAME_MAP = {
    "lon": "longitude", "lat": "latitude", "baroaltitude": "baro_altitude",
    "onground": "on_ground", "heading": "true_track", "vertrate": "vertical_rate",
    "geoaltitude": "geo_altitude", "lastposupdate": "time_position",
    "lastcontact": "last_contact",
}

OUT_COLS = [
    "time", "icao24", "callsign", "longitude", "latitude", "baro_altitude", "on_ground",
    "velocity", "true_track", "vertical_rate", "geo_altitude", "squawk", "spi",
    "time_position", "last_contact", "timestamp_utc",
]


def _to_epoch(series: pd.Series) -> pd.Series:
    """Coerce to float epoch seconds, whether the column arrives as a tz-aware datetime
    (pyopensky's default) or as a numeric unix timestamp."""
    if pd.api.types.is_numeric_dtype(series.dtype):
        return pd.to_numeric(series, errors="coerce")
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    ns = dt.to_numpy(dtype="datetime64[ns]").astype("float64")
    return pd.Series(ns / 1e9, index=series.index)


def _postprocess(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=OUT_COLS)
    df = df.copy().rename(columns=RENAME_MAP)
    for col in ("time", "time_position", "last_contact"):
        if col in df.columns:
            df[col] = _to_epoch(df[col])
    for col in ("longitude", "latitude", "baro_altitude", "geo_altitude",
                "velocity", "true_track", "vertical_rate"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "callsign" in df.columns:
        df["callsign"] = df["callsign"].astype("string").str.strip()
    df["timestamp_utc"] = df["time"].apply(
        lambda t: datetime.fromtimestamp(float(t), tz=timezone.utc).isoformat()
        if pd.notna(t) else None)
    for col in OUT_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[OUT_COLS].reset_index(drop=True)


def fetch_area_history(start_utc: datetime, end_utc: datetime, bbox: dict = BAY_AREA_BBOX,
                       chunk_hours: int = 1, username: str | None = None,
                       password: str | None = None, cached: bool = True,
                       drop_stale: bool = True, max_staleness_s: float = 15.0
                       ) -> pd.DataFrame:
    """Fetch state vectors over `bbox` between two timezone-aware UTC datetimes.

    `drop_stale` applies OpenSky's recommended fair-use filter, discarding rows whose
    position is older than `max_staleness_s` relative to last contact. Those are coasted or
    estimated vectors, so keeping them would both violate fair use and corrupt the cadence
    analysis the segmentation depends on.
    """
    if username:
        os.environ["OPENSKY_USERNAME"] = username
    if password:
        os.environ["OPENSKY_PASSWORD"] = password

    # Imported lazily so the module loads without pyopensky or credentials present.
    from pyopensky.trino import Trino

    bounds = (bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"])
    trino = Trino()
    cursor = start_utc.astimezone(timezone.utc)
    end_utc = end_utc.astimezone(timezone.utc)
    chunks = []
    while cursor < end_utc:
        chunk_end = min(cursor + timedelta(hours=chunk_hours), end_utc)
        print(f"opensky - {cursor.isoformat()} -> {chunk_end.isoformat()}")
        c = trino.history(cursor, chunk_end, bounds=bounds, cached=cached)
        print(f"opensky - {0 if c is None else len(c)} rows")
        if c is not None and not c.empty:
            chunks.append(c)
        cursor = chunk_end

    if not chunks:
        return pd.DataFrame(columns=OUT_COLS)
    result = _postprocess(pd.concat(chunks, ignore_index=True))
    if drop_stale and not result.empty:
        # keep rows within threshold, and rows whose staleness is unknown
        keep = ~((result["time"] - result["last_contact"]) > max_staleness_s)
        if int((~keep).sum()):
            print(f"opensky - dropped {int((~keep).sum())} stale vectors")
        result = result[keep].reset_index(drop=True)
    return result


def cadence_report(df: pd.DataFrame, threshold_s: float = 3.0) -> pd.DataFrame:
    """Distribution of the time step between consecutive position reports per aircraft.

    Worth checking before building windows: the window builder assumes a step small enough
    that interpolation onto a 1 Hz grid is honest, and flags anything above GAP_TH as
    fabricated.
    """
    if df.empty:
        return pd.DataFrame()
    d = (df.dropna(subset=["time_position"]).sort_values(["icao24", "time_position"]))
    step = d.groupby("icao24")["time_position"].diff().dropna()
    step = step[step > 0]
    q = step.quantile([0.5, 0.9, 0.99])
    return pd.DataFrame([{
        "aircraft": d["icao24"].nunique(), "steps": len(step),
        "median_s": round(float(q.loc[0.5]), 2), "p90_s": round(float(q.loc[0.9]), 2),
        "p99_s": round(float(q.loc[0.99]), 2),
        f"frac_under_{threshold_s:g}s": round(float((step <= threshold_s).mean()), 3),
    }])


def save_parquet(df: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")
    print(f"opensky - {len(df)} rows -> {path} ({path.stat().st_size / 1e6:.0f} MB)")
    return path


def fetch_days(dates: list[str], out_dir="data/adsb", bbox: dict = BAY_AREA_BBOX,
               chunk_hours: int = 1, **kwargs) -> list[Path]:
    """Fetch whole UTC days, one Parquet file per day.

    Resumable: days whose file already exists are skipped, so an interrupted run can be
    relaunched safely.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for d in dates:
        out = out_dir / f"adsb_{d}.parquet"
        if out.exists():
            print(f"opensky - {d}: already present, skipping")
            written.append(out)
            continue
        day = datetime.fromisoformat(d).replace(tzinfo=timezone.utc)
        print(f"\nopensky - fetching {d}")
        df = fetch_area_history(day, day + timedelta(days=1), bbox=bbox,
                                chunk_hours=chunk_hours, **kwargs)
        if df.empty:
            print(f"opensky - {d}: no data")
            continue
        save_parquet(df, out)
        written.append(out)
    return written


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Fetch historical ADS-B from OpenSky Trino.")
    ap.add_argument("days", nargs="+", help="UTC dates, YYYY-MM-DD")
    ap.add_argument("--out-dir", default="data/adsb")
    ap.add_argument("--lat-min", type=float, default=BAY_AREA_BBOX["lat_min"])
    ap.add_argument("--lat-max", type=float, default=BAY_AREA_BBOX["lat_max"])
    ap.add_argument("--lon-min", type=float, default=BAY_AREA_BBOX["lon_min"])
    ap.add_argument("--lon-max", type=float, default=BAY_AREA_BBOX["lon_max"])
    ap.add_argument("--chunk-hours", type=int, default=1)
    a = ap.parse_args()
    bbox = dict(lat_min=a.lat_min, lat_max=a.lat_max, lon_min=a.lon_min, lon_max=a.lon_max)
    paths = fetch_days(a.days, out_dir=a.out_dir, bbox=bbox, chunk_hours=a.chunk_hours)
    if paths:
        print(cadence_report(pd.read_parquet(paths[-1])).to_string(index=False))


if __name__ == "__main__":
    main()
