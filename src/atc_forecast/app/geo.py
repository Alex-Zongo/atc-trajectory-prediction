"""Geography for the viewer: absolute positions and the airspace around them.

The Stage-2 windows store positions RELATIVE to the clearance onset, in metres east and
north of it, so placing a forecast on a map needs the onset's latitude and longitude. Those
come from the raw ADS-B the windows were cut from, read per aircraft on demand, with the
same interpolation `data.windows._build_day` used to anchor the window.

The background features (airports, runways, navaids) come from the public OurAirports
dataset, downloaded once into the cache directory. Without network access the map still
draws, just without the reference features.
"""

import glob
import math
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..units import DEG_TO_M, M_TO_FT, MS_TO_FPM, MS_TO_KT, POS_SCALE_M

OURAIRPORTS = "https://davidmegginson.github.io/ourairports-data/{}.csv"
DEFAULT_CACHE = Path("data/cache/ourairports")
AIRPORT_TYPES = ("large_airport", "medium_airport", "small_airport")
NAVAID_TYPES = ("VOR", "VOR-DME", "VORTAC", "DME", "NDB", "TACAN")


class AdsbStore:
    """Raw ADS-B day files, read one aircraft at a time.

    A day file is hundreds of MB, but parquet row-group filtering on icao24 reads one
    aircraft in well under a second, so nothing is held in memory beyond what is shown.
    """

    def __init__(self, pattern: str | None):
        self.files = {}
        for p in sorted(glob.glob(pattern)) if pattern else []:
            self.files[Path(p).stem.replace("adsb_", "")] = Path(p)

    def __bool__(self):
        return bool(self.files)

    def has_day(self, day: str) -> bool:
        return day in self.files

    def aircraft(self, day: str, icao24: str) -> pd.DataFrame:
        return _read_aircraft(str(self.files[day]), str(icao24))


@lru_cache(maxsize=64)
def _read_aircraft(path: str, icao24: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p, filters=[("icao24", "==", icao24)])
    else:
        df = pd.read_csv(p)
        df = df[df["icao24"].astype(str) == icao24]
    return df.sort_values("time_position").drop_duplicates("time_position").reset_index(drop=True)


def clean_track(df: pd.DataFrame) -> pd.DataFrame:
    """The raw samples the windows were built from, in aviation units.

    Rows missing any kinematic field are dropped exactly as `_build_day` drops them, so an
    onset interpolated here lands where the window's origin was.
    """
    need = ["time_position", "latitude", "longitude", "geo_altitude", "velocity",
            "true_track", "vertical_rate"]
    d = df.dropna(subset=need)
    return pd.DataFrame({
        "t": d["time_position"].to_numpy(float),
        "lat": d["latitude"].to_numpy(float), "lon": d["longitude"].to_numpy(float),
        "alt_ft": d["geo_altitude"].to_numpy(float) * M_TO_FT,
        "gs_kt": d["velocity"].to_numpy(float) * MS_TO_KT,
        "trk_deg": d["true_track"].to_numpy(float) % 360,
        "vr_fpm": d["vertical_rate"].to_numpy(float) * MS_TO_FPM,
    })


def onset_latlon(track: pd.DataFrame, onset: float) -> tuple[float, float]:
    return (float(np.interp(onset, track["t"], track["lat"])),
            float(np.interp(onset, track["t"], track["lon"])))


def rel_to_latlon(dx_norm, dy_norm, lat0: float, lon0: float):
    """Window-relative normalised (dx, dy) back to latitude and longitude. The exact inverse
    of the projection in `data.windows._build_day`."""
    cosl = math.cos(math.radians(lat0))
    lon = lon0 + np.asarray(dx_norm) * POS_SCALE_M / (DEG_TO_M * cosl)
    lat = lat0 + np.asarray(dy_norm) * POS_SCALE_M / DEG_TO_M
    return lat, lon


def _fetch(name: str, cache: Path) -> Path | None:
    p = cache / f"{name}.csv"
    if p.exists():
        return p
    try:
        cache.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        urllib.request.urlretrieve(OURAIRPORTS.format(name), tmp)
        tmp.rename(p)
        return p
    except Exception:
        return None


@lru_cache(maxsize=4)
def reference_features(cache: str = str(DEFAULT_CACHE)) -> dict:
    """Airports, runways and navaids worldwide, trimmed to the columns the map uses.
    Each table is empty if it could not be downloaded."""
    cache = Path(cache)
    out = {"airports": pd.DataFrame(), "runways": pd.DataFrame(), "navaids": pd.DataFrame()}
    if (p := _fetch("airports", cache)) is not None:
        a = pd.read_csv(p, usecols=["ident", "type", "name", "latitude_deg", "longitude_deg",
                                    "elevation_ft", "iata_code"], low_memory=False)
        out["airports"] = a[a["type"].isin(AIRPORT_TYPES)].rename(
            columns={"latitude_deg": "lat", "longitude_deg": "lon"})
    if (p := _fetch("runways", cache)) is not None:
        r = pd.read_csv(p, usecols=["airport_ident", "le_ident", "he_ident", "closed",
                                    "le_latitude_deg", "le_longitude_deg",
                                    "he_latitude_deg", "he_longitude_deg"], low_memory=False)
        r = r[r["closed"] == 0].dropna(subset=["le_latitude_deg", "he_latitude_deg"])
        out["runways"] = r.rename(columns={
            "le_latitude_deg": "lat1", "le_longitude_deg": "lon1",
            "he_latitude_deg": "lat2", "he_longitude_deg": "lon2"})
    if (p := _fetch("navaids", cache)) is not None:
        n = pd.read_csv(p, usecols=["ident", "name", "type", "latitude_deg", "longitude_deg"],
                        low_memory=False)
        out["navaids"] = n[n["type"].isin(NAVAID_TYPES)].rename(
            columns={"latitude_deg": "lat", "longitude_deg": "lon"})
    return out


def features_near(lat_min, lat_max, lon_min, lon_max, margin_deg: float = 0.25,
                  cache: str = str(DEFAULT_CACHE)) -> dict:
    """Reference features inside a bounding box padded by `margin_deg`. Small airports are
    kept, since a GA track is often flying between exactly those."""
    f = reference_features(cache)
    box = lambda d, la="lat", lo="lon": d[(d[la] >= lat_min - margin_deg)
                                         & (d[la] <= lat_max + margin_deg)
                                         & (d[lo] >= lon_min - margin_deg)
                                         & (d[lo] <= lon_max + margin_deg)]
    ap = box(f["airports"]) if len(f["airports"]) else f["airports"]
    rw = box(f["runways"], "lat1", "lon1") if len(f["runways"]) else f["runways"]
    if len(rw) and len(ap):
        rw = rw[rw["airport_ident"].isin(ap["ident"])]
    nv = box(f["navaids"]) if len(f["navaids"]) else f["navaids"]
    return {"airports": ap, "runways": rw, "navaids": nv}


def features_payload(feats: dict) -> dict:
    """The features as plain lists, for the JavaScript components."""
    ap = feats["airports"]
    return {
        "airports": [{"id": r.ident, "iata": "" if pd.isna(r.iata_code) else r.iata_code,
                      "name": r.name, "kind": r.type.replace("_airport", ""),
                      "lat": float(r.lat), "lon": float(r.lon)} for r in ap.itertuples()],
        "runways": [{"ap": r.airport_ident, "le": str(r.le_ident), "he": str(r.he_ident),
                     "a": [float(r.lon1), float(r.lat1)], "b": [float(r.lon2), float(r.lat2)]}
                    for r in feats["runways"].itertuples()],
        "navaids": [{"id": r.ident, "name": r.name, "type": r.type,
                     "lat": float(r.lat), "lon": float(r.lon)}
                    for r in feats["navaids"].itertuples()],
    }
