"""Aircraft performance metadata: icao24 to typecode, engine class, wake category.

The same clearance produces very different trajectories depending on the airframe. For
"climb and maintain 5,000" an airline jet averages roughly 220 kt and +2,400 fpm where a
light single averages 100 kt and +620 fpm, so over twice the speed and nearly four times
the climb rate. Left unmodelled that airframe variance masquerades as ATC uncertainty,
which is exactly the quantity Stage 1 is trying to isolate. Performance is therefore an
explicit conditioning variable rather than unexplained noise.

Source: the free OpenSky aircraft database. Its `icaoaircrafttype` field already encodes
what is needed, as [L]andplane/[H]elicopter, engine count, then [P]iston/[T]urboprop/[J]et,
so L2J is a twin jet and L1P a piston single. Wake category by MTOW is ATC's own
performance classification, which makes it the natural conditioning feature.
"""

from pathlib import Path

import pandas as pd

DB_URL = "https://opensky-network.org/datasets/metadata/aircraftDatabase.csv"
DEFAULT_CACHE = Path("data/cache/aircraftDatabase.csv")

# Engine class alone cannot decide wake: a light business jet and a narrowbody are both
# L2J but Light and Medium respectively. The common types are enumerated, with an
# engine-class fallback for the long tail.
_SUPER = {"A388"}
_HEAVY = {
    "B772", "B773", "B77W", "B77L", "B788", "B789", "B78X", "B763", "B764",
    "B744", "B748", "B752", "B753", "A332", "A333", "A338", "A339", "A342",
    "A343", "A345", "A346", "A359", "A35K", "MD11", "IL76", "C17",
}
_MEDIUM = {
    "B737", "B738", "B739", "B736", "B735", "B38M", "B39M", "B3XM", "B712",
    "A319", "A320", "A321", "A318", "A19N", "A20N", "A21N",
    "E170", "E175", "E75L", "E75S", "E190", "E195", "E290", "E295",
    "CRJ2", "CRJ7", "CRJ9", "CRJX", "DH8D", "AT72", "AT76", "AT75",
    "MD82", "MD83", "MD88", "MD90", "B463", "BCS1", "BCS3", "C208",
}
_ENGINE = {"P": "piston", "T": "turboprop", "J": "jet", "E": "electric"}


def _wake(typecode, engine) -> str:
    """ICAO wake turbulence category: S / H / M / L."""
    tc = str(typecode).upper()
    if tc in _SUPER:
        return "S"
    if tc in _HEAVY:
        return "H"
    if tc in _MEDIUM:
        return "M"
    e = str(engine).upper()
    if e.endswith("J"):
        return "M"
    return "L"


def download_db(cache: Path = DEFAULT_CACHE, force: bool = False) -> Path:
    """Download and cache the OpenSky aircraft database (around 95 MB)."""
    cache = Path(cache)
    if cache.exists() and not force:
        return cache
    import requests

    cache.parent.mkdir(parents=True, exist_ok=True)
    print(f"aircraft - downloading {DB_URL}")
    r = requests.get(DB_URL, timeout=300)
    r.raise_for_status()
    cache.write_bytes(r.content)
    return cache


def load_aircraft_db(cache: Path = DEFAULT_CACHE) -> pd.DataFrame:
    """icao24 to typecode, engine_class, n_engines, wake."""
    path = download_db(cache)
    db = pd.read_csv(path, low_memory=False,
                     usecols=["icao24", "typecode", "icaoaircrafttype", "model"])
    db = db.dropna(subset=["icao24"])
    db["icao24"] = db["icao24"].astype(str).str.lower().str.strip()
    iat = db["icaoaircrafttype"].astype(str).str.upper()
    db["engine_class"] = iat.str[-1].map(_ENGINE).fillna("unknown")
    db["n_engines"] = pd.to_numeric(iat.str[1], errors="coerce")
    db["wake"] = [_wake(t, e) for t, e in zip(db["typecode"], iat)]
    db["typecode"] = db["typecode"].fillna("UNKNOWN")
    return db[["icao24", "typecode", "model", "engine_class", "n_engines", "wake"]]


def annotate_aircraft(df: pd.DataFrame, icao_col: str = "icao24",
                      cache: Path = DEFAULT_CACHE) -> pd.DataFrame:
    """Left-join performance features onto any frame with an icao24 column.

    Unmatched aircraft get engine_class and wake set to "unknown", which is an honest
    level the model can condition on rather than a silent default. The models train an
    explicit "unknown" embedding, so this degrades gracefully at inference.
    """
    db = load_aircraft_db(cache)
    out = df.copy()
    out[icao_col] = out[icao_col].astype(str).str.lower().str.strip()
    out = out.merge(db, left_on=icao_col, right_on="icao24", how="left", suffixes=("", "_db"))
    for c, fill in (("typecode", "UNKNOWN"), ("engine_class", "unknown"), ("wake", "unknown")):
        out[c] = out[c].fillna(fill)
    return out.drop(columns=[c for c in out.columns if c.endswith("_db")])
