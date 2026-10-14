import numpy as np
import pandas as pd
import pytest


def _leg(icao, t0, n=400, callsign="SWA1234"):
    """One synthetic leg: a climb, then a right turn, then level flight.

    Built kinematically so segmentation has real change points to find, which is what the
    tests need; it is not meant to be physically exact.
    """
    t = t0 + np.arange(n, dtype=float)
    alt = np.concatenate([np.linspace(2000, 9000, n // 2),
                          np.full(n - n // 2, 9000.0)]) / 3.28084
    trk = np.concatenate([np.full(n // 3, 90.0),
                          np.linspace(90, 200, n // 3),
                          np.full(n - 2 * (n // 3), 200.0)])
    vel = np.concatenate([np.full(n // 2, 180.0), np.linspace(180, 250, n - n // 2)]) / 1.94384
    vr = np.gradient(alt, t)
    lat = 37.6 + np.cumsum(np.cos(np.radians(trk)) * vel) / 111320.0
    lon = -122.4 + np.cumsum(np.sin(np.radians(trk)) * vel) / (111320.0 * np.cos(np.radians(37.6)))
    return pd.DataFrame({
        "time": t, "time_position": t, "last_contact": t, "icao24": icao,
        "callsign": callsign, "latitude": lat, "longitude": lon,
        "geo_altitude": alt, "baro_altitude": alt, "velocity": vel,
        "true_track": trk % 360, "vertical_rate": vr,
        "on_ground": False, "squawk": "2451", "spi": False,
    })


@pytest.fixture(scope="session")
def adsb():
    """A small multi-aircraft ADS-B frame."""
    return pd.concat([_leg(f"abc{i:03d}", 1_700_000_000 + 10_000 * i, callsign=f"SWA12{i:02d}")
                      for i in range(6)], ignore_index=True)


@pytest.fixture(scope="session")
def adsb_file(adsb, tmp_path_factory):
    d = tmp_path_factory.mktemp("adsb")
    p = d / "adsb_2026-01-15.parquet"
    adsb.to_parquet(p, index=False)
    return p
