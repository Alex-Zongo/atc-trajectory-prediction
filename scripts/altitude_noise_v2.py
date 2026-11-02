"""Does the vertical filter setting explain the small altitude wiggles on level flight?

    uv run python scripts/altitude_noise_v2.py --out results/v2/altitude_noise.json

Rebuilds 400 level windows and 400 climb/descent windows of one day from raw ADS-B under
four settings of the vertical Kalman noise (data/kalman.py VERTICAL), and reports for each:
the wiggle of the smoothed future on level flight (peak-to-peak after removing a straight
line, ft) and how closely the smoothed altitude follows the raw reports (RMS, ft), on level
flight and through vertical maneuvers. Raw barometric altitude comes in 25 ft steps and
the vertical rate in 64 fpm steps, so a few feet of either is below what the data resolves.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import atc_forecast.data.resample as RS
from atc_forecast.data.kalman import AxisNoise
from atc_forecast.units import M_TO_FT

VARIANTS = {
    "current (sig_v 0.8 m/s)": AxisNoise(q=0.15, sig_p=4.0, sig_v=0.8, sig_a0=1.0),
    "sig_v 2.0 m/s": AxisNoise(q=0.15, sig_p=4.0, sig_v=2.0, sig_a0=1.0),
    "sig_v 2.0 m/s, q 0.08": AxisNoise(q=0.08, sig_p=4.0, sig_v=2.0, sig_a0=1.0),
    "no vertical rate (sig_v 20 m/s)": AxisNoise(q=0.15, sig_p=4.0, sig_v=20.0, sig_a0=1.0),
}


def score(w, level):
    t = np.arange(1, 61)
    wig, rms = [], []
    for k in range(len(w["fut_pos"])):
        z = w["fut_pos"][k][:, 2] * M_TO_FT
        r = w["raw_future"][k]
        tr, zr = r[:, 0], (r[:, 3] - w["offset0"][k][2]) * M_TO_FT
        ok = (tr >= 1) & (tr <= 60)
        zi = np.interp(tr, np.r_[0, t], np.r_[0, z])
        rms.append(np.sqrt(np.mean((zi[ok] - zr[ok]) ** 2)))
        if level:
            wig.append(np.ptp(z - np.polyval(np.polyfit(t, z, 1), t)))
    out = {"rms_vs_raw_ft": float(np.median(rms)), "rms_p90": float(np.percentile(rms, 90))}
    if level:
        out |= {"wiggle_p50": float(np.median(wig)), "wiggle_p90": float(np.percentile(wig, 90))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/v2")
    ap.add_argument("--adsb", default="data/adsb")
    ap.add_argument("--day", default="2026-01-15")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="results/v2/altitude_noise.json")
    a = ap.parse_args()
    m = pd.read_parquet(Path(a.data) / "windows.parquet")
    m = m[m["day"] == a.day]
    sets = {
        "level": m[(m["kind"] == "steady") & (m["vr_fpm"].abs() < 100)
                   & (m["next_onset_rel"] > 60)].sample(a.n, random_state=0),
        "vertical maneuver": m[(m["kind"] == "maneuver")
                               & m["vertical"].isin(["climb", "descend"])].sample(a.n, random_state=0),
    }
    raw = pd.read_parquet(Path(a.adsb) / f"adsb_{a.day}.parquet")
    out = {}
    for name, noise in VARIANTS.items():
        RS.VERTICAL = noise                      # _run reads the module constant
        out[name] = {tag: score(RS.windows_from_adsb(raw, req[["icao24", "t0"]])[0],
                                tag == "level") for tag, req in sets.items()}
        print(name, json.dumps(out[name]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
