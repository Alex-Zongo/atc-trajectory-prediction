"""Build the full dataset from raw ADS-B day files.

    python scripts/build_dataset.py --adsb "data/adsb/adsb_*.parquet" --out data/dataset

Three stages, each resumable by pointing at a different --out:

  1. segment every day's tracks into maneuver segments, assemble legs, synthesise the
     clearance for each maneuver onset           -> segments/legs/meta parquet
  2. slice legs into Stage-2 (past, clearance, future) windows   -> windows.npz
  3. join the onset state onto meta, for Stage-1 state grounding -> meta_grounded.parquet

The window step is run twice by default, once for the raw-observation past and once for the
interpolated grid, into separate directories so both stay reproducible. Pass --past to pick
one.
"""

import argparse
import glob
from pathlib import Path

import pandas as pd

from atc_forecast.data.legs import build_dataset_from_files, save_dataset
from atc_forecast.data.windows import attach_onset_state, build_windows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adsb", required=True, help="glob for ADS-B day files")
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--past", default="both", choices=("raw", "grid", "both"))
    ap.add_argument("--h-steps", type=int, default=60, help="prediction horizon (s)")
    ap.add_argument("--p-steps", type=int, default=30, help="past horizon (s), grid mode")
    ap.add_argument("--past-window", type=float, default=30.0, help="raw-past look-back (s)")
    ap.add_argument("--max-obs", type=int, default=30, help="cap on raw past tokens")
    ap.add_argument("--skip-segments", action="store_true",
                    help="reuse an existing segments build and only rebuild windows")
    a = ap.parse_args()

    days = sorted(glob.glob(a.adsb))
    if not days:
        raise SystemExit(f"no files matched {a.adsb}")
    out = Path(a.out)

    if a.skip_segments:
        meta = pd.read_parquet(out / "meta.parquet")
        print(f"reusing {len(meta)} maneuver segments from {out}")
    else:
        print(f"=== segmenting {len(days)} day file(s) ===")
        ds, summary = build_dataset_from_files(days)
        if ds.empty:
            raise SystemExit("no segments produced; check the ADS-B schema")
        save_dataset(ds, summary, out)
        meta = ds[ds["is_maneuver"]].reset_index(drop=True)
        print(f"\nleg types: {summary['leg_type'].value_counts().to_dict()}")

    kinds = ("raw", "grid") if a.past == "both" else (a.past,)
    for kind in kinds:
        wdir = out.parent / f"windows_{kind}"
        print(f"\n=== building {kind} windows -> {wdir} ===")
        build_windows(days, meta, wdir, p_steps=a.p_steps, h_steps=a.h_steps,
                      raw_past=(kind == "raw"), past_window_s=a.past_window,
                      max_obs=a.max_obs)

    # Stage-1 state grounding reads the onset state out of the windows, so this must come
    # last. Inner join, so it restricts to segments that have a trajectory window; train the
    # grounded and ungrounded variants on this same subset for a fair ablation.
    primary = out.parent / f"windows_{'raw' if 'raw' in kinds else kinds[0]}"
    grounded = attach_onset_state(meta, primary)
    grounded.to_parquet(out / "meta_grounded.parquet", index=False)
    print(f"\nwrote {out / 'meta_grounded.parquet'} ({len(grounded)} segments)")
    print(f"\nnext:\n"
          f"  python -m atc_forecast.training.stage2 --data {primary} --out runs/stage2.pt\n"
          f"  python -m atc_forecast.training.stage1 --meta {out / 'meta_grounded.parquet'} "
          f"--window-dir {primary} --state-grounded --out runs/stage1.pt")


if __name__ == "__main__":
    main()
