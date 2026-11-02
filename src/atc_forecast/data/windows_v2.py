"""Build the v2 trajectory dataset: strict 1 Hz windows around prediction times.

    python -m atc_forecast.data.windows_v2 --adsb "data/adsb/adsb_*.parquet" --out data/v2

What changed from v1 (data/windows.py), and why:

  prediction time   t0 = onset - delta, delta ~ U(2, 12) s, three draws per maneuver. The
                    message precedes the maneuver; v1 put t0 at the onset itself.
  steady windows    t0 inside steady segments too, so "nothing commanded" is not the same
                    as "nothing happening".
  full futures      every window has all 60 future seconds; nothing is cut at the next
                    maneuver. The next onset is recorded (next_onset_rel) so results can be
                    split by whether an unannounced maneuver follows.
  no masks          a window with a raw gap over 4 s on either side is dropped, not masked.
  no leakage        history and future are resampled separately (resample.py).
  ground removed    on-ground rows and windows slower than 50 kt anywhere are dropped.
  pressure altitude what ATC assigns; geometric altitude sits a weather-dependent few
                    hundred feet away and breaks the rounding to assigned levels.

Outputs in --out:

  arrays/hist.npy      [N, 30, 9]  float32   history features (frames.HIST_FEATURES)
  arrays/ctx.npy       [N, 5]      float32   context (frames.CTX_FEATURES)
  arrays/center.npy    [N, 3]      float32   centering velocity c (m/s, track frame)
  arrays/fut_vel.npy   [N, 60, 3]  float32   future 1 s velocities (m/s, track frame)
  arrays/raw_future.npy [M, 100, 4] float32  raw future (t, x, y, z) for val/test windows,
                                             NaN padded; raw_row in meta points into it
  windows.parquet      one row per window: ids, split, kind, delta, labels, atoms, state
  segments.parquet     the v2 segmentation (pressure altitude)
  manifest.json        config, counts, rejection reasons, target scales, PCA basis
"""

import argparse
import glob
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import atoms as A
from .frames import CTX_FEATURES, HIST_FEATURES, make_features, pca_basis, step_scale
from .legs import build_leg_summary, build_segment_dataset
from .resample import REJECT_REASONS, WindowSpec, gather, prepare_track, resample

RAW_PAD = 100
CHUNK = 4096


@dataclass(frozen=True)
class BuildConfig:
    spec: WindowSpec = field(default_factory=WindowSpec)
    delays: int = 3                 # delay draws per maneuver
    delay_lo: float = 2.0
    delay_hi: float = 12.0
    steady_every_s: float = 60.0    # one steady window per this much steady flight
    steady_max: int = 3
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 0


def leg_split(leg_ids, val_frac=0.1, test_frac=0.1) -> np.ndarray:
    """Train/val/test by a hash of the leg id: stable across rebuilds and day additions,
    and a leg never straddles two splits."""
    u = np.array([int(hashlib.md5(str(x).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                  for x in leg_ids])
    return np.where(u < test_frac, "test", np.where(u < test_frac + val_frac, "val", "train"))


def _requests(seg: pd.DataFrame, cfg: BuildConfig, rng) -> pd.DataFrame:
    """Prediction times: delayed maneuver onsets and points inside steady segments."""
    seg = seg.sort_values(["leg_id", "seg_idx"]).reset_index(drop=True)
    man = seg[seg["is_maneuver"]]
    rows = []
    for d in range(cfg.delays):
        delta = rng.uniform(cfg.delay_lo, cfg.delay_hi, len(man))
        rows.append(pd.DataFrame({"seg_row": man.index, "kind": "maneuver", "draw": d,
                                  "delta": delta, "t0": man["t_start"].to_numpy() - delta}))
    st = seg[~seg["is_maneuver"]]
    n = np.clip((st["duration_s"] // cfg.steady_every_s).astype(int), 1, cfg.steady_max)
    rep = st.loc[st.index.repeat(n)]
    u = rng.uniform(0, 1, len(rep))
    rows.append(pd.DataFrame({"seg_row": rep.index, "kind": "steady",
                              "draw": rep.groupby(level=0).cumcount().to_numpy(),
                              "delta": np.nan,
                              "t0": rep["t_start"].to_numpy() + u * rep["duration_s"].to_numpy()}))
    req = pd.concat(rows, ignore_index=True)
    # The first maneuver onset after the window's own message (maneuver windows) or after
    # t0 (steady windows): where an unannounced maneuver may start inside the future.
    onsets = man.groupby("leg_id")["t_start"].apply(np.sort).to_dict()
    legs = seg["leg_id"].to_numpy()[req["seg_row"]]
    after = np.where(req["kind"] == "maneuver",
                     seg["t_start"].to_numpy()[req["seg_row"]], req["t0"])
    nxt = np.array([o[np.searchsorted(o, a, side="right")] if (o := onsets.get(lg)) is not None
                    and np.searchsorted(o, a, side="right") < len(o) else np.inf
                    for lg, a in zip(legs, after)])
    req["next_onset_rel"] = nxt - req["t0"]
    return req


def _labels(seg_rows: pd.DataFrame, kind) -> dict:
    man = (np.asarray(kind) == "maneuver")
    lat = A.lateral_atom(seg_rows["lateral"], seg_rows["hdg_exit"])
    ver = A.vertical_atom(seg_rows["vertical"], seg_rows["alt_exit_ft"])
    spd = A.speed_atom(seg_rows["speed"], seg_rows["spd_exit_kt"])
    return {"atom_lateral": np.where(man, lat, A.NONE),
            "atom_vertical": np.where(man, ver, A.NONE),
            "atom_speed": np.where(man, spd, A.NONE)}


def build_day(path, cfg: BuildConfig, rng):
    day = Path(path).stem.replace("adsb_", "")
    t = time.time()
    raw = pd.read_parquet(path)
    raw["icao24"] = raw["icao24"].astype(str)
    if cfg.spec.altitude == "baro":
        # segmentation reads geo_altitude; give it pressure altitude instead
        raw = raw.assign(geo_altitude=raw["baro_altitude"])
    seg = build_segment_dataset(raw)
    seg["leg_id"] = day + "_" + seg["leg_id"].astype(str)
    seg["day"] = day
    seg = seg.merge(build_leg_summary(seg)[["leg_id", "leg_type"]], on="leg_id", how="left")
    seg = seg.sort_values(["leg_id", "seg_idx"]).reset_index(drop=True)
    req = _requests(seg, cfg, rng)
    req["icao24"] = seg["icao24"].to_numpy()[req["seg_row"]]

    groups = dict(tuple(raw.groupby("icao24", sort=False)))
    reasons = dict.fromkeys(REJECT_REASONS, 0)
    items = []
    for ic, r in req.groupby("icao24", sort=False):
        tr = prepare_track(groups[ic], cfg.spec.altitude) if ic in groups else None
        it, rs = gather(tr, r["t0"].to_numpy(), cfg.spec)
        for x in rs:
            if x is not None:
                reasons[x] += 1
        for x in it:
            x["req"] = r.index[x["i"]]
        items += it
    del raw, groups

    feats, keep_req = [], []
    for i in range(0, len(items), CHUNK):
        chunk = items[i:i + CHUNK]
        w = resample(chunk, cfg.spec)
        reasons["slow"] += len(w["slow"])
        if len(w["keep"]):
            f = make_features(w)
            f["n_hist_raw"] = w["n_hist_raw"]
            f["offset0"] = w["offset0"]
            feats.append(f)
            keep_req += [chunk[k]["req"] for k in w["keep"]]
    req = req.loc[keep_req].reset_index(drop=True)
    sr = seg.loc[req["seg_row"]].reset_index(drop=True)

    meta = pd.DataFrame({
        "day": day, "leg_id": sr["leg_id"], "seg_idx": sr["seg_idx"], "icao24": req["icao24"],
        "callsign": sr["callsign"], "kind": req["kind"], "draw": req["draw"],
        "delta": req["delta"], "t0": req["t0"], "next_onset_rel": req["next_onset_rel"],
        "lateral": sr["lateral"], "vertical": sr["vertical"], "speed": sr["speed"],
        "hdg_exit": sr["hdg_exit"], "alt_exit_ft": sr["alt_exit_ft"],
        "spd_exit_kt": sr["spd_exit_kt"], "phase": sr["phase"], "leg_type": sr["leg_type"],
        "wake": sr["wake"], "engine_class": sr["engine_class"], "n_engines": sr["n_engines"],
        "typecode": sr["typecode"], "ac_class": sr["ac_class"],
    })
    for k, v in _labels(sr, req["kind"]).items():
        meta[k] = v
    cat = lambda k: np.concatenate([f[k] for f in feats])
    for k in ("trk_deg", "alt_ft", "gs_kt", "vr_fpm"):
        meta[k] = np.concatenate([f["state"][k] for f in feats])
    meta["n_hist_raw"] = cat("n_hist_raw")
    off = cat("offset0")
    meta["offset0_h_m"], meta["offset0_v_m"] = np.hypot(off[:, 0], off[:, 1]), off[:, 2]
    arrays = {k: cat(k) for k in ("hist", "ctx", "center", "fut_vel")}
    raw_future = [r for f in feats for r in f["raw_future"]]
    print(f"v2 - {day}: {len(seg)} segments, {len(req)} windows "
          f"({(meta['kind'] == 'maneuver').sum()} maneuver) in {time.time() - t:.0f}s; "
          f"rejected {reasons}", flush=True)
    return seg, meta, arrays, raw_future, reasons


def build(day_files, out_dir, cfg: BuildConfig = BuildConfig()):
    out = Path(out_dir)
    (out / "arrays").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    segs, metas, arrs, raws = [], [], [], []
    reasons = dict.fromkeys(REJECT_REASONS, 0)
    for p in sorted(day_files):
        seg, meta, a, rf, rs = build_day(p, cfg, rng)
        segs.append(seg)
        metas.append(meta)
        arrs.append(a)
        raws.append(rf)
        for k, v in rs.items():
            reasons[k] += v
    meta = pd.concat(metas, ignore_index=True)
    meta.insert(0, "window_id", np.arange(len(meta)))
    meta["split"] = leg_split(meta["leg_id"], cfg.val_frac, cfg.test_frac)
    arrays = {k: np.concatenate([a[k] for a in arrs]) for k in arrs[0]}
    raw_future = [r for rf in raws for r in rf]

    # raw future only for evaluation windows
    ev = np.where(meta["split"].to_numpy() != "train")[0]
    rf = np.full((len(ev), RAW_PAD, 4), np.nan, np.float32)
    for j, i in enumerate(ev):
        r = raw_future[i][:RAW_PAD]
        rf[j, :len(r)] = r
    meta["raw_row"] = -1
    meta.loc[ev, "raw_row"] = np.arange(len(ev))

    for k, v in arrays.items():
        np.save(out / "arrays" / f"{k}.npy", v)
    np.save(out / "arrays" / "raw_future.npy", rf)
    meta.to_parquet(out / "windows.parquet", index=False)
    pd.concat(segs, ignore_index=True).to_parquet(out / "segments.parquet", index=False)

    # Target statistics from the training split only.
    tr = meta["split"].to_numpy() == "train"
    R = arrays["fut_vel"][tr] - arrays["center"][tr][:, None, :]
    scale = step_scale(R)
    mean, comps, kept = pca_basis(R, scale, k=(6, 4, 8))
    Xc = R / scale - mean
    coef = np.concatenate([Xc[..., i] @ comps[i].T for i in range(3)], 1)
    np.savez(out / "basis.npz", scale=scale, mean=mean, lateral_along=comps[0],
             lateral_right=comps[1], vertical=comps[2], coef_std=coef.std(0))
    manifest = {
        "version": 2, "built": time.strftime("%Y-%m-%d %H:%M"),
        "config": asdict(cfg), "days": [Path(p).stem for p in sorted(day_files)],
        "n_windows": len(meta),
        "by_split": meta["split"].value_counts().to_dict(),
        "by_kind": meta["kind"].value_counts().to_dict(),
        "rejected": reasons,
        "hist_features": HIST_FEATURES, "ctx_features": CTX_FEATURES,
        "atom_sizes": A.SIZES,
        "pca_variance_kept": dict(zip(["along", "right", "vertical"], kept)),
        "full_60s_without_next_onset": float((meta["next_onset_rel"] > 60).mean()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps(manifest, indent=2, default=str))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adsb", default="data/adsb/adsb_*.parquet")
    ap.add_argument("--out", default="data/v2")
    ap.add_argument("--delays", type=int, default=3)
    ap.add_argument("--delay-lo", type=float, default=2.0)
    ap.add_argument("--delay-hi", type=float, default=12.0)
    ap.add_argument("--gap", type=float, default=4.0)
    ap.add_argument("--altitude", default="baro", choices=("baro", "geo"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    days = sorted(glob.glob(a.adsb))
    if not days:
        raise SystemExit(f"no files matched {a.adsb}")
    cfg = BuildConfig(spec=WindowSpec(gap_s=a.gap, altitude=a.altitude), delays=a.delays,
                      delay_lo=a.delay_lo, delay_hi=a.delay_hi, seed=a.seed)
    build(days, a.out, cfg)


if __name__ == "__main__":
    main()
