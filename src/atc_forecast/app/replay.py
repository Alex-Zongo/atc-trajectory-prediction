"""A whole flight leg, replayed with the models running at every clearance.

The aircraft flies its recorded ADS-B track. At each maneuver onset in the leg the
clearance is observed (clean, through the channel, or as typed), Stage 1 decodes it, and
Stage 2 forecasts the next minute from the past it has actually seen; the aircraft then
keeps flying its true path through that forecast. Every window in a held-out leg is held
out from both models, so nothing shown was trained on.
"""

import numpy as np
import torch

from ..models.stage2 import collate
from ..units import ALT_SCALE_FT, GS_SCALE_KT
from .forecast import clearance_text, posterior_modes, run_forecast
from .geo import clean_track, features_near, features_payload, onset_latlon, rel_to_latlon
from .observe import has_clause, observe


def _r(a, nd):
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def build_replay(ds, wj, idx, positions, store, stage1, vocab, max_len, flow, device, lam,
                 obs, meta_row, *, k=8, exec_k=3, n_steps=10, guidance=1.0,
                 conditions=("belief", "point"), stride=2, seed=0) -> dict:
    """`positions` are the indices into `idx` (and `ds`) of the leg's windows, in time order."""
    first = wj.iloc[idx[positions[0]]]
    day, icao = first["day"], str(first["icao24"])
    track = clean_track(store.aircraft(day, icao))
    onsets = np.array([float(wj.iloc[idx[p]]["onset"]) for p in positions])
    t0 = float(onsets.min()) - 120.0
    t1 = float(onsets.max()) + 120.0
    seg_track = track[(track["t"] >= t0) & (track["t"] <= t1)].reset_index(drop=True)
    tstart = float(seg_track["t"].iloc[0])

    batch = {kk: v.to(device) for kk, v in collate([ds[p] for p in positions]).items()}
    rng = np.random.default_rng(seed)
    observed, toks, conf = [], [], []
    for p in positions:
        row = wj.iloc[idx[p]]
        seg = {c: row[c] for c in ("lateral", "vertical", "speed", "hdg_exit", "alt_exit_ft",
                                   "spd_exit_kt")}
        o = observe(seg, obs, leg_type=row.get("leg_type"), phase=row.get("phase"), rng=rng)
        o["has_clause"] = has_clause(seg)
        observed.append(o)
        toks.append(vocab.encode(o["text"], max_len))
        conf.append(o["confidence"])
    toks = torch.tensor(toks, device=device)
    conf = torch.tensor(conf, device=device, dtype=torch.float32)

    out = run_forecast(stage1, flow, batch, toks, conf, k=k, exec_k=exec_k, n_steps=n_steps,
                       guidance=guidance, lam=lam, conditions=conditions, seed=seed,
                       metrics=False)
    H = batch["fut_ctrl"].shape[1]
    steps = np.r_[np.arange(0, H, stride), H - 1]
    steps = np.unique(steps)

    events = []
    for b, p in enumerate(positions):
        onset = onsets[b]
        lat0, lon0 = onset_latlon(track, onset)
        fc = {}
        for cond, d in out["conditions"].items():
            pos = d["pos"][b][:, steps]                         # [N, S, 3]
            la, lo = rel_to_latlon(pos[..., 0], pos[..., 1], lat0, lon0)
            alt = pos[..., 2] * ALT_SCALE_FT
            gs = d["ctrl"][b][:, steps, 0] * GS_SCALE_KT
            fc[cond] = {"paths": np.stack([np.round(lo, 5), np.round(la, 5), np.round(alt)], -1)
                        .tolist(), "gs": np.round(gs).tolist(), "src": d["src"].tolist()}
        samples = out["samples"]
        events.append({
            "t": round(onset - tstart, 2), "onset_epoch": onset,
            "text": observed[b]["text"], "core": observed[b]["core"],
            "conf": round(observed[b]["confidence"], 3), "corrupted": observed[b]["corrupted"],
            "issued": " · ".join(observed[b]["issued"]) or "no clause",
            "map": clearance_text(out["greedy"], None, b),
            "modes": [[t, round(s, 3)] for t, s in posterior_modes(samples, b)[:5]],
            "sample_text": [clearance_text(samples, j, b) for j in range(k)],
            "lat": lat0, "lon": lon0, "forecast": fc,
            "t_steps": steps.tolist(),
        })

    feats = features_near(seg_track["lat"].min(), seg_track["lat"].max(),
                          seg_track["lon"].min(), seg_track["lon"].max(), margin_deg=0.15)
    return {
        "meta": {"callsign": meta_row.get("callsign", icao), "icao24": icao, "day": day,
                 "leg_id": first["leg_id"], "leg_type": meta_row.get("leg_type", ""),
                 "wake": first.get("wake", "unknown"),
                 "engine": first.get("engine_class", "unknown"),
                 "t0_epoch": tstart, "obs_mode": obs.mode},
        "track": {"t": _r(seg_track["t"] - tstart, 2), "lat": _r(seg_track["lat"], 5),
                  "lon": _r(seg_track["lon"], 5), "alt": _r(seg_track["alt_ft"], 0),
                  "gs": _r(seg_track["gs_kt"], 0), "trk": _r(seg_track["trk_deg"], 1),
                  "vr": _r(seg_track["vr_fpm"], 0)},
        "events": events, "conditions": list(conditions), "H": H, "k": k, "exec_k": exec_k,
        "features": features_payload(feats),
    }
