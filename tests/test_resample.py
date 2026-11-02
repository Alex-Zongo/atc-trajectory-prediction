"""Resampling and feature construction on synthetic tracks with known truth.

Each test checks one promise the v2 data makes: 1 Hz on both sides, no information from
one side of t0 in the other, refusal of gaps instead of patching them, targets that
integrate exactly to positions, and a smoother that follows a real turn.
"""

import numpy as np
import pandas as pd
import pytest

from atc_forecast.data.frames import (
    cumulative_turn,
    make_features,
    pca_basis,
    positions,
    residuals,
    step_scale,
)
from atc_forecast.data.resample import WindowSpec, extract_windows, prepare_track, windows_from_adsb
from atc_forecast.units import DEG_TO_M, KT_TO_MS

LAT0, LON0 = 37.6, -122.0


def synth(duration=200.0, gs_kt=250.0, trk0=90.0, turn_rate=0.0, turn_start=1e9,
          vz=0.0, z0=3000.0, dt_mean=1.0, pos_noise=3.0, seed=0, gaps=(), on_ground=None):
    """A raw OpenSky-like frame for one aircraft, plus the true track for checking.

    Constant speed, a constant-rate turn from turn_start, constant vertical speed. Sample
    times jitter around dt_mean, and gaps=[(a, b)] removes all samples inside (a, b).
    """
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.uniform(0.6, 1.4, int(duration / dt_mean * 1.2)) * dt_mean)
    t = t[t < duration] + 1_700_000_000.0
    tr = t - t[0]
    for a, b in gaps:
        keep = (tr <= a) | (tr >= b)
        t, tr = t[keep], tr[keep]
    gs = gs_kt * KT_TO_MS
    # heading: constant, then a constant-rate turn
    dur_turn = np.clip(tr - turn_start, 0, None)
    trk = np.radians(trk0 + turn_rate * dur_turn)
    # integrate position finely so the truth is exact
    tf = np.arange(0, duration + 1, 0.01)
    trkf = np.radians(trk0 + turn_rate * np.clip(tf - turn_start, 0, None))
    e = np.concatenate([[0], np.cumsum(gs * np.sin(trkf[:-1]) * 0.01)])
    n = np.concatenate([[0], np.cumsum(gs * np.cos(trkf[:-1]) * 0.01)])
    E, N = np.interp(tr, tf, e), np.interp(tr, tf, n)
    z = z0 + vz * tr
    lat = LAT0 + (N + rng.normal(0, pos_noise, len(tr))) / DEG_TO_M
    lon = LON0 + (E + rng.normal(0, pos_noise, len(tr))) / (DEG_TO_M * np.cos(np.radians(lat)))
    df = pd.DataFrame({
        "icao24": "abc123", "time_position": t, "latitude": lat, "longitude": lon,
        "baro_altitude": np.round(z / 7.62) * 7.62, "geo_altitude": z + 100,
        "velocity": gs + rng.normal(0, 0.3, len(t)),
        "true_track": np.degrees(trk) % 360 + rng.normal(0, 0.2, len(t)),
        "vertical_rate": vz + rng.normal(0, 0.3, len(t)),
        "on_ground": False if on_ground is None else on_ground(tr),
    })
    truth = {"t": tf + t[0], "e": e, "n": n, "trk": trkf}
    return df, truth


def one(df, t0, spec=WindowSpec()):
    tr = prepare_track(df)
    w, reasons = extract_windows(tr, [t0], spec)
    return w, reasons[0]


def test_grid_is_one_hertz_on_both_sides():
    df, _ = synth()
    t0 = df["time_position"].iloc[0] + 80.3          # deliberately off any sample time
    w, r = one(df, t0)
    assert r is None
    assert w["hist_pos"].shape == (1, 30, 3) and w["fut_pos"].shape == (1, 60, 3)
    f = make_features(w)
    assert f["hist"].shape == (1, 30, 9) and f["fut_vel"].shape == (1, 60, 3)
    assert np.isfinite(f["hist"]).all() and np.isfinite(f["fut_vel"]).all()
    # the present is the origin
    assert np.allclose(w["hist_pos"][0, -1], 0.0)


def test_history_never_sees_the_future():
    df, _ = synth()
    t0 = df["time_position"].iloc[0] + 80.3
    w1, _ = one(df, t0)
    df2 = df.copy()
    fut = df2["time_position"] > t0
    df2.loc[fut, "latitude"] += 0.01                 # move every future sample by about 1 km
    df2.loc[fut, "baro_altitude"] += 500
    df2.loc[fut, "velocity"] *= 1.3
    w2, _ = one(df2, t0)
    for k in ("hist_pos", "hist_vel", "hist_acc", "z0"):
        assert np.array_equal(w1[k], w2[k]), k
    f1, f2 = make_features(w1), make_features(w2)
    assert np.array_equal(f1["hist"], f2["hist"]) and np.array_equal(f1["ctx"], f2["ctx"])
    assert not np.allclose(f1["fut_vel"], f2["fut_vel"])


def test_history_is_causal_within_itself():
    """The history value at t0-k uses samples up to t0-k only."""
    df, _ = synth()
    t0 = df["time_position"].iloc[0] + 80.3
    w1, _ = one(df, t0)
    df2 = df.copy()
    late = (df2["time_position"] > t0 - 10) & (df2["time_position"] <= t0)
    df2.loc[late, "latitude"] += 0.001
    w2, _ = one(df2, t0)
    # Grid points t0-29 ... t0-10 see nothing that changed. Positions are relative to p0,
    # which did change, and the local frame is anchored at the (moved) last sample, so
    # compare shapes and velocities up to floating-point rounding of the frame shift.
    abs1 = w1["hist_pos"][0] - w1["hist_pos"][0, 0]
    abs2 = w2["hist_pos"][0] - w2["hist_pos"][0, 0]
    assert np.allclose(abs1[:20], abs2[:20], atol=1e-6)
    assert np.allclose(w1["hist_vel"][0, :20], w2["hist_vel"][0, :20], atol=1e-9)
    assert not np.allclose(abs1[-1], abs2[-1], atol=1.0)  # the tail did change


def test_future_uses_only_future_samples():
    """Changing history samples leaves the future targets untouched."""
    df, _ = synth()
    t0 = df["time_position"].iloc[0] + 80.3
    w1, _ = one(df, t0)
    df2 = df.copy()
    past = df2["time_position"] < t0 - 2
    df2.loc[past, "velocity"] *= 0.9
    df2.loc[past, "baro_altitude"] += 300
    w2, _ = one(df2, t0)
    assert np.allclose(w1["fut_pos"], w2["fut_pos"], atol=1e-9)
    assert np.allclose(w1["fut_vel"], w2["fut_vel"], atol=1e-9)


def test_present_offset_is_small_and_kept_out_of_targets():
    df, _ = synth(pos_noise=4.0)
    t0 = df["time_position"].iloc[0] + 80.3
    w, _ = one(df, t0)
    assert np.hypot(*w["offset0"][0, :2]) < 15.0
    # the first increment is a normal one-second step, not a jump to the other estimate
    f = make_features(w)
    gs = np.hypot(*f["fut_vel"][0, :3, :2].T)
    assert np.abs(gs - 250 * KT_TO_MS).max() < 2.0


@pytest.mark.parametrize("gap, reason", [((70.0, 76.0), "history_gap"),
                                         ((95.0, 101.0), "future_gap")])
def test_gaps_are_refused_not_patched(gap, reason):
    df, _ = synth(gaps=[gap])
    t0 = df["time_position"].iloc[0] + 80.3
    w, r = one(df, t0)
    assert w is None and r == reason


def test_gap_under_threshold_is_bridged_smoothly():
    df, _ = synth(gaps=[(100.0, 102.3)], turn_rate=3.0, turn_start=90.0, seed=3)
    t0 = df["time_position"].iloc[0] + 80.3
    w, r = one(df, t0)
    assert r is None
    # A 3 deg/s turn at 250 kt has a centripetal acceleration of v * omega = 6.7 m/s^2.
    # Across the gap (future steps 20-22) the bridged velocity must stay on that, no spike.
    true_acc = 250 * KT_TO_MS * np.radians(3.0)
    acc = np.hypot(*np.diff(w["fut_vel"][0, :, :2], axis=0).T)
    assert acc.max() < 1.25 * true_acc
    assert np.abs(acc[25:] - true_acc).max() < 1.5


def test_ground_and_slow_windows_are_dropped():
    df, _ = synth(gs_kt=30.0)
    t0 = df["time_position"].iloc[0] + 80.3
    assert one(df, t0)[1] == "slow"
    df, _ = synth(on_ground=lambda tr: (tr > 100) & (tr < 140))
    assert one(df, df["time_position"].iloc[0] + 80.3)[1] == "future_gap"


def test_stale_repeats_are_dropped():
    df, _ = synth()
    dup = df.iloc[::4].copy()
    dup["latitude"] += 0.0                               # same time_position, repeated row
    tr = prepare_track(pd.concat([df, dup]))
    assert len(tr["t"]) == len(df)


def test_targets_integrate_exactly_to_positions():
    df, _ = synth(turn_rate=2.0, turn_start=85.0, vz=-5.0)
    t0 = df["time_position"].iloc[0] + 80.3
    w, _ = one(df, t0)
    f = make_features(w)
    from atc_forecast.data.frames import to_track
    psi = np.radians(f["state"]["trk_deg"])
    assert np.allclose(positions(f["fut_vel"]), to_track(w["fut_pos"], psi), atol=1e-3)
    R = residuals(f["fut_vel"], f["center"])
    assert np.allclose(R + f["center"][:, None], f["fut_vel"])


def test_smoothed_future_follows_a_turn():
    rate = 3.0                                            # standard-rate turn, deg/s
    df, _ = synth(turn_rate=rate, turn_start=90.0, pos_noise=4.0, seed=1)
    t0 = df["time_position"].iloc[0] + 80.0
    w, _ = one(df, t0)
    f = make_features(w)
    turn = cumulative_turn(f["fut_vel"])[0]
    # truth: straight until t0 + 10, then 3 deg/s to the right
    k = np.arange(1, 61) - 0.5                            # increments are mid-step averages
    expect = rate * np.clip(k - 10.0, 0, None)
    err = np.abs(turn - expect)
    assert err[20:].max() < 4.0                           # within 4 deg once established
    assert abs(turn[-1] - expect[-1]) < 3.0


def test_smoothed_speed_beats_raw_differences():
    df, _ = synth(pos_noise=5.0, seed=2)
    t0 = df["time_position"].iloc[0] + 80.3
    w, _ = one(df, t0)
    f = make_features(w)
    gs = np.hypot(f["fut_vel"][0, :, 0], f["fut_vel"][0, :, 1])
    true = 250 * KT_TO_MS
    tr = prepare_track(df)
    raw = np.hypot(np.diff(tr["lat"]) * DEG_TO_M,
                   np.diff(tr["lon"]) * DEG_TO_M * np.cos(np.radians(LAT0))) / np.diff(tr["t"])
    assert np.std(gs - true) < 0.25 * np.std(raw - true)
    assert abs(np.mean(gs) - true) < 0.5


def test_external_interface_matches_training_path():
    df, _ = synth()
    t0s = df["time_position"].iloc[0] + np.array([60.3, 80.3, 300.0])  # last one: no future
    req = pd.DataFrame({"icao24": ["abc123"] * 3, "t0": t0s})
    w, rep = windows_from_adsb(df, req)
    assert rep["accepted"].tolist() == [True, True, False]
    w1, _ = one(df, t0s[1])
    assert np.array_equal(w["fut_vel"][1], w1["fut_vel"][0])


def test_step_scale_and_basis_shapes():
    rng = np.random.default_rng(0)
    R = rng.normal(size=(500, 60, 3)) * np.linspace(0.1, 3, 60)[None, :, None]
    s = step_scale(R)
    assert s.shape == (60, 3) and (s >= 0.05).all()
    _, comps, kept = pca_basis(R, s, k=(4, 4, 8))
    assert [c.shape for c in comps] == [(4, 60), (4, 60), (8, 60)]
    assert all(0 < x <= 1 for x in kept)
