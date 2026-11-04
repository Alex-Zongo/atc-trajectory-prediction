"""Real-data path (inference/, evaluation/real_v2.py) on a synthetic track and tiny models:
the inputs match the training pipeline, the readers convert units, the CLI runs."""

import gzip
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from atc_forecast.data.resample import WindowSpec, extract_windows, history_windows, prepare_track
from atc_forecast.inference import Forecaster, load_states
from atc_forecast.inference.adsb import callsign_at, read_readsb
from atc_forecast.inference.transcript import Transmission, words_to_tokens
from atc_forecast.models.ctp import CTP, CTPConfig, save_ctp
from atc_forecast.models.sbm import SBM, SBMConfig

T0 = 1_767_225_600.0


def track(icao="abc123", callsign="UAL2", n=240, lat0=37.6, lon0=-122.3, seed=0):
    """OpenSky-style state vectors at 1 Hz: straight, then a right turn and a climb."""
    rng = np.random.default_rng(seed)
    t = T0 + np.arange(n) + rng.uniform(-0.1, 0.1, n)
    trk = 90.0 + np.clip(np.arange(n) - 120, 0, 30) * 3.0
    vz = np.where(np.arange(n) > 125, 8.0, 0.0)
    gs = np.full(n, 120.0)
    e = np.cumsum(gs * np.sin(np.radians(trk)))
    nn = np.cumsum(gs * np.cos(np.radians(trk)))
    z = 3000.0 + np.cumsum(vz)
    return pd.DataFrame({
        "icao24": icao, "callsign": callsign, "time_position": t,
        "latitude": lat0 + nn / 111320.0 + rng.normal(0, 2e-5, n),
        "longitude": lon0 + e / (111320.0 * np.cos(np.radians(lat0))) + rng.normal(0, 2e-5, n),
        "baro_altitude": z + rng.normal(0, 3, n), "geo_altitude": z + 60,
        "velocity": gs, "true_track": trk % 360, "vertical_rate": vz, "on_ground": False})


def test_words_carry_their_confidence():
    toks, conf = words_to_tokens([{"word": "United", "probability": 0.98},
                                  {"word": "descend", "prob": 0.8},
                                  {"word": "5,000.", "probability": 0.7},
                                  ("FL350", 0.6), ("nine", 0.5)])
    assert toks == ["united", "descend", "five", "thousand", "flight", "level", "three",
                    "five", "zero", "niner"]
    assert conf == [0.98, 0.8, 0.7, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5]
    toks, conf = words_to_tokens("turn left heading 270", default_conf=0.9)
    assert toks[-3:] == ["two", "seven", "zero"] and set(conf) == {0.9}


def test_history_windows_match_full_windows():
    tr = prepare_track(track())
    t0s = T0 + np.array([60.0, 110.0, 150.0])
    full, _ = extract_windows(tr, t0s)
    live, _ = history_windows(tr, t0s)
    assert np.array_equal(full["index"], live["index"])
    for k in ("hist_pos", "hist_vel", "hist_acc", "z0", "lat0", "lon0"):
        assert np.allclose(full[k], live[k]), k
    assert np.isnan(live["fut_pos"]).all()


def test_readsb_trace_units(tmp_path):
    d = {"icao": "ABC123", "timestamp": T0, "trace": [
        [0.0, 37.6, -122.3, 10000, 250, 90.0, 0, -1000, {"flight": "UAL2  "}, "adsb_icao", 10200],
        [1.0, 37.6, -122.299, 9990, 250, 90.0, 1, -1000, None, "adsb_icao", 10190],   # stale
        [2.0, 37.6, -122.298, "ground", 10, 90.0, 0, 0, None, "adsb_icao", None]]}
    p = tmp_path / "trace_full_abc123.json"
    p.write_bytes(gzip.compress(json.dumps(d).encode()))
    df = read_readsb(p)
    assert len(df) == 2 and df["icao24"].iloc[0] == "abc123"
    assert np.isclose(df["baro_altitude"].iloc[0], 3048.0)
    assert np.isclose(df["velocity"].iloc[0], 250 * 0.514444)
    assert np.isclose(df["vertical_rate"].iloc[0], -1000 * 0.3048 / 60)
    assert df["on_ground"].iloc[1] and df["callsign"].iloc[1] == "UAL2"
    assert load_states(str(p)).shape[0] == 2


def test_callsign_is_the_one_broadcast_at_t0():
    s = track(callsign="UAL2")
    s.loc[s["time_position"] > T0 + 100, "callsign"] = "UAL7"
    assert callsign_at(s, "abc123", T0 + 90) == "UAL2"
    assert callsign_at(s, "abc123", T0 + 150) == "UAL7"


def tiny_models(tmp_path):
    vocab = ["<pad>", "<unk>", "united", "two", "turn", "right", "heading", "one", "eight",
             "zero", "climb", "and", "maintain", "five", "thousand"]
    torch.manual_seed(0)
    sbm = SBM(SBMConfig(vocab_size=len(vocab), d=32, text_layers=1, hist_layers=1,
                        slot_blocks=1))
    torch.save({"cfg": vars(sbm.cfg), "state": sbm.state_dict(), "vocab": vocab},
               tmp_path / "sbm.pt")
    rng = np.random.default_rng(0)
    comps = [np.linalg.qr(rng.normal(size=(60, n)))[0].T.astype(np.float32) for n in (6, 4, 8)]
    basis = {"scale": np.full((60, 3), 2.0, np.float32), "mean": np.zeros((60, 3), np.float32),
             "lateral_along": comps[0], "lateral_right": comps[1], "vertical": comps[2],
             "coef_std": np.ones(18, np.float32)}
    for name, cond in (("ctp", "atoms"), ("hist", "none")):
        save_ctp(CTP(CTPConfig(d=32, dec_layers=1, hist_layers=1, condition=cond), basis),
                 tmp_path / f"{name}.pt")
    return tmp_path / "sbm.pt", tmp_path / "ctp.pt", tmp_path / "hist.pt"


def test_forecaster_end_to_end(tmp_path):
    sbm, ctp, _ = tiny_models(tmp_path)
    fc = Forecaster.load(sbm, ctp, device="cpu", aircraft_db=None)
    words = [{"word": w, "probability": 0.9} for w in
             "united two turn right heading one eight zero".split()]
    out = fc.forecast(track(), "abc123", T0 + 110, words, K=3, S=2)
    assert out["accepted"] and out["n_unknown"] == 0 and out["callsign"] == "UAL2"
    assert out["lat"].shape == (6, 60) and np.isfinite(out["alt_ft"]).all()
    assert abs(out["lat"][:, 0].mean() - 37.6) < 0.1           # starts at the aircraft
    assert len(out["readings"]) == 5 and abs(sum(p for _, p, _ in out["readings"])) <= 1 + 1e-6
    assert fc.forecast(track(), "abc123", T0 + 5, words)["reason"].startswith("history")


def test_real_eval_cli(tmp_path):
    sbm, ctp, hist = tiny_models(tmp_path)
    states = pd.concat([track(), track("def456", "DAL9", lat0=37.65, seed=1)])
    states.to_parquet(tmp_path / "adsb.parquet")
    txs = [{"id": "a", "t_end": T0 + 100, "icao24": "abc123", "group": "turn",
            "words": [{"word": w, "probability": 0.9} for w in "united two turn right".split()]},
           {"id": "b", "t_end": T0 + 110, "callsign": "DAL9", "text": "climb and maintain five thousand"},
           {"id": "c", "t_end": T0 + 110, "callsign": "XYZ1", "text": "unknown"},
           {"id": "d", "t_end": T0 + 230, "icao24": "abc123", "text": "turn right"}]
    (tmp_path / "tx.jsonl").write_text("\n".join(json.dumps(t) for t in txs))
    assert Transmission.from_dict(txs[0]).t0 == T0 + 100
    out = tmp_path / "real.json"
    r = subprocess.run([sys.executable, "-m", "atc_forecast.evaluation.real_v2",
                        "--transmissions", str(tmp_path / "tx.jsonl"),
                        "--adsb", str(tmp_path / "adsb.parquet"), "--sbm", str(sbm),
                        "--ctp", str(ctp), "--hist-only", str(hist), "--k", "3", "--s", "2",
                        "--scene-radius-nm", "20", "--aircraft-db", "none", "--device", "cpu",
                        "--out", str(out), "--per-message", str(tmp_path / "pm.jsonl")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(out.read_text())
    assert res["n_evaluated"] == 2
    assert res["rejected"] == {"no_addressee": 1, "future_short": 1}
    assert set(res["by_condition"]) == {"composed", "known", "mode", "hist_only", "scene"}
    assert res["addressee"]["aircraft_on_frequency_mean"] == 2.0
    assert len((tmp_path / "pm.jsonl").read_text().splitlines()) == 2


def test_callsign_matching_respects_token_boundaries():
    from atc_forecast.inference.callsign import find_callsigns
    c = ["UAL2", "UAL2437", "SKW514", "UAL2480", "DAL785"]
    assert find_callsigns("17F1 80 knots, United 20367", c) == []          # not UAL2
    m = find_callsigns("climb and maintain 5000 United 24-37 heavy", c)
    assert [(x.callsign, x.role) for x in m] == [("UAL2437", "pilot")]
    assert find_callsigns("Skywest 514.90, radar contact", c)[0].callsign == "SKW514"
    assert find_callsigns("united twenty four thirty seven turn left", c)[0].role == "controller"
    assert find_callsigns("NorCal, Delta seven eight five, with you", c)[0].role == "pilot"


def test_group_form_numbers_become_digits():
    toks, conf = words_to_tokens([("nineteen", 0.8), ("forty-nine", 0.6), ("twenty", 0.9)])
    assert toks == ["one", "niner", "four", "niner", "two", "zero"]
    assert conf == [0.8, 0.8, 0.6, 0.6, 0.9, 0.9]
