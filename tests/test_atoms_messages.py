"""Atom vocabulary and precomputed messages."""

import numpy as np
import pandas as pd

from atc_forecast.data import atoms as A
from atc_forecast.data.messages import MessageConfig, confusable, generate, word_errors


def test_atom_round_trip_and_sizes():
    assert A.SIZES == {"status": 3, "lateral": 146, "vertical": 106, "speed": 112}
    lat = A.lateral_atom(["L", "R", "S", "R"], [271.0, 2.4, 90.0, 357.6])
    assert [A.describe("lateral", a) for a in lat] == ["L270", "R360", "-", "R360"]
    ver = A.vertical_atom(["descend", "climb", "level"], [11_620, 35_200, 5_000])
    assert [A.describe("vertical", a) for a in ver] == ["descend 12,000", "climb FL350", "-"]
    spd = A.speed_atom(["decel", "accel", "const"], [254, 18, 300])
    assert [A.describe("speed", a) for a in spd] == ["reduce 250", "increase 60", "-"]


def test_relative_lateral_turns_follow_the_commanded_direction():
    l270, r270 = A.lateral_atom(["L", "R"], [270, 270])
    f = A.relative_features("lateral", [l270, r270], trk_deg=[90, 90], alt_ft=0, gs_kt=0)
    assert np.isclose(f[0, 2] * 180, -180) and np.isclose(f[1, 2] * 180, 180)
    l100 = A.lateral_atom(["L"], [100])[0]
    f = A.relative_features("lateral", [l100], trk_deg=[90], alt_ft=0, gs_kt=0)
    assert np.isclose(f[0, 2] * 180, -350)                 # left to 100 from 090 is 350 left


def test_vertical_consistency_bit():
    climb5, desc5 = A.vertical_atom(["climb", "descend"], [5000, 5000])
    f = A.relative_features("vertical", [climb5, desc5, A.MAINTAIN, A.NONE],
                            trk_deg=0, alt_ft=[8000, 8000, 8000, 8000], gs_kt=0)
    assert f[0, 2] == 0 and f[1, 2] == 1                   # climb to below: inconsistent
    assert np.isclose(f[1, 1], -0.3)
    assert np.allclose(f[2], [0.8, 0.0, 1.0]) and np.allclose(f[3], 0)


def test_word_errors_ignore_rephrasing_and_flag_wrong_digits():
    said = "united four two turn right heading one three five"
    assert not word_errors(said, "united four two fly heading one three five").any()
    e = word_errors(said, "united four two turn right heading one three niner")
    assert e.sum() == 1 and e[-1]


def test_confusable_callsign_differs_by_one_slip():
    rng = np.random.default_rng(0)
    cs = ["united", "four", "two", "one"]
    for _ in range(20):
        c = confusable(cs, rng)
        assert c != cs and sum(a != b for a, b in zip(c, cs)) == 1


def _windows():
    base = {"callsign": "UAL421", "leg_type": "arrival", "phase": "descent",
            "hdg_exit": 135.0, "alt_exit_ft": 6000.0, "spd_exit_kt": 250.0,
            "trk_deg": 90.0, "alt_ft": 9000.0, "gs_kt": 280.0}
    rows = [dict(base, window_id=0, kind="maneuver", lateral="R", vertical="descend",
                 speed="const", vr_fpm=-1500.0),
            dict(base, window_id=1, kind="steady", lateral="S", vertical="level",
                 speed="const", vr_fpm=0.0)]
    df = pd.DataFrame(rows)
    df["atom_lateral"] = A.lateral_atom(df["lateral"], df["hdg_exit"])
    df["atom_vertical"] = np.where(df["kind"] == "maneuver",
                                   A.vertical_atom(df["vertical"], df["alt_exit_ft"]), 0)
    df["atom_speed"] = 0
    return df


def test_message_variants_and_labels():
    w = _windows()
    m = generate(w, w[w["kind"] == "maneuver"], MessageConfig(draws_own=3), 0)
    own = m[m["variant"] == "own"]
    assert len(own) == 3 and (own["window_id"] == 0).all() and (own["status"] == 0).all()
    # labels are what was SAID: the issued instruction, whatever the channel did
    assert set(own["atom_vertical"]) <= {int(w.loc[0, "atom_vertical"])}
    assert set(m[m["variant"] == "maintain"]["window_id"]) == {1}
    for v, status in (("own_other", 1), ("other", 2)):
        x = m[m["variant"] == v]
        assert (x["status"] == status).all()
        assert (x[["atom_lateral", "atom_vertical", "atom_speed"]] == A.NONE).all().all()
    # other-aircraft messages are never addressed to the own callsign
    assert not m[m["variant"] == "other"]["said"].str.startswith("united four two one").any()
    assert all(len(k) == len(t.split()) for k, t in zip(m["kappa"], m["text"]))


def test_messages_are_deterministic_in_the_seed():
    w = _windows()
    a = generate(w, w[w["kind"] == "maneuver"], MessageConfig(), 7)
    b = generate(w, w[w["kind"] == "maneuver"], MessageConfig(), 7)
    assert a["text"].tolist() == b["text"].tolist()
    assert all(np.array_equal(x, y) for x, y in zip(a["kappa"], b["kappa"]))


def test_other_messages_come_from_concurrent_traffic():
    """'Other' instructions are ones given to another aircraft flying at the same time."""
    w = _windows()
    near = dict(w.iloc[0], window_id=2, callsign="SKW55", hdg_exit=200.0)
    far = dict(w.iloc[0], window_id=3, callsign="DAL9", hdg_exit=310.0)
    w = pd.concat([w, pd.DataFrame([near, far])], ignore_index=True)
    w["t0"] = [1000.0, 1010.0, 1050.0, 9000.0]
    w["icao24"] = ["a", "b", "c", "d"]
    m = generate(w.iloc[:2], w[w["kind"] == "maneuver"],
                 MessageConfig(draws_aux=20, distractor_dt=120.0), 0)
    other = m[m["variant"] == "other"]
    # window 0 hears only the near aircraft (heading 200), never the one 2 h later (310)
    assert other[other["window_id"] == 0]["said"].str.contains("two zero zero").all()
    assert not other["said"].str.contains("three one zero").any()
