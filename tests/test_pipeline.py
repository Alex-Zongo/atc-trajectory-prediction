"""End-to-end smoke tests: data build, both models, scoring, recalibration."""

import numpy as np
import pandas as pd
import torch

from atc_forecast.channel.model import NoiseConfig, corrupt_segment, load_channel
from atc_forecast.data.legs import assign_legs, build_leg_summary
from atc_forecast.data.phraseology import (add_instructions, round_heading, segment_directive,
                                           spoken_altitude, telephony)
from atc_forecast.data.segmentation import segment_trajectories
from atc_forecast.data.windows import add_next_onset, directive_vector
from atc_forecast.models.stage2 import deadreckon, horizon_weights, last_valid_index


def test_segmentation_finds_maneuvers(adsb):
    segs = segment_trajectories(adsb, airborne_only=True)
    assert len(segs) > 0
    assert segs["is_maneuver"].any()
    assert set(segs["lateral"]) <= {"L", "R", "S"}
    assert set(segs["vertical"]) <= {"climb", "descend", "level"}


def test_phraseology_matches_faa_conventions():
    assert round_heading(0) == 360               # heading 000 is spoken 360
    assert round_heading(357) == 355
    assert spoken_altitude(4000) == "four thousand"
    assert spoken_altitude(24000).startswith("flight level")
    assert telephony("SWA2407") == "Southwest two four zero seven"
    assert telephony("N123AB").startswith("November")
    seg = {"lateral": "R", "vertical": "descend", "speed": "const",
           "hdg_exit": 182.0, "alt_exit_ft": 4100.0, "spd_exit_kt": 210.0}
    d = segment_directive(seg, spoken=True)
    assert "turn right heading" in d and "descend and maintain" in d
    assert "speed" not in d                      # no speed clause was commanded


def test_directive_vector_layout():
    row = pd.Series({"lateral": "L", "vertical": "climb", "speed": "accel",
                     "hdg_exit": 90.0, "alt_exit_ft": 9000.0, "spd_exit_kt": 250.0})
    v = directive_vector(row.to_frame().T.itertuples().__next__())
    assert v.shape == (10,)
    assert v[0] == 1.0 and v[1] == 0.0           # has_turn, turn_right=False for a left turn
    assert v[4] == 1.0 and v[5] == 1.0           # has_vert, climb
    assert np.isclose(v[2], 1.0, atol=1e-6)      # sin(90 deg)
    assert v[7] == 1.0 and v[8] == 1.0           # has_speed, accel


def test_legs_and_instructions(adsb):
    segs = assign_legs(segment_trajectories(adsb))
    assert segs["leg_id"].nunique() >= 6
    assert (segs.groupby("leg_id")["seg_idx"].min() == 0).all()
    segs = add_instructions(segs)
    man = segs[segs["is_maneuver"]]
    assert (man["atc_directive"] != "").all()
    # steady segments carry no clearance, which is the point of keeping them
    assert (segs[~segs["is_maneuver"]]["atc_directive"] == "").all()
    summary = build_leg_summary(segs)
    assert set(summary["leg_type"]) <= {"ground", "arrival", "departure", "overflight"}


def test_next_onset_truncation_boundary(adsb):
    segs = add_next_onset(assign_legs(segment_trajectories(adsb)))
    g = segs[segs["leg_id"] == segs["leg_id"].iloc[0]].sort_values("seg_idx")
    assert pd.isna(g["next_onset"].iloc[-1])     # the last segment has no next clearance
    assert (g["next_onset"].iloc[:-1].values == g["t_start"].iloc[1:].values).all()


def test_channel_is_many_to_one_and_measures_against_issued():
    seg = {"lateral": "R", "vertical": "descend", "speed": "const",
           "hdg_exit": 183.0, "alt_exit_ft": 4351.0, "spd_exit_kt": 210.0}
    rng = np.random.default_rng(0)
    out = [corrupt_segment(seg, NoiseConfig(profile="jacktol", phrase_variant_p=0.5,
                                            keyword_dropout_p=0.3), rng) for _ in range(60)]
    # the issued values are the quantised ones; phraseology rounding must not look like error
    assert all(o["issued_hdg"] == 185.0 for o in out)
    assert all(o["issued_alt_ft"] == 4000.0 for o in out)
    assert len({o["noisy_directive"] for o in out}) > 1      # the channel is stochastic
    assert all(0.0 <= o["asr_confidence"] <= 1.0 for o in out)


def test_channel_profile_loads():
    ch = load_channel("jacktol")
    assert ch.spoken and ch.scaling in ("fixed", "by_degradation")
    assert 0.0 <= ch.rate("altitude", "dropout") <= 1.0


def test_last_valid_index_handles_internal_gaps():
    # a trailing-gap mask: sum - 1 would be right here
    m = torch.tensor([[1.0, 1, 1, 0, 0]])
    last, has = last_valid_index(m)
    assert int(last[0]) == 2 and bool(has[0])
    # an INTERNAL hole: sum - 1 would give 2, which is itself invalid
    m = torch.tensor([[1.0, 1, 0, 1, 0]])
    last, has = last_valid_index(m)
    assert int(last[0]) == 3
    assert int(m.long().sum(1) - 1) == 2         # the expression this function replaces
    # an all-invalid row must be flagged, not silently indexed
    last, has = last_valid_index(torch.zeros(1, 5))
    assert not bool(has[0])


def test_horizon_weights_favour_early_steps():
    w = horizon_weights(10, "cpu")
    assert np.isclose(float(w.mean()), 1.0)
    assert w[0] > w[-1]                          # early error displaces more positions
    assert torch.allclose(horizon_weights(10, "cpu", "uniform"), torch.ones(10))


def test_deadreckon_integrates_before_the_step():
    # due east at a constant speed: north and altitude must not move
    H = 10
    ctrl = torch.zeros(1, H, 4)
    ctrl[..., 0] = 200.0 / 600.0                 # gs normalised
    ctrl[..., 1] = 1.0                           # sin(heading) = due east
    pos = deadreckon(ctrl, torch.tensor([5000.0]))
    assert torch.allclose(pos[0, 0], torch.tensor([0.0, 0.0, 5000.0 / 45000.0]), atol=1e-6)
    assert (pos[0, 1:, 0] > 0).all()             # east increases
    assert torch.allclose(pos[0, :, 1], torch.zeros(H), atol=1e-7)
    assert torch.allclose(pos[0, :, 2], torch.full((H,), 5000.0 / 45000.0), atol=1e-7)
