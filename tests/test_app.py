"""The viewer's backend: traces must match the models they depict, and the dataset
walkthrough must reproduce the pipeline it narrates. Small random models, no data."""

import numpy as np
import pytest
import torch

from atc_forecast.app.geo import rel_to_latlon
from atc_forecast.app.introspect import trace_stage1, trace_stage2
from atc_forecast.app.observe import Observation, observe
from atc_forecast.app.story import build_story
from atc_forecast.data.legs import assign_legs
from atc_forecast.data.segmentation import segment_trajectories
from atc_forecast.data.text import Vocab
from atc_forecast.models.backbone import EncoderConfig
from atc_forecast.models.stage1 import ClearancePosterior
from atc_forecast.models.stage2 import FlowPredictor
from atc_forecast.units import DEG_TO_M, POS_SCALE_M

SEG = {"lateral": "R", "vertical": "descend", "speed": "const",
       "hdg_exit": 183.0, "alt_exit_ft": 4351.0, "spd_exit_kt": 210.0}


@pytest.fixture(scope="module")
def stage1():
    torch.manual_seed(0)
    vocab = Vocab(["turn right heading one eight five descend and maintain four thousand",
                   "southwest one two three climb reduce speed knots"])
    cfg = EncoderConfig(vocab_size=len(vocab), max_len=24, hidden=32, n_heads=2, n_layers=2)
    return ClearancePosterior(cfg, state_dim=5, n_slots=3).eval(), vocab


def test_stage1_trace_reproduces_the_model(stage1):
    model, vocab = stage1
    state = torch.randn(1, 5)
    tr = trace_stage1(model, vocab, 24, "turn right heading one eight five", 0.8, state,
                      k=5, seed=3)
    # the trace raised if its replay diverged; check it also covers every field and draw
    assert [f["name"] for f in tr["fields"]][0] == "is_clearance"
    assert len(tr["samples"]) == 10 and len(tr["samples"][0]["value"]) == 5
    assert tr["tokens"][0] == "[conf]" and len(tr["layers"]) == 2
    hd = tr["greedy"][3]
    assert hd["name"] == "hdg" and abs(sum(hd["dist"]["probs"][0]) - 1) < 1e-3
    assert "ungrounded" in hd                 # the grounding counterfactual is recorded


@pytest.mark.parametrize("guidance", [1.0, 0.0, 2.0])
def test_stage2_trace_reproduces_the_sampler(guidance):
    torch.manual_seed(0)
    flow = FlowPredictor(dim=32, heads=2, layers=2, past_layers=1, h_steps=12, p_steps=8,
                         time_encoding=True).eval()
    with torch.no_grad():                     # open the adaLN-zero gates so blocks matter
        for blk in flow.blocks:
            blk.adaln.weight.normal_(0, 0.05)
        flow.head.weight.normal_(0, 0.05)
    batch = {"past": torch.randn(1, 8, 7) * 0.1, "past_mask": torch.ones(1, 8),
             "past_time": torch.linspace(8, 1, 8)[None], "directive": torch.rand(1, 10),
             "wake": torch.tensor([2]), "engine": torch.tensor([2]),
             "n_engines": torch.tensor([2.0]), "fut_ctrl": torch.zeros(1, 12, 4),
             "fut_pos": torch.zeros(1, 12, 3), "fut_mask": torch.ones(1, 12)}
    tr = trace_stage2(flow, batch, n_steps=4, k=3, guidance=guidance, seed=1)
    assert len(tr["flow"]["ctrl"]) == 5        # source plus one state per Euler step
    assert tr["memory"]["labels"][-3:] == ["LATERAL", "VERTICAL", "SPEED"]


def test_observation_modes():
    clean = observe(SEG, Observation(mode="clean", context=False))
    assert clean["text"] == "turn right heading one eight five, descend and maintain four thousand."
    assert clean["confidence"] == pytest.approx(0.82) and clean["corrupted"] == ""
    noisy = observe(SEG, Observation(mode="channel", seed=4))
    assert noisy["issued"] == clean["issued"] == ["turn right heading 185",
                                                  "descend and maintain 4,000"]
    custom = observe(SEG, Observation(mode="custom", custom_text=" climb ", confidence=0.3))
    assert custom["text"] == "climb" and custom["confidence"] == 0.3


def test_rel_to_latlon_inverts_the_window_projection():
    lat0, lon0 = 37.6, -122.4
    lat, lon = 37.65, -122.31
    dx = (lon - lon0) * DEG_TO_M * np.cos(np.radians(lat0)) / POS_SCALE_M
    dy = (lat - lat0) * DEG_TO_M / POS_SCALE_M
    la, lo = rel_to_latlon(dx, dy, lat0, lon0)
    assert la == pytest.approx(lat) and lo == pytest.approx(lon)


def test_story_reproduces_the_pipeline(adsb):
    raw = adsb[adsb["icao24"] == "abc000"]
    segs = assign_legs(segment_trajectories(raw))
    man = segs[segs["is_maneuver"]].iloc[0]
    window = {"past": np.zeros((30, 7), np.float32), "past_time": np.zeros(30, np.float32),
              "past_mask": np.zeros(30, np.float32), "fut_pos": np.zeros((60, 3), np.float32),
              "fut_mask": np.ones(60, np.float32), "directive": np.zeros(10, np.float32)}
    window["past_mask"][-5:] = 1
    s = build_story(raw, "2026-01-15", "2026-01-15_" + man["leg_id"], float(man["t_start"]),
                    window, Observation(mode="clean"))
    focus = [x for x in s["segments"] if x["is_focus"]]
    assert len(focus) == 1 and focus[0]["is_maneuver"]
    assert s["counts"]["segments"] == len(segs)
    assert s["draws"] and s["draws"][0]["text"]
    # merging only ever reduces the number of label changes
    for sig in s["signals"]:
        flips = lambda a: sum(x != y for x, y in zip(a[1:], a[:-1]))
        assert flips(sig["state"]) <= flips(sig["raw_state"])


def test_custom_text_speaks_numbers_like_training():
    from atc_forecast.app.observe import speak_numbers
    assert speak_numbers("turn left heading 270") == "turn left heading two seven zero"
    assert speak_numbers("descend to 5,000") == "descend to five thousand"
    assert speak_numbers("climb FL350") == "climb flight level three five zero"
    assert speak_numbers("contact 135.65") == "contact one three five point six five"
    o = observe(SEG, Observation(mode="custom", custom_text="heading 090", speak_numbers=False))
    assert o["text"] == "heading 090"
