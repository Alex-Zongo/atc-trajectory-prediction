"""Interactive viewer for the two-stage trajectory prediction.

    streamlit run src/atc_forecast/app/viewer.py -- \
        --stage1 runs/stage1.pt --stage2 runs/stage2.pt \
        --data data/windows_raw --meta data/dataset/meta_grounded.parquet \
        --lambda-json runs/lambda.json --adsb "data/adsb/adsb_*.parquet"

Four pages share one sidebar (the window, how its clearance is observed, and how the
models sample), so every page is always looking at the same case:

  Prediction           Stage 1's distribution over instructions and the trajectory
                       distribution under each condition, on the real airspace, with per-axis profiles and a 3-D view
  Flight replay        a whole held-out leg flown back with both models running at every
                       clearance
  Inside the models    both architectures, traced on this input, operation by operation
  How the data is built  the dataset pipeline replayed on this leg's raw ADS-B

`--adsb` points at the raw ADS-B day files the windows were cut from. Without it the
prediction page falls back to a plan view in nautical miles, and the replay and dataset
pages are unavailable, since both need absolute positions.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

from atc_forecast.app import components, plots
from atc_forecast.app.forecast import (CONDITIONS, LABELS, past_physical, posterior_modes,
                                       run_forecast, truth_physical, ctrl_physical, pos_physical)
from atc_forecast.app.geo import (AdsbStore, clean_track, features_near, onset_latlon)
from atc_forecast.app.introspect import trace_stage1, trace_stage2
from atc_forecast.app.observe import (MODES, TRAINED_CONFIDENCE, Observation, observe,
                                      unknown_words)
from atc_forecast.app.replay import build_replay
from atc_forecast.app.story import build_story
from atc_forecast.channel.model import list_channels
from atc_forecast.evaluation.compose import CLEAR_COLS, build_windows, onset_state
from atc_forecast.evaluation.stage2 import lam_per_step
from atc_forecast.models.stage1 import load_stage1
from atc_forecast.models.stage2 import WindowDataset, collate, load_stage2
from atc_forecast.training.common import pick_device
from atc_forecast.units import ALT_SCALE_FT, GS_SCALE_KT


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--lambda-json", default=None)
    ap.add_argument("--adsb", default="data/adsb/adsb_*.parquet",
                    help="glob of the raw ADS-B day files the windows were built from")
    return ap.parse_args(sys.argv[1:])


A = cli()
st.set_page_config(page_title="ATC trajectory prediction", layout="wide", page_icon="✈️")


# --------------------------------------------------------------------------------------
# resources
# --------------------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading models and windows…")
def resources(stage1_path, stage2_path, data_dir, meta_path, lambda_path, adsb):
    device = pick_device()
    stage1, vocab, max_len = load_stage1(stage1_path, device)
    flow = load_stage2(stage2_path, device)
    npz, wj, idx = build_windows(data_dir, meta_path)
    H = npz["fut_ctrl"].shape[1]
    lam = (lam_per_step(json.loads(Path(lambda_path).read_text())["fit"], H, device)
           if lambda_path else torch.ones(H, 3, device=device))
    meta = pd.read_parquet(meta_path, columns=["leg_id", "seg_idx", "callsign", "leg_type",
                                               "ac_class", "is_vfr"])
    sub = wj.iloc[idx].reset_index(drop=True)
    sub = sub.merge(meta, on=["leg_id", "seg_idx"], how="left", suffixes=("", "_m"))
    return {"device": device, "stage1": stage1, "vocab": vocab, "max_len": max_len,
            "flow": flow, "npz": npz, "wj": wj, "idx": idx, "lam": lam, "held": sub,
            "ds": WindowDataset(npz, wj, idx), "store": AdsbStore(adsb)}


R = resources(A.stage1, A.stage2, A.data, A.meta, A.lambda_json, A.adsb)
HELD = R["held"]


@dataclass(frozen=True)
class Sampling:
    k: int = 12
    exec_k: int = 6
    n_steps: int = 12
    guidance: float = 1.0
    calibrate: bool = True
    seed: int = 0


def batch_of(pos: int):
    return {kk: v.to(R["device"]) for kk, v in collate([R["ds"][pos]]).items()}


def seg_of(row) -> dict:
    return {c: row[c] for c in CLEAR_COLS}


def obs_for(pos: int, obs: Observation) -> dict:
    row = HELD.iloc[pos]
    return observe(seg_of(row), obs, leg_type=row.get("leg_type"), phase=row.get("phase"))


# --------------------------------------------------------------------------------------
# sidebar: one case, shared by every page
# --------------------------------------------------------------------------------------

def _filtered() -> np.ndarray:
    h = HELD
    keep = np.ones(len(h), bool)
    f = st.session_state.get("filt_clause", [])
    if "turn" in f:
        keep &= h["lateral"].isin(["L", "R"]).to_numpy()
    if "climb" in f:
        keep &= (h["vertical"] == "climb").to_numpy()
    if "descend" in f:
        keep &= (h["vertical"] == "descend").to_numpy()
    if "speed" in f:
        keep &= h["speed"].isin(["accel", "decel"]).to_numpy()
    lt = st.session_state.get("filt_leg", "any")
    if lt != "any":
        keep &= (h["leg_type"] == lt).to_numpy()
    return np.where(keep)[0]


def sidebar():
    with st.sidebar:
        st.markdown("### Case")
        st.multiselect("instruction must contain", ["turn", "climb", "descend", "speed"],
                       key="filt_clause")
        st.selectbox("leg type", ["any", "arrival", "departure", "overflight"], key="filt_leg")
        cand = _filtered()
        if len(cand) == 0:
            st.warning("No held-out window matches these filters.")
            cand = np.arange(len(HELD))
        c1, c2 = st.columns([3, 1])
        st.session_state["win_n"] = min(int(st.session_state.get("win_n", 0)), len(cand) - 1)
        n = c1.number_input(f"window (of {len(cand)})", 0, len(cand) - 1, key="win_n")
        if c2.button("🎲", help="random window"):
            st.session_state["win_n_next"] = int(np.random.default_rng().integers(len(cand)))
            st.rerun()
        pos = int(cand[n])
        row = HELD.iloc[pos]
        st.caption(f"{row.get('callsign') or row['icao24']} · {row.get('leg_type', '?')} · "
                   f"{row['wake']} {row['engine_class']} · {row['leg_id']} seg {row['seg_idx']}. "
                   f"{len(HELD)} windows are held out from both stages.")
        days = sorted(R["store"].files) if R["store"] else []
        if days:
            st.caption(f"Raw ADS-B: {len(days)} days from `{A.adsb}`"
                       + ("" if row["day"] in days else f" (but not {row['day']})"))
        else:
            st.caption(f"No raw ADS-B found at `{A.adsb}`: no basemap, replay or dataset page.")

        st.markdown("### Observation")
        mode = st.segmented_control("what Stage 1 hears", list(MODES), format_func=MODES.get,
                                    default="channel", key="obs_mode") or "channel"
        profile, phrase_p, kw_p, conf, text = "jacktol", 0.35, 0.25, TRAINED_CONFIDENCE, ""
        speak = True
        if mode == "channel":
            chans = list_channels() or ["jacktol"]
            profile = st.selectbox("channel profile", chans,
                                   index=chans.index("jacktol") if "jacktol" in chans else 0)
            phrase_p = st.slider("phrase variation", 0.0, 1.0, 0.35, 0.05)
            kw_p = st.slider("keyword dropout", 0.0, 1.0, 0.25, 0.05)
        if mode == "custom":
            text = st.text_area("transcript", "southwest one two three turn left heading "
                                "two seven zero descend and maintain five thousand",
                                help="Stage 1 reads this with this window's aircraft state; "
                                     "Stage 2 predicts from this window's recorded past.")
            speak = st.toggle("speak numbers", value=True,
                              help="Rewrite numerals as a controller says them (270 → two "
                                   "seven zero, 5,000 → five thousand, FL350 → flight level "
                                   "three five zero). Training text always spelled numbers "
                                   "out, and the model misreads bare numerals.")
        if mode in ("clean", "custom"):
            conf = st.slider("confidence κ", 0.0, 1.0, TRAINED_CONFIDENCE, 0.01,
                             help="The shipped channel emits 0.82 ± 0.05, so that is the "
                                  "range Stage 1 was trained on.")
        context = st.toggle("wrap in radio context", value=(mode == "channel"),
                            disabled=(mode == "custom"),
                            help="Add a callsign and phase-coherent phraseology, as training did.")
        oseed = st.number_input("transcript seed", 0, 10_000, 0)
        obs = Observation(mode=mode, profile=profile, phrase_variant_p=phrase_p,
                          keyword_dropout_p=kw_p, context=context, confidence=conf,
                          custom_text=text, speak_numbers=speak, seed=int(oseed))

        st.markdown("### Sampling")
        k = st.slider("instruction draws (Stage 1)", 1, 32, 12)
        exec_k = st.slider("executions per instruction", 1, 16, 6)
        n_steps = st.slider("ODE steps", 4, 40, 12)
        guidance = st.slider("guidance", 0.0, 3.0, 1.0, 0.1)
        calibrate = st.toggle("apply fitted λ", value=A.lambda_json is not None,
                              disabled=A.lambda_json is None)
        sseed = st.number_input("sampling seed", 0, 10_000, 0)
        if A.lambda_json is None:
            st.caption("No λ file was passed, so coverage will read low.")
        samp = Sampling(k=k, exec_k=exec_k, n_steps=n_steps, guidance=guidance,
                        calibrate=calibrate, seed=int(sseed))
    return pos, obs, samp


if "win_n_next" in st.session_state:
    st.session_state["win_n"] = st.session_state.pop("win_n_next")


# --------------------------------------------------------------------------------------
# cached computations, keyed on plain values
# --------------------------------------------------------------------------------------

@st.cache_data(max_entries=24, show_spinner="Running both stages…")
def forecast_for(pos: int, obs: Observation, samp: Sampling):
    o = obs_for(pos, obs)
    b = batch_of(pos)
    dev = R["device"]
    toks = torch.tensor([R["vocab"].encode(o["text"], R["max_len"])], device=dev)
    conf = torch.tensor([o["confidence"]], device=dev, dtype=torch.float32)
    lam = R["lam"] if samp.calibrate else None
    out = run_forecast(R["stage1"], R["flow"], b, toks, conf, k=samp.k, exec_k=samp.exec_k,
                       n_steps=samp.n_steps, guidance=samp.guidance, lam=lam, seed=samp.seed)
    conds = {}
    for c, d in out["conditions"].items():
        conds[c] = {**pos_physical(d["pos"][0]), **ctrl_physical(d["ctrl"][0]),
                    "metrics": d["metrics"], "src": d["src"]}
    view = {"past": past_physical(b), "truth": truth_physical(b)}
    return o, out["greedy"], out["samples"], conds, view


@st.cache_data(max_entries=24, show_spinner=False)
def geo_for(pos: int):
    """Onset lat/lon and the reference features around the window, if ADS-B is present."""
    row = HELD.iloc[pos]
    if not R["store"] or not R["store"].has_day(row["day"]):
        return None
    track = clean_track(R["store"].aircraft(row["day"], str(row["icao24"])))
    lat0, lon0 = onset_latlon(track, float(row["onset"]))
    pad = 0.12
    feats = features_near(lat0 - pad, lat0 + pad, lon0 - pad, lon0 + pad, margin_deg=0.1)
    return {"lat0": lat0, "lon0": lon0, "feats": feats}


@st.cache_data(max_entries=16, show_spinner="Tracing Stage 1…")
def stage1_trace_for(text: str, conf: float, pos: int, k: int, seed: int):
    return trace_stage1(R["stage1"], R["vocab"], R["max_len"], text, conf,
                        onset_state(batch_of(pos)), k=k, seed=seed)


@st.cache_data(max_entries=16, show_spinner="Tracing Stage 2…")
def stage2_trace_for(pos: int, directive: tuple | None, n_steps: int, k: int,
                     guidance: float, seed: int):
    b = batch_of(pos)
    if directive is not None:
        b["directive"] = torch.tensor([directive], device=R["device"], dtype=torch.float32)
    return trace_stage2(R["flow"], b, n_steps=n_steps, k=k, guidance=guidance, seed=seed)


def leg_positions(leg_id: str) -> list[int]:
    m = HELD[HELD["leg_id"] == leg_id].sort_values("onset")
    return m.index.tolist()


@st.cache_data(max_entries=8, show_spinner="Flying the leg with both models…")
def replay_for(leg_id: str, obs: Observation, k: int, exec_k: int, n_steps: int,
               guidance: float, calibrate: bool, conditions: tuple, seed: int):
    positions = leg_positions(leg_id)
    row = HELD.iloc[positions[0]]
    return build_replay(R["ds"], R["wj"], R["idx"], positions, R["store"], R["stage1"],
                        R["vocab"], R["max_len"], R["flow"], R["device"],
                        R["lam"] if calibrate else None, obs,
                        {"callsign": row.get("callsign"), "leg_type": row.get("leg_type")},
                        k=k, exec_k=exec_k, n_steps=n_steps, guidance=guidance,
                        conditions=conditions, seed=seed)


@st.cache_data(max_entries=8, show_spinner="Rebuilding this leg from raw ADS-B…")
def story_for(pos: int, obs: Observation):
    row = HELD.iloc[pos]
    raw = R["store"].aircraft(row["day"], str(row["icao24"]))
    i = R["idx"][pos]
    npz = R["npz"]
    window = {k: npz[k][i] for k in ("past", "past_time", "past_mask", "fut_pos", "fut_mask",
                                     "directive")}
    return build_story(raw, row["day"], row["leg_id"], float(row["onset"]), window, obs,
                       leg_type=row.get("leg_type"))


# --------------------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------------------

def _card(title, body_md, caption=None):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(body_md)
        if caption:
            st.caption(caption)


def page_forecast():
    pos, obs, samp = CASE
    o, greedy, samples, conds, view = forecast_for(pos, obs, samp)
    row = HELD.iloc[pos]

    c1, c2, c3 = st.columns([1.25, 1, 1.2])
    with c1:
        unk = unknown_words(o["text"], R["vocab"])
        _card("What Stage 1 hears", f"`{o['text'] or '(nothing)'}`",
              f"{MODES[obs.mode]} · confidence {o['confidence']:.2f}"
              + (f" · channel: {o['corrupted'].replace('|', ', ')}" if o["corrupted"] else "")
              + (f" · typed: “{o['core']}”" if obs.mode == "custom" and o["core"] != o["text"]
                 else "")
              + (f" · not in vocabulary, read as <unk>: {', '.join(unk)}" if unk else ""))
    with c2:
        _card("What was issued", "  \n".join(f"- {c}" for c in o["issued"]) or "(no clause)",
              f"{row.get('callsign') or row['icao24']} · {row.get('leg_type', '?')} · "
              f"{row['wake']} {row['engine_class']}")
    with c3:
        modes = posterior_modes(samples)
        with st.container(border=True):
            st.markdown(f"**Stage 1 posterior** · {samp.k} draws, T = "
                        f"{R['stage1'].temperature:.2f}")
            st.dataframe(pd.DataFrame(modes[:5], columns=["instruction", "share"]),
                         hide_index=True, width="stretch", height=150,
                         column_config={"share": st.column_config.ProgressColumn(
                             "share", min_value=0.0, max_value=1.0, format="percent")})

    chosen = st.pills("conditions", list(CONDITIONS), selection_mode="multi",
                      default=["belief", "point", "oracle"], format_func=LABELS.get,
                      key="conds")
    if not chosen:
        st.info("Pick at least one condition.")
        return
    sel = {c: conds[c] for c in CONDITIONS if c in chosen}

    tab_map, tab_3d, tab_m = st.tabs(["Plan view and axis profiles", "3-D", "Metrics"])
    with tab_map:
        left, right = st.columns([1.05, 1])
        with left:
            geo = geo_for(pos)
            if geo:
                op = st.slider("basemap opacity", 0.0, 1.0, 0.55, 0.05, key="tile_op")
                st.plotly_chart(plots.plan_map({**view, **geo}, sel, geo["feats"], op),
                                width="stretch", config={"scrollZoom": True})
                st.caption("Airports, runways and navaids from OurAirports; the thick line of "
                           "each colour is that condition's median path.")
            else:
                st.plotly_chart(plots.relative_plan(view, sel), width="stretch")
                st.caption("No raw ADS-B for this day, so the plan view is relative to the "
                           "onset. Pass --adsb to place it on the map.")
        with right:
            targets = {"lateral": None, "vertical": None, "speed": None}
            d = batch_of(pos)["directive"][0].cpu().numpy()
            if d[0] > 0.5:
                targets["lateral"] = float(np.degrees(np.arctan2(d[2], d[3])) % 360)
            if d[4] > 0.5:
                targets["vertical"] = float(d[6] * ALT_SCALE_FT)
            if d[7] > 0.5:
                targets["speed"] = float(d[9] * GS_SCALE_KT)
            g, s = greedy, samples
            draws = {
                "lateral": (s["heading_deg"][0][s["has_turn"][0] > 0.5],
                            float(g["heading_deg"][0]) if g["has_turn"][0] > 0.5 else None),
                "vertical": (s["altitude_ft"][0][s["has_vert"][0] > 0.5],
                             float(g["altitude_ft"][0]) if g["has_vert"][0] > 0.5 else None),
                "speed": (s["speed_kt"][0][s["has_speed"][0] > 0.5],
                          float(g["speed_kt"][0]) if g["has_speed"][0] > 0.5 else None),
            }
            show_lines = st.toggle("show individual samples", value=False, key="show_lines")
            st.plotly_chart(plots.axis_profiles(view, sel, targets, draws, show_lines),
                            width="stretch")
            st.caption("Bands are the 10–90% and 25–75% ranges of each condition. Dashed: the "
                       "issued target. Right margin: the targets Stage 1 decoded (ticks are "
                       "draws, the diamond is the best guess). Instruction speeds are indicated "
                       "airspeed and this is ground speed, so wind separates the two.")
    with tab_3d:
        c1, c2 = st.columns([1, 1])
        color_by = c1.segmented_control("colour by", ["condition", "speed"], default="condition",
                                        key="c3d") or "condition"
        zx = c2.slider("vertical exaggeration", 0.5, 4.0, 1.0, 0.25, key="z3d")
        st.plotly_chart(plots.view_3d(view, sel, color_by=color_by, z_exag=zx), width="stretch")
        st.caption("Drag to rotate. The shadow is the recorded ground track. Colouring by "
                   "speed adds the fourth dimension an instruction can command.")
    with tab_m:
        show = ["kde", "cov80", "energy_traj", "energy_endpoint", "avgADE_nm", "minADE_nm",
                "vMAE_ft"]
        df = pd.DataFrame({LABELS[c]: conds[c]["metrics"] for c in chosen}).T[show]
        st.dataframe(df.round(4), width="stretch")
        st.caption("kde is the log-density of the true endpoint under the predicted cloud, "
                   "so higher is better; the energy scores are proper, so lower is better. "
                   "One window is noisy; evaluation.compose gives the aggregate.")


def page_replay():
    pos, obs, samp = CASE
    row = HELD.iloc[pos]
    if not R["store"] or not R["store"].has_day(row["day"]):
        st.warning("The replay needs the raw ADS-B for this day. Pass --adsb with a glob of "
                   "the adsb_<day>.parquet files the windows were built from.")
        return
    legs = (HELD.groupby("leg_id").agg(n=("onset", "size"), cs=("callsign", "first"),
                                       lt=("leg_type", "first")).reset_index())
    legs = legs[legs["n"] >= 2].sort_values("n", ascending=False)
    opts = [row["leg_id"]] + [g for g in legs["leg_id"] if g != row["leg_id"]]
    lab = {r.leg_id: f"{r.cs or r.leg_id.split('_')[1]} · {r.lt} · {r.n} instructions"
           for r in legs.itertuples()}
    c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1.6])
    leg = c1.selectbox("leg", opts, format_func=lambda g: lab.get(g, g) + (
        "  (current window)" if g == row["leg_id"] else ""))
    k = c2.slider("instruction draws", 2, 16, 8, key="rk")
    ek = c3.slider("executions each", 1, 8, 3, key="rek")
    conds = c4.pills("predictions", list(CONDITIONS), selection_mode="multi",
                     default=["belief", "point"], format_func=LABELS.get, key="rconds")
    if not conds:
        st.info("Pick at least one condition.")
        return
    n = len(leg_positions(leg))
    if n > 60:
        st.caption(f"This leg has {n} instructions; building it takes a little longer.")
    payload = replay_for(leg, obs, k, ek, min(samp.n_steps, 12), samp.guidance,
                         samp.calibrate, tuple(c for c in CONDITIONS if c in conds), samp.seed)
    if obs.mode == "custom":
        st.info("Custom text is heard at EVERY instruction in the leg, so each prediction shows "
                "how this flight would respond to that one instruction at that moment.")
    components.render("replay", payload, height=820)
    st.caption("The aircraft flies its recorded ADS-B track. At each maneuver onset the "
               "instruction is heard, Stage 1 decodes it and Stage 2 predicts the next minute "
               "from the past it has actually seen; the aircraft then flies on through that "
               "prediction. Space toggles playback; drag the timeline to scrub; click a radio "
               "log entry to jump to it.")


def page_models():
    pos, obs, samp = CASE
    o = obs_for(pos, obs)
    t1, t2 = st.tabs(["Stage 1 · instruction posterior", "Stage 2 · trajectory flow"])
    with t1:
        c1, c2, c3 = st.columns([3, 1, 1])
        text = c1.text_input("transcript to trace", o["text"], key="tr_text",
                             help="Defaults to the current observation; edit it to probe.")
        conf = c2.slider("confidence κ", 0.0, 1.0, float(round(o["confidence"], 2)), 0.01,
                         key="tr_conf")
        k = c3.slider("draws", 4, 32, 12, key="tr_k")
        tr = stage1_trace_for(text, conf, pos, k, samp.seed)
        components.render("stage1", {"trace": tr}, height=1000)
        st.caption("Traced on the real checkpoint, operation by operation. The trace checks "
                   "itself against the model's own forward pass and sampler before it is "
                   "shown. The aircraft state that grounds the value heads is this window's "
                   "onset state.")
    with t2:
        greedy_src = forecast_for(pos, obs, samp)[1]
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        src = c1.segmented_control("instruction fed to Stage 2",
                                   ["oracle", "Stage-1 MAP", "none (dropped)"],
                                   default="oracle", key="tr2_src") or "oracle"
        n_steps = c2.slider("ODE steps", 4, 20, 10, key="tr2_n")
        k = c3.slider("samples", 2, 16, 8, key="tr2_k")
        guidance = c4.slider("guidance", 0.0, 3.0, 1.0, 0.1, key="tr2_g")
        directive = None
        if src == "Stage-1 MAP":
            from atc_forecast.evaluation.compose import vec_from_fields
            g = {kk: torch.tensor(v) for kk, v in greedy_src.items()}
            directive = tuple(float(x) for x in vec_from_fields(g)[0])
        if src == "none (dropped)":
            guidance = 0.0
        tr = stage2_trace_for(pos, directive, n_steps, k, guidance, samp.seed)
        components.render("stage2", {"trace": tr}, height=1180)
        st.caption("Traced on the real checkpoint and checked against FlowPredictor.sample. "
                   "Attention is shown for the first sample; the samples share the memory, so "
                   "their cross-attention differs only through their own noise.")


def page_dataset():
    pos, obs, samp = CASE
    row = HELD.iloc[pos]
    if not R["store"] or not R["store"].has_day(row["day"]):
        st.warning("Rebuilding the leg needs the raw ADS-B for this day. Pass --adsb.")
        return
    story = story_for(pos, obs)
    components.render("dataset", {"story": story}, height=900)
    st.caption("Every step runs the package's own functions on this aircraft's raw ADS-B "
               "(segmentation, phraseology, legs) and shows the stored window it produced. "
               "Use ← → or the step pills; the window in focus is the one selected in the "
               "sidebar.")


CASE = sidebar()
nav = st.navigation([
    st.Page(page_forecast, title="Prediction", icon=":material/flight:", url_path="prediction"),
    st.Page(page_replay, title="Flight replay", icon=":material/play_circle:", url_path="replay"),
    st.Page(page_models, title="Inside the models", icon=":material/hub:", url_path="models"),
    st.Page(page_dataset, title="How the data is built", icon=":material/dataset:",
            url_path="dataset"),
], position="top")
nav.run()
