"""Interactive viewer for the two-stage forecast.

Pick a held-out window, see the transcript the channel produced, the clearance posterior
Stage 1 inferred from it, and the trajectory distribution Stage 2 produced under each
condition, with the metrics underneath.

    streamlit run src/atc_forecast/app/viewer.py -- \
        --stage1 runs/stage1.pt --stage2 runs/stage2.pt \
        --data data/windows_raw --meta data/dataset/meta.parquet --lambda-json runs/lambda.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from atc_forecast.channel.model import NoiseConfig, corrupt_segment, list_channels
from atc_forecast.data.text import augment_transcript
from atc_forecast.evaluation.compose import (CLEAR_COLS, build_windows, calibrated_cloud,
                                             onset_state, stage1_clearances)
from atc_forecast.evaluation.scores import endpoint_metrics, energy_score_traj
from atc_forecast.evaluation.stage2 import lam_per_step
from atc_forecast.models.stage1 import load_stage1
from atc_forecast.models.stage2 import WindowDataset, collate, load_stage2
from atc_forecast.training.common import pick_device
from atc_forecast.units import ALT_SCALE_FT, M_TO_NM, POS_SCALE_M

CONDITION_STYLE = {"oracle": ("tab:green", "oracle clearance"),
                   "none": ("tab:gray", "no instruction"),
                   "point": ("tab:red", "point estimate"),
                   "belief": ("tab:blue", "belief (sampled)")}


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--lambda-json", default=None)
    return ap.parse_args(sys.argv[1:])


@st.cache_resource
def load_all(stage1_path, stage2_path, data_dir, meta_path, lambda_path):
    device = pick_device()
    stage1, vocab, max_len = load_stage1(stage1_path, device)
    flow = load_stage2(stage2_path, device)
    npz, wj, idx = build_windows(data_dir, meta_path)
    H = npz["fut_ctrl"].shape[1]
    if lambda_path:
        lam = lam_per_step(json.loads(Path(lambda_path).read_text())["fit"], H, device)
    else:
        lam = torch.ones(H, 3, device=device)
    return device, stage1, vocab, max_len, flow, npz, wj, idx, lam


def plot_cloud(ax, cloud, truth, mask, colour, label, alpha=0.18):
    """Horizontal plan view in nm, with the valid part of the truth overlaid."""
    c = cloud[..., :2].cpu().numpy() * POS_SCALE_M * M_TO_NM
    m = mask.bool().cpu().numpy()
    for i in range(c.shape[0]):
        ax.plot(c[i, m, 0], c[i, m, 1], color=colour, alpha=alpha, lw=0.8)
    ax.plot([], [], color=colour, lw=1.5, label=label)
    t = truth[..., :2].cpu().numpy() * POS_SCALE_M * M_TO_NM
    ax.plot(t[m, 0], t[m, 1], "k-", lw=2.0, label="truth")
    ax.plot(0, 0, "ko", ms=5)


def main():
    a = cli()
    st.set_page_config(page_title="ATC trajectory forecast", layout="wide")
    st.title("Trajectory forecasting under clearance uncertainty")

    device, stage1, vocab, max_len, flow, npz, wj, idx, lam = load_all(
        a.stage1, a.stage2, a.data, a.meta, a.lambda_json)

    with st.sidebar:
        st.header("window")
        j = st.number_input("index", 0, len(idx) - 1, 0, step=1)
        st.caption(f"{len(idx)} windows held out from both stages")
        st.header("channel")
        channels = list_channels() or ["jacktol"]
        profile = st.selectbox("observation channel", channels,
                               index=channels.index("jacktol") if "jacktol" in channels else 0)
        phrase_p = st.slider("phrase variation", 0.0, 1.0, 0.35, 0.05)
        kw_p = st.slider("keyword dropout", 0.0, 1.0, 0.25, 0.05)
        seed = st.number_input("transcript seed", 0, 10_000, 0, step=1)
        st.header("sampling")
        k = st.slider("clearance samples (Stage 1)", 1, 32, 12)
        exec_k = st.slider("execution draws per clearance", 1, 16, 6)
        n_steps = st.slider("ODE steps", 4, 40, 12)
        guidance = st.slider("guidance", 0.0, 3.0, 1.0, 0.1)
        calibrate = st.checkbox("apply fitted lambda", value=a.lambda_json is not None)
        if a.lambda_json is None:
            st.caption("no lambda file passed, so coverage will read low")

    ds = WindowDataset(npz, wj, idx)
    batch = {kk: v.to(device) for kk, v in collate([ds[int(j)]]).items()}
    row = wj.iloc[idx[int(j)]]
    onset_alt = batch["past"][:, -1, 2] * ALT_SCALE_FT
    H = batch["fut_ctrl"].shape[1]
    lam_used = lam if calibrate else torch.ones(H, 3, device=device)

    rng = np.random.default_rng(int(seed))
    seg = {c: row[c] for c in CLEAR_COLS}
    noisy = corrupt_segment(seg, NoiseConfig(profile=profile, phrase_variant_p=phrase_p,
                                             keyword_dropout_p=kw_p), rng)
    text = augment_transcript(noisy["noisy_directive"], rng,
                             leg_type=row.get("leg_type"), phase=row.get("phase"))
    toks = torch.tensor([vocab.encode(text, max_len)], device=device)
    conf = torch.tensor([noisy["asr_confidence"]], device=device, dtype=torch.float32)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("what the recognizer heard")
        st.code(text or "(nothing)", language=None)
        st.caption(f"confidence {noisy['asr_confidence']:.2f} | "
                   f"corruption: {noisy['corrupted'] or 'none'}")
    with c2:
        st.subheader("what was issued")
        issued = []
        if seg["lateral"] in ("L", "R"):
            issued.append(f"turn {'left' if seg['lateral'] == 'L' else 'right'} "
                          f"heading {noisy['issued_hdg']:.0f}")
        if seg["vertical"] in ("climb", "descend"):
            issued.append(f"{seg['vertical']} and maintain {noisy['issued_alt_ft']:.0f}")
        if seg["speed"] in ("accel", "decel"):
            verb = "increase" if seg["speed"] == "accel" else "reduce"
            issued.append(f"{verb} speed to {noisy['issued_spd_kt']:.0f}")
        st.write("; ".join(issued) or "(no clause)")
        st.caption(f"{row['leg_id']} seg {row['seg_idx']} | {row.get('wake', '?')} "
                   f"{row.get('engine_class', '?')}")

    point, sampled = stage1_clearances(stage1, toks, conf, k, state=onset_state(batch))

    st.subheader("Stage 1: the clearance posterior")
    s = stage1.sample(toks, conf, k, state=onset_state(batch))
    post = pd.DataFrame({
        "heading": s["heading_deg"][0].cpu().numpy().round(0),
        "altitude_ft": s["altitude_ft"][0].cpu().numpy().round(0),
        "speed_kt": s["speed_kt"][0].cpu().numpy().round(0),
        "turn": np.where(s["has_turn"][0].cpu().numpy() > 0.5,
                         np.where(s["turn_right"][0].cpu().numpy() > 0.5, "right", "left"), "-"),
        "vertical": np.where(s["has_vert"][0].cpu().numpy() > 0.5,
                             np.where(s["climb"][0].cpu().numpy() > 0.5, "climb", "descend"), "-"),
    })
    cols = st.columns(4)
    cols[0].metric("distinct headings", int(post.heading.nunique()))
    cols[1].metric("heading spread", f"{post.heading.std():.0f} deg")
    cols[2].metric("distinct altitudes", int(post.altitude_ft.nunique()))
    cols[3].metric("temperature", f"{stage1.temperature:.2f}")
    st.dataframe(post, height=200, width="stretch")
    st.caption("Each row is one coherent sampled clearance. Spread across rows is the "
               "communication uncertainty; a collapsed table means the model is certain.")

    st.subheader("Stage 2: the trajectory distribution")
    plans = {"oracle": ([batch["directive"]], k * exec_k), "none": ([None], k * exec_k),
             "point": ([point], k * exec_k), "belief": (sampled, exec_k)}
    chosen = st.multiselect("conditions", list(plans), default=["point", "belief"])
    if not chosen:
        st.info("pick at least one condition")
        return

    fig, axes = plt.subplots(1, len(chosen), figsize=(5.2 * len(chosen), 5.0), squeeze=False)
    metrics = {}
    for ax, cond in zip(axes[0], chosen):
        dirs, ek = plans[cond]
        cloud = calibrated_cloud(flow, batch, dirs, ek, n_steps, lam_used, onset_alt, guidance)
        colour, label = CONDITION_STYLE[cond]
        plot_cloud(ax, cloud[0], batch["fut_pos"][0], batch["fut_mask"][0], colour, label)
        ax.set_title(label)
        ax.set_xlabel("east (nm)")
        ax.set_ylabel("north (nm)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        m = endpoint_metrics(cloud, batch["fut_pos"], batch["fut_mask"])
        m["energy_traj"] = float(energy_score_traj(cloud, batch["fut_pos"],
                                                   batch["fut_mask"]).mean())
        metrics[cond] = m
    fig.tight_layout()
    st.pyplot(fig)

    show = ["kde", "cov80", "energy_traj", "energy_endpoint", "avgADE_nm", "vMAE_ft"]
    st.dataframe(pd.DataFrame(metrics).T[show].round(4), width="stretch")
    st.caption("kde is the log-density of the true endpoint under the predicted cloud, so "
               "higher is better. energy is a proper score, so lower is better. One window "
               "is noisy; use evaluation.compose for the aggregate.")


main()
