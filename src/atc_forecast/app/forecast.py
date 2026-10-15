"""Run the two stages on a batch of windows and return everything in physical units.

This is the composition `evaluation.compose` scores, kept intact: the same Stage-1
calls, the same per-clearance inflation before pooling, the same four conditions. It only
adds what a display needs: the controls behind each trajectory, which clearance sample
each belief trajectory came from, and a fixed seed so moving a slider does not silently
redraw every other sample.
"""

import numpy as np
import torch

from ..evaluation.compose import calibrated_cloud, onset_state, vec_from_fields
from ..evaluation.scores import endpoint_metrics, energy_score_traj
from ..units import (ALT_SCALE_FT, GS_SCALE_KT, M_TO_NM, POS_SCALE_M, VR_SCALE_FPM)

CONDITIONS = ("oracle", "none", "point", "belief")
LABELS = {"oracle": "Oracle clearance", "none": "No instruction",
          "point": "Point (Stage-1 MAP)", "belief": "Belief (Stage-1 samples)"}


def pos_physical(pos: np.ndarray) -> dict:
    """Normalised [..., 3] positions to east/north in nm and altitude in ft."""
    return {"east_nm": pos[..., 0] * POS_SCALE_M * M_TO_NM,
            "north_nm": pos[..., 1] * POS_SCALE_M * M_TO_NM,
            "alt_ft": pos[..., 2] * ALT_SCALE_FT}


def ctrl_physical(ctrl: np.ndarray) -> dict:
    """Normalised [..., 4] controls to ground speed, heading and vertical rate."""
    return {"gs_kt": ctrl[..., 0] * GS_SCALE_KT,
            "hdg_deg": np.degrees(np.arctan2(ctrl[..., 1], ctrl[..., 2])) % 360,
            "vr_fpm": ctrl[..., 3] * VR_SCALE_FPM}


def past_physical(batch, i: int = 0) -> dict:
    """The observed past of window i, oldest first, valid rows only."""
    p = batch["past"][i].cpu().numpy()
    m = batch["past_mask"][i].cpu().numpy() > 0.5
    t = (-batch["past_time"][i].cpu().numpy() if "past_time" in batch
         else np.arange(-p.shape[0], 0, dtype=float))
    out = {"t_s": t[m]}
    out.update(pos_physical(p[m][:, :3]))
    out.update(ctrl_physical(p[m][:, 3:7]))
    return out


def truth_physical(batch, i: int = 0) -> dict:
    """The recorded future of window i on the 1 Hz grid, with its validity mask."""
    out = {"mask": batch["fut_mask"][i].cpu().numpy() > 0.5,
           "t_s": np.arange(batch["fut_mask"].shape[1], dtype=float)}
    out.update(pos_physical(batch["fut_pos"][i].cpu().numpy()))
    out.update(ctrl_physical(batch["fut_ctrl"][i].cpu().numpy()))
    return out


def clearance_text(f: dict, j=None, i: int = 0) -> str:
    """A compact rendering of one decoded clearance, e.g. 'L 270 | climb 5,000'.

    Magnitudes are snapped to the grid a controller speaks on (1,000 ft, 10 kt), so two
    draws read as the same clearance exactly when they would have been issued as one.
    """
    g = {k: float(v[i, j] if j is not None else v[i]) for k, v in f.items()}
    if g["is_clearance"] < 0.5:
        return "no instruction"
    parts = []
    if g["has_turn"] > 0.5:
        hdg = int(round(g["heading_deg"])) % 360 or 360
        parts.append(f"{'R' if g['turn_right'] > 0.5 else 'L'} {hdg:03d}")
    if g["has_vert"] > 0.5:
        a = max(int(round(g["altitude_ft"] / 1000.0)), 1) * 1000
        parts.append(f"{'climb' if g['climb'] > 0.5 else 'descend'} "
                     + (f"FL{a // 100}" if a >= 18000 else f"{a:,}"))
    if g["has_speed"] > 0.5:
        parts.append(f"{'+' if g['accel'] > 0.5 else '-'}spd {int(round(g['speed_kt'] / 10) * 10)}")
    return " | ".join(parts) if parts else "clearance, no clause"


def _fields_np(f: dict) -> dict:
    return {k: v.detach().cpu().numpy() for k, v in f.items()}


@torch.no_grad()
def run_stage1(stage1, toks, conf, state, k: int, seed: int):
    """MAP clearance, k sampled clearances, and both as field dicts and vectors."""
    # the body of evaluation.compose.stage1_clearances, keeping the field dicts it discards
    st_ = state if getattr(stage1, "state_films", None) is not None else None
    torch.manual_seed(seed)
    greedy = stage1.greedy(toks, conf, state=st_)
    samples = stage1.sample(toks, conf, k, state=st_)
    point_vec = vec_from_fields(greedy)
    sampled_vecs = [vec_from_fields(samples, j) for j in range(k)]
    return point_vec, sampled_vecs, _fields_np(greedy), _fields_np(samples)


@torch.no_grad()
def run_forecast(stage1, flow, batch, toks, conf, *, k=12, exec_k=6, n_steps=12,
                 guidance=1.0, lam=None, conditions=CONDITIONS, seed=0, metrics=True):
    """Every requested condition for every window in `batch`.

    Returns a dict with the Stage-1 decodes and, per condition, numpy arrays
    pos [B, N, H, 3] (inflated), ctrl [B, N, H, 4] (raw) and, for belief, `src` [N]: the
    index of the clearance sample each trajectory was drawn under.
    """
    device = batch["past"].device
    H = batch["fut_ctrl"].shape[1]
    lam = lam if lam is not None else torch.ones(H, 3, device=device)
    onset_alt = batch["past"][:, -1, 2] * ALT_SCALE_FT
    state = onset_state(batch)
    point, sampled, greedy, samples = run_stage1(stage1, toks, conf, state, k, seed)

    plans = {"oracle": ([batch["directive"]], k * exec_k), "none": ([None], k * exec_k),
             "point": ([point], k * exec_k), "belief": (sampled, exec_k)}
    out = {"greedy": greedy, "samples": samples, "conditions": {}}
    for ci, cond in enumerate(conditions):
        dirs, ek = plans[cond]
        torch.manual_seed(seed * 1000 + ci)
        pos, ctrl = calibrated_cloud(flow, batch, dirs, ek, n_steps, lam, onset_alt, guidance,
                                     return_ctrl=True)
        c = {"pos": pos.cpu().numpy(), "ctrl": ctrl.cpu().numpy(),
             "src": np.repeat(np.arange(len(dirs)), ek)}
        if metrics:
            m = endpoint_metrics(pos, batch["fut_pos"], batch["fut_mask"])
            m["energy_traj"] = float(energy_score_traj(pos, batch["fut_pos"],
                                                       batch["fut_mask"]).mean())
            c["metrics"] = m
        out["conditions"][cond] = c
    return out


def posterior_modes(samples: dict, i: int = 0) -> list[tuple[str, float]]:
    """Distinct sampled clearances for window i and the share of samples behind each."""
    k = next(iter(samples.values())).shape[1]
    texts = [clearance_text(samples, j, i) for j in range(k)]
    vals, counts = np.unique(texts, return_counts=True)
    order = np.argsort(-counts)
    return [(str(vals[o]), float(counts[o] / k)) for o in order]
