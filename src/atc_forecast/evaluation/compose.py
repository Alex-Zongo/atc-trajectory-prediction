"""The headline experiment: composing Stage 1 and Stage 2.

For each held-out window the clearance's transcript is passed through the observation
channel, Stage 1 produces a distribution over what was said, and Stage 2 turns that into
trajectories. Four ways of using the instruction are compared:

  oracle    the TRUE clearance; the ceiling, instruction known perfectly
  none      clearance dropped (guidance 0); the past-only baseline
  point     Stage 1's single MAP clearance; a confident point estimate
  belief    k clearances SAMPLED from Stage 1; propagates the communication uncertainty

The claim under test: as the transcript degrades, the POINT estimate goes confidently
wrong, with its samples clustered around one mistaken clearance, while the BELIEF estimate
hedges, spreading across the plausible clearances and continuing to cover the truth. So the
two have similar best-of-k point error, and belief wins on the proper distributional scores
exactly where it matters.

The composition is done PER CLEARANCE, not pooled. For each sampled clearance, `exec_k`
execution trajectories are drawn and THAT clearance's ensemble is inflated by the Stage-2
recalibration factor, which fixes the flow's execution under-dispersion; only then are the
clouds pooled. The pooled cloud is therefore communication spread plus calibrated execution
spread. Reusing the execution factor on an already-pooled, communication-dominated cloud
would inflate the wrong thing.

Windows are held out from BOTH models (the intersection of the two leg-level validation
splits), so neither stage has seen them.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..channel.model import NoiseConfig, corrupt_segment
from ..data.text import augment_transcript
from ..models.stage1 import load_stage1
from ..models.stage2 import N_CTRL, WindowDataset, collate, deadreckon, load_stage2
from ..units import ALT_SCALE_FT, GS_SCALE_KT
from .scores import endpoint_metrics, energy_score_traj
from .stage2 import lam_per_step

CLEAR_COLS = ["lateral", "vertical", "speed", "hdg_exit", "alt_exit_ft", "spd_exit_kt"]
CONDITIONS = ("oracle", "none", "point", "belief")


def clearance_to_vec(ht, tr, hdg_deg, hv, climb, alt_ft, hs, accel, spd_kt):
    """Build the 10-dim clearance vector, matching windows.directive_vector exactly."""
    h = torch.deg2rad(hdg_deg)
    return torch.stack([ht, tr, torch.sin(h) * ht, torch.cos(h) * ht,
                        hv, climb, (alt_ft / ALT_SCALE_FT) * hv,
                        hs, accel, (spd_kt / GS_SCALE_KT) * hs], dim=-1)


def vec_from_fields(f, j=None):
    """A Stage-1 field dict to a clearance vector. `j` selects the j-th sample column."""
    g = {k: (v[:, j] if j is not None else v) for k, v in f.items()}
    return clearance_to_vec(g["has_turn"], g["turn_right"], g["heading_deg"], g["has_vert"],
                            g["climb"], g["altitude_ft"], g["has_speed"], g["accel"],
                            g["speed_kt"])


def onset_state(batch):
    """The current-state vector for Stage-1 grounding.

    The past feature order is [dx, dy, alt, gs, hdg_sin, hdg_cos, vrate], so the grounding
    state [alt, vrate, gs, sin, cos] is past[:, -1] indexed [2, 6, 3, 4, 5]. This must match
    `attach_onset_state`, since a grounded model run on a mis-ordered state degrades
    silently and badly.
    """
    return batch["past"][:, -1][:, [2, 6, 3, 4, 5]]


def stage1_clearances(model, tokens, confidence, k, state=None):
    """(point MAP vector [B, 10], list of k sampled vectors).

    A grounded model called WITHOUT its state falls back to the identity modulation and
    loses most of its numeric accuracy, so this warns rather than letting a caller that
    forgot the state quietly get much worse numbers.
    """
    if getattr(model, "state_films", None) is not None and state is None:
        import warnings
        warnings.warn("state-grounded Stage 1 called with state=None; the value heads fall "
                      "back to ungrounded and accuracy drops sharply. Pass onset_state(batch).",
                      RuntimeWarning, stacklevel=2)
    st = state if getattr(model, "state_films", None) is not None else None
    point = vec_from_fields(model.greedy(tokens, confidence, state=st))
    s = model.sample(tokens, confidence, k, state=st)
    return point, [vec_from_fields(s, j) for j in range(k)]


@torch.no_grad()
def calibrated_cloud(flow, batch, clearances, exec_k, n_steps, lam, onset_alt, guidance=1.0,
                     return_ctrl=False):
    """Positions [B, ., H, 3] pooled over clearances, each inflated before pooling.

    `return_ctrl` also returns the sampled controls [B, ., H, 4] in the same order. The
    inflation acts on positions only, so these are the flow's own, uninflated controls.
    """
    H = batch["fut_ctrl"].shape[1]
    clouds, ctrls = [], []
    for dv in clearances:
        b = dict(batch)
        if dv is not None:
            b["directive"] = dv
        # a dropped clearance is always guidance 0, which IS the no-instruction baseline
        ctrl = flow.sample(b, n_steps=n_steps, k=exec_k,
                           guidance=0.0 if dv is None else guidance).view(-1, exec_k, H, N_CTRL)
        B = ctrl.shape[0]
        pos = deadreckon(ctrl.reshape(B * exec_k, H, N_CTRL),
                         onset_alt.repeat_interleave(exec_k)).reshape(B, exec_k, H, 3)
        mean = pos.mean(1, keepdim=True)
        clouds.append(mean + lam * (pos - mean))
        ctrls.append(ctrl)
    if return_ctrl:
        return torch.cat(clouds, dim=1), torch.cat(ctrls, dim=1)
    return torch.cat(clouds, dim=1)


def build_windows(window_dir, meta_path, seed=0):
    """Windows held out from BOTH stages, with the clearance fields joined on so their
    transcripts can be regenerated."""
    window_dir = Path(window_dir)
    npz = np.load(window_dir / "windows.npz")
    wm = pd.read_parquet(window_dir / "windows_meta.parquet")
    meta = pd.read_parquet(meta_path)

    def valset(legs):
        rng = np.random.default_rng(seed)
        return set(rng.choice(legs, size=max(1, int(0.1 * len(legs))), replace=False))

    both = valset(wm["leg_id"].unique()) & valset(meta["leg_id"].unique())
    extra = [c for c in ("leg_type", "phase") if c in meta.columns]
    wj = wm.merge(meta[["leg_id", "seg_idx", *CLEAR_COLS[1:], "lateral", *extra]],
                  on=["leg_id", "seg_idx"], how="left")
    keep = wj["leg_id"].isin(both) & wj["lateral"].notna()
    return npz, wj, np.where(keep.values)[0]


@torch.no_grad()
def compose(stage1, vocab, max_len, flow, npz, wj, idx, device, lam, profile="jacktol",
            k=12, exec_k=6, n_steps=12, batch_size=64, guidance=1.0, seed=0,
            phrase_variant_p=0.0, keyword_dropout_p=0.0, conditions=CONDITIONS):
    """Run every condition over the held-out windows and return one row per condition."""
    ds = WindowDataset(npz, wj, idx)
    rng = np.random.default_rng(seed)
    rows = {c: [] for c in conditions}
    traj = {c: [] for c in conditions}

    for s0 in range(0, len(idx), batch_size):
        loc = list(range(s0, min(s0 + batch_size, len(idx))))
        batch = {kk: v.to(device) for kk, v in collate([ds[j] for j in loc]).items()}
        onset_alt = batch["past"][:, -1, 2] * ALT_SCALE_FT
        sub = wj.iloc[idx[loc]]

        texts = []
        for r in sub.itertuples():
            seg = {c: getattr(r, c) for c in CLEAR_COLS}
            n = corrupt_segment(seg, NoiseConfig(profile=profile,
                                                 phrase_variant_p=phrase_variant_p,
                                                 keyword_dropout_p=keyword_dropout_p), rng)
            texts.append(augment_transcript(n["noisy_directive"], rng,
                                            leg_type=getattr(r, "leg_type", None),
                                            phase=getattr(r, "phase", None)))
        toks = torch.tensor([vocab.encode(t, max_len) for t in texts], device=device)
        conf = torch.full((len(loc),), 0.82, device=device)
        point, sampled = stage1_clearances(stage1, toks, conf, k, state=onset_state(batch))

        plans = {
            "oracle": ([batch["directive"]], k * exec_k),
            "none": ([None], k * exec_k),
            "point": ([point], k * exec_k),
            "belief": (sampled, exec_k),
        }
        for cond in conditions:
            dirs, ek = plans[cond]
            cloud = calibrated_cloud(flow, batch, dirs, ek, n_steps, lam, onset_alt, guidance)
            rows[cond].append(endpoint_metrics(cloud, batch["fut_pos"], batch["fut_mask"]))
            traj[cond].append(float(energy_score_traj(cloud, batch["fut_pos"],
                                                      batch["fut_mask"]).mean()))
        print(f"  {min(s0 + batch_size, len(idx))}/{len(idx)}", flush=True)

    out = []
    for cond in conditions:
        m = pd.DataFrame(rows[cond]).mean().to_dict()
        m["energy_traj"] = float(np.mean(traj[cond]))
        m["condition"] = cond
        out.append(m)
    return pd.DataFrame(out).set_index("condition")


def main():
    from ..training.common import pick_device

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--data", required=True, help="window directory")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--lambda-json", default=None,
                    help="fitted lambda from evaluation.stage2; without it no inflation is applied")
    ap.add_argument("--profile", default="jacktol")
    ap.add_argument("--k", type=int, default=12, help="clearance samples from Stage 1")
    ap.add_argument("--exec-k", type=int, default=6, help="execution draws per clearance")
    ap.add_argument("--n-steps", type=int, default=12, help="ODE integration steps")
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--phrase-variant-p", type=float, default=0.35)
    ap.add_argument("--keyword-dropout-p", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the table to this CSV")
    a = ap.parse_args()

    device = pick_device()
    stage1, vocab, max_len = load_stage1(a.stage1, device)
    flow = load_stage2(a.stage2, device)
    npz, wj, idx = build_windows(a.data, a.meta, a.seed)
    H = npz["fut_ctrl"].shape[1]
    if a.lambda_json:
        lam = lam_per_step(json.loads(Path(a.lambda_json).read_text())["fit"], H, device)
    else:
        print("no --lambda-json: running UNCALIBRATED, so coverage will read low")
        lam = torch.ones(H, 3, device=device)

    print(f"{len(idx)} windows held out from both stages | channel {a.profile}")
    df = compose(stage1, vocab, max_len, flow, npz, wj, idx, device, lam, profile=a.profile,
                 k=a.k, exec_k=a.exec_k, n_steps=a.n_steps, batch_size=a.batch_size,
                 guidance=a.guidance, seed=a.seed, phrase_variant_p=a.phrase_variant_p,
                 keyword_dropout_p=a.keyword_dropout_p)
    cols = ["kde", "cov80", "energy_traj", "energy_endpoint", "avgADE_nm", "minADE_nm", "vMAE_ft"]
    print("\n" + df[cols].round(4).to_string())
    if {"belief", "point"} <= set(df.index):
        print(f"\nbelief over point: {df.loc['belief', 'kde'] - df.loc['point', 'kde']:+.2f} nats "
              f"of endpoint log-density, at a cost of "
              f"{df.loc['belief', 'avgADE_nm'] - df.loc['point', 'avgADE_nm']:+.4f} nm avgADE")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
