"""Evaluate and recalibrate the Stage-1 clearance posterior.

The posterior is scored directly, as a joint distribution, not field by field. A per-field
marginal score cannot see the flaw the autoregressive model exists to fix: whether the ten
fields are jointly COHERENT and whether the posterior can be jointly multimodal.

  joint NLL         the exact joint log-score, which the model computes directly
  joint energy      the energy score on the 10-dim clearance vector, which penalises
                    incoherent joint samples that a per-field score misses. Its accuracy
                    and dispersion terms are reported separately, because a rising energy
                    score is otherwise ambiguous between samples drifting from the truth and
                    the posterior sharpening.
  heading modes     under digit confusion the true posterior is multimodal (200 becoming
                    290, 100 or 300). The mean number of occupied 30-degree arcs and the
                    circular resultant make that visible; a unimodal head collapses both.
"""

import argparse
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data.text import ClearanceDataset, DataConfig
from ..units import ALT_SCALE_FT, GS_SCALE_KT, HEAD_BIN_DEG, SPD_SCALE_KT
from .scores import energy_score, energy_terms


def sampled_clearance_vec(s: dict) -> torch.Tensor:
    """Stage-1 sample dict to [B, k, 10] clearance vectors, in the Stage-2 layout."""
    ih = torch.deg2rad(s["heading_deg"])
    ht = s["has_turn"]
    return torch.stack([ht, s["turn_right"], ih.sin() * ht, ih.cos() * ht,
                        s["has_vert"], s["climb"], s["altitude_ft"] / ALT_SCALE_FT,
                        s["has_speed"], s["accel"], s["speed_kt"] / GS_SCALE_KT], dim=-1)


def true_clearance_vec(batch: dict) -> torch.Tensor:
    """Batch targets to the [B, 10] true clearance vector, same layout and scaling."""
    ih = batch["head_bin"].float() * math.radians(HEAD_BIN_DEG)
    ht = batch["has_turn"]
    return torch.stack([ht, batch["turn_dir"], ih.sin() * ht, ih.cos() * ht,
                        batch["has_vert"], batch["vert_dir"], batch["alt_norm"],
                        batch["has_speed"], batch["spd_dir"], batch["spd_norm"]], dim=-1)


def heading_multimodality(heading_deg, has_turn, arc_deg=30.0):
    """For turn windows: mean occupied arcs and the circular resultant R. Low R means the
    posterior is spread or multimodal."""
    m = has_turn.float().mean(1) > 0.5
    if m.sum() < 3:
        return float("nan"), float("nan")
    hd = heading_deg[m]
    a = torch.deg2rad(hd)
    R = torch.sqrt(a.cos().mean(1) ** 2 + a.sin().mean(1) ** 2)
    n_arc = int(360 // arc_deg)
    bins = (hd / arc_deg).long() % n_arc
    occ = torch.zeros(hd.shape[0], n_arc, device=hd.device)
    occ.scatter_add_(1, bins, torch.ones_like(hd))
    return float(((occ / hd.shape[1]) > 0.05).sum(1).float().mean()), float(R.mean())


def _loader(meta, vocab, profile, seed, batch_size, degradation=None,
            phrase_variant_p=0.0, keyword_dropout_p=0.0):
    """Deterministic loader for scoring.

    `batch_size` is clamped to the split size, because with drop_last a split smaller than
    one batch yields NOTHING and every metric comes back as nan.
    """
    dcfg = DataConfig(profile=profile, fixed_degradation=degradation,
                      phrase_variant_p=phrase_variant_p, keyword_dropout_p=keyword_dropout_p)
    ds = ClearanceDataset(meta, vocab, dcfg, seed=seed)
    if len(ds) == 0:
        raise ValueError("empty split passed to Stage-1 scoring")
    return DataLoader(ds, min(batch_size, len(ds)), shuffle=False, drop_last=True)


@torch.no_grad()
def evaluate_posterior(model, meta, vocab, device, profile="jacktol", k=32, batch_size=256,
                       max_batches=20, seed=123, degradation=None, phrase_variant_p=0.0,
                       keyword_dropout_p=0.0) -> dict:
    """Joint NLL, joint energy with its two terms, and heading multimodality.

    Evaluate on the channel and the phraseology rates the model was TRAINED on. A model
    trained on spoken channels and evaluated on canonical text is being scored on a
    distribution it never saw, and as training specialises that gap WIDENS, which shows up
    as the energy score rising while the NLL falls and is easily mistaken for collapse.
    """
    model.eval()
    dl = _loader(meta, vocab, profile, seed, batch_size, degradation,
                 phrase_variant_p, keyword_dropout_p)
    nll, es, acc, dis, mods, Rs = [], [], [], [], [], []
    for b, batch in enumerate(dl):
        if b >= max_batches:
            break
        batch = {kk: v.to(device) for kk, v in batch.items()}
        nll.append(model.loss(batch)["loss"].item())
        s = model.sample(batch["tokens"], batch["confidence"], k=k, state=batch.get("state"))
        vs, vy = sampled_clearance_vec(s), true_clearance_vec(batch)
        es.append(float(energy_score(vs, vy).mean()))
        a, d = energy_terms(vs, vy)
        acc.append(a)
        dis.append(d)
        mo, R = heading_multimodality(s["heading_deg"], s["has_turn"])
        mods.append(mo)
        Rs.append(R)
    if not es:
        raise ValueError("evaluate_posterior scored no batches")
    finite = [m for m in mods if m == m]
    return {"joint_nll": float(np.mean(nll)), "energy": float(np.mean(es)),
            "energy_accuracy": float(np.mean(acc)), "energy_dispersion": float(np.mean(dis)),
            # undefined when too few scored windows carry a turn clause
            "heading_modes": float(np.mean(finite)) if finite else float("nan"),
            "heading_R": float(np.mean([r for r in Rs if r == r])) if finite else float("nan")}


@torch.no_grad()
def fit_temperature(model, meta, vocab, device, profile="jacktol", grid=None, k=24,
                    batch_size=256, max_batches=8, seed=321, phrase_variant_p=0.0,
                    keyword_dropout_p=0.0) -> float:
    """Fit the sampling temperature that RECALIBRATES the posterior.

    The objective is the multivariate energy score on held-out data. Because that score is
    proper, its minimiser is the calibrated spread: it cannot be gamed by collapsing to a
    point. T below 1 sharpens an over-dispersed posterior and T above 1 widens an
    over-confident one, so the grid spans both; a state-grounded model commits harder and
    its optimum can sit above 1.

    The calibration text distribution MUST match deployment. A model trained with
    phraseology augmentation but calibrated on canonical text looks spuriously
    over-dispersed, and the fit then drives T to the floor and collapses the posterior, so
    pass the same augmentation rates used in training.

    Stores the result on `model.temperature` and returns it.
    """
    model.eval()
    grid = np.round(np.linspace(0.45, 2.00, 32), 3) if grid is None else grid
    dl = _loader(meta, vocab, profile, seed, batch_size,
                 phrase_variant_p=phrase_variant_p, keyword_dropout_p=keyword_dropout_p)
    cached = []
    for b, batch in enumerate(dl):
        if b >= max_batches:
            break
        batch = {kk: v.to(device) for kk, v in batch.items()}
        cached.append((batch["tokens"], batch["confidence"], true_clearance_vec(batch),
                       batch.get("state")))
    if not cached:
        raise ValueError("fit_temperature cached no batches")
    best_T, best = 1.0, float("inf")
    for T in grid:
        total = 0.0
        for toks, conf, vy, st in cached:
            s = model.sample(toks, conf, k=k, temperature=float(T), state=st)
            total += float(energy_score(sampled_clearance_vec(s), vy).mean())
        if total < best:
            best_T, best = float(T), total
    model.temperature = best_T
    return best_T


def main():
    from ..models.stage1 import load_stage1, save_stage1
    from ..training.common import pick_device, split_by_leg

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--profile", default="jacktol")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--max-batches", type=int, default=20)
    ap.add_argument("--phrase-variant-p", type=float, default=0.35)
    ap.add_argument("--keyword-dropout-p", type=float, default=0.25)
    ap.add_argument("--fit-temperature", action="store_true",
                    help="fit the recalibration temperature on a disjoint half and save it")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = pick_device()
    model, vocab, _ = load_stage1(a.model, device)
    meta = pd.read_parquet(a.meta)
    aug = dict(phrase_variant_p=a.phrase_variant_p, keyword_dropout_p=a.keyword_dropout_p)

    if a.fit_temperature:
        # Fit and evaluate on DISJOINT leg halves, so the correction is not graded on its
        # own data.
        is_eval = split_by_leg(meta, a.seed, val_frac=0.5)
        T = fit_temperature(model, meta[~is_eval], vocab, device, profile=a.profile, **aug)
        print(f"temperature {T:.3f}")
        before = evaluate_posterior(model, meta[is_eval], vocab, device, profile=a.profile,
                                    k=a.k, max_batches=a.max_batches, **aug)
        model.temperature = 1.0
        raw = evaluate_posterior(model, meta[is_eval], vocab, device, profile=a.profile,
                                 k=a.k, max_batches=a.max_batches, **aug)
        model.temperature = T
        print(f"energy  raw {raw['energy']:.4f}  ->  calibrated {before['energy']:.4f}")
        save_stage1(model, vocab, a.model, temperature=T)
        print(f"temperature saved into {a.model}")
        return

    r = evaluate_posterior(model, meta, vocab, device, profile=a.profile, k=a.k,
                           max_batches=a.max_batches, **aug)
    print(pd.Series(r).round(4).to_string())
    print("\nlower joint_nll and energy are better; heading_modes above 1 under noise means "
          "the posterior is multimodal, and heading_R falls as it spreads")


if __name__ == "__main__":
    main()
