"""Train the Stage-1 clearance posterior.

Checkpoint selection matters once scheduled sampling is on. `val_nll` is teacher-forced,
but the model is deployed free-running, and scheduled sampling deliberately makes the
teacher-forced objective harder, so selecting on `val_nll` would prefer exactly the runs
that did the LEAST of what was turned on. Selecting on `energy` instead uses a proper
scoring rule over the SAMPLED clearances, which is the thing actually deployed. It is only
computed on evaluation epochs, so selection waits for those.

    python -m atc_forecast.training.stage1 --meta data/dataset/meta.parquet \
        --out runs/stage1.pt --epochs 14 --ss-max 0.35 --select-on energy
"""

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.lexicon import build_vocab, load_transcripts
from ..data.text import ClearanceDataset, DataConfig
from ..data.windows import attach_onset_state
from ..models.backbone import EncoderConfig
from ..models.stage1 import ClearancePosterior, save_stage1
from .common import pick_device, split_by_leg


def train_stage1(meta, epochs=14, batch_size=256, lr=3e-4, device=None, seed=0, num_workers=4,
                 aug_factor=1, max_degradation=0.6, eval_every=2, log_cb=None,
                 profile="jacktol", transcripts=None, context_scale=1.0, state_grounded=False,
                 ground_fields=("hdg", "alt", "spd"), phrase_variant_p=0.0,
                 keyword_dropout_p=0.0, cont_head="mix", ss_max=0.0, ss_warmup=0.4,
                 select_on="energy", n_slots=None, out=None):
    from ..evaluation.stage1 import evaluate_posterior

    device = pick_device(device)
    torch.manual_seed(seed)
    is_val = split_by_leg(meta, seed)
    tr_meta, va_meta = meta[~is_val], meta[is_val]
    vocab = build_vocab(meta, transcripts=transcripts)

    dcfg = DataConfig(aug_factor=aug_factor, max_degradation=max_degradation, profile=profile,
                      resample=True, context_scale=context_scale,
                      phrase_variant_p=phrase_variant_p, keyword_dropout_p=keyword_dropout_p)
    dcfg_val = DataConfig(max_degradation=max_degradation, profile=profile,
                          context_scale=context_scale, phrase_variant_p=phrase_variant_p,
                          keyword_dropout_p=keyword_dropout_p)
    tr_ds = ClearanceDataset(tr_meta, vocab, dcfg, seed)
    va_ds = ClearanceDataset(va_meta, vocab, dcfg_val, seed + 1)
    kw = dict(num_workers=num_workers, persistent_workers=num_workers > 0,
              pin_memory=(device == "cuda"), prefetch_factor=4 if num_workers > 0 else None)
    tl = DataLoader(tr_ds, batch_size, shuffle=True, drop_last=True, **kw)
    vl = DataLoader(va_ds, batch_size, shuffle=False, drop_last=True, **kw)

    cfg = EncoderConfig(vocab_size=len(vocab), max_len=dcfg.max_len)
    state_dim = tr_ds.state_dim if state_grounded else 0
    model = ClearancePosterior(cfg, state_dim=state_dim, ground_fields=ground_fields,
                               cont_head=cont_head, n_slots=n_slots).to(device)
    print(f"stage1: train {len(tr_meta)} | val {len(va_meta)} | vocab {len(vocab)} | {device}")
    print(f"  params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | "
          f"cont_head={cont_head} | slots={model.n_slots} | state_dim={state_dim}")
    if state_grounded and state_dim == 0:
        raise SystemExit(
            "--state-grounded was asked for but meta carries no 'alt_onset_ft' column, so "
            "training would silently run UNGROUNDED and the ablation would be meaningless. "
            "Pass a meta built by data.windows.attach_onset_state (build_dataset.py writes "
            "meta_grounded.parquet), and check the column was not suffixed by a double join.")
    if ss_max > 0:
        print(f"  scheduled sampling: p ramps 0 -> {ss_max:g} over the first {ss_warmup:.0%}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs * len(tl))
    best, best_ep, best_state = float("inf"), 0, None

    for ep in range(epochs):
        # Linear ramp then hold: early epochs need clean teacher forcing to learn the fields
        # at all, and feeding the model its own noise from step one only slows convergence.
        ss_p = 0.0 if ss_max <= 0 else ss_max * min(1.0, ep / max(1.0, ss_warmup * epochs))
        model.train()
        tot = nb = 0
        comp = {}
        for batch in tqdm(tl, desc=f"ep {ep + 1}/{epochs}", leave=False, dynamic_ncols=True):
            batch = {k: v.to(device) for k, v in batch.items()}
            L = model.loss(batch, ss_p=ss_p)
            opt.zero_grad(set_to_none=True)
            L["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += L["loss"].item()
            nb += 1
            for k in ("clearance", "heading", "altitude", "presence"):
                comp[k] = comp.get(k, 0.0) + L[k].item()

        model.eval()
        vt = vnb = 0
        with torch.no_grad():
            for batch in vl:
                batch = {k: v.to(device) for k, v in batch.items()}
                vt += model.loss(batch)["loss"].item()      # ss_p=0, comparable across epochs
                vnb += 1
        metrics = {"train_nll": tot / nb, "val_nll": vt / max(vnb, 1), "ss_p": ss_p,
                   "lr": sched.get_last_lr()[0],
                   **{f"nll_{k}": v / nb for k, v in comp.items()}}

        if eval_every and (ep + 1) % eval_every == 0:
            # Evaluate on the channel the model is TRAINED on, with the SAME phraseology
            # augmentation. Evaluating on canonical text a specialised model never saw makes
            # the energy score rise monotonically while val_nll falls, which is easily
            # mistaken for the posterior collapsing.
            ev = evaluate_posterior(model, va_meta, vocab, device, profile=profile, k=16,
                                    batch_size=batch_size, max_batches=6,
                                    phrase_variant_p=phrase_variant_p,
                                    keyword_dropout_p=keyword_dropout_p)
            metrics.update(ev)

        sel = metrics.get(select_on)
        improved = sel is not None and sel < best
        if improved:
            best, best_ep = sel, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if out:                                          # crash-safe: write each new best
                save_stage1(model, vocab, out, ss_max=ss_max, profile=profile,
                            selected_on=select_on, selected_value=best, epoch=best_ep)
        metrics[f"best_{select_on}"] = best
        print(f"ep{ep + 1} train={metrics['train_nll']:.3f} val={metrics['val_nll']:.3f}"
              + (f" energy={metrics['energy']:.4f}"
                 f" (acc {metrics['energy_accuracy']:.3f}/disp {metrics['energy_dispersion']:.3f})"
                 if "energy" in metrics else "")
              + ("  *best*" if improved else ""))
        if log_cb:
            log_cb(ep + 1, metrics)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"restored epoch {best_ep} ({select_on}={best:.4f})")
    elif out:
        print(f"WARNING: {select_on} was never computed, keeping the FINAL epoch. "
              f"Check that eval_every divides into epochs.")
        save_stage1(model, vocab, out, ss_max=ss_max, profile=profile)
    return model, vocab


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True, help="meta.parquet from the dataset build")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default="jacktol", help="observation channel to train on")
    ap.add_argument("--aug-factor", type=int, default=1,
                    help="independent corruptions per sample per epoch")
    ap.add_argument("--max-degradation", type=float, default=0.6)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--cont-head", default="mix", choices=("mix", "cat"),
                    help="alt/speed head: Gaussian mixture, or a categorical physical grid")
    ap.add_argument("--n-slots", type=int, default=None)
    ap.add_argument("--phrase-variant-p", type=float, default=0.35,
                    help="probability of a non-standard but real phrasing per clause")
    ap.add_argument("--keyword-dropout-p", type=float, default=0.25,
                    help="probability of dropping the scale/slot keyword, leaving bare digits")
    ap.add_argument("--context-scale", type=float, default=1.0)
    ap.add_argument("--ss-max", type=float, default=0.35,
                    help="peak scheduled-sampling probability; 0 is pure teacher forcing")
    ap.add_argument("--ss-warmup", type=float, default=0.4)
    ap.add_argument("--select-on", default="energy",
                    help="checkpoint metric, lower better: energy (sampled posterior) or val_nll")
    ap.add_argument("--state-grounded", action="store_true",
                    help="ground the numeric heads in the aircraft state at clearance onset")
    ap.add_argument("--window-dir", default=None,
                    help="Stage-2 windows to source the onset state from")
    ap.add_argument("--ground-fields", default="hdg,alt,spd")
    ap.add_argument("--transcripts-glob", default=None,
                    help="ASR record JSONL glob, harvested into the vocabulary")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="atc-stage1")
    a = ap.parse_args()

    meta = pd.read_parquet(a.meta)
    if a.state_grounded:
        if not a.window_dir:
            raise SystemExit("--state-grounded needs --window-dir")
        meta = attach_onset_state(meta, a.window_dir)

    run = None
    if a.wandb:
        import wandb
        run = wandb.init(project=a.wandb_project, config=vars(a))

    train_stage1(meta, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, seed=a.seed,
                 num_workers=a.workers, aug_factor=a.aug_factor,
                 max_degradation=a.max_degradation, eval_every=a.eval_every,
                 profile=a.profile, cont_head=a.cont_head, n_slots=a.n_slots,
                 phrase_variant_p=a.phrase_variant_p, keyword_dropout_p=a.keyword_dropout_p,
                 context_scale=a.context_scale, ss_max=a.ss_max, ss_warmup=a.ss_warmup,
                 select_on=a.select_on, state_grounded=a.state_grounded,
                 ground_fields=tuple(a.ground_fields.split(",")),
                 transcripts=load_transcripts(a.transcripts_glob) if a.transcripts_glob else None,
                 log_cb=(lambda ep, m: run.log(m, step=ep)) if run else None,
                 out=a.out)
    if run:
        run.finish()
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
