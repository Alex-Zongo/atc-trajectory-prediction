"""Train the Stage-2 flow predictor.

Two ablations are logged at every evaluation, so their value is visible DURING training
rather than only afterwards:

    clearance_gain_nm  ADE without the clearance minus ADE with it; above 0 means the
                       instruction helps
    aircraft_gain_nm   ADE with the airframe forced to "unknown" minus ADE with it known;
                       this doubles as the graceful-degradation cost

The clearance gain is also reported stratified by whether the clearance carries any clause,
because the aggregate is diluted by steady windows where the past trivially determines the
future.

    python -m atc_forecast.training.stage2 --data data/windows_raw --out runs/stage2.pt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..models.stage2 import (FlowPredictor, WindowDataset, decimate_past, save_stage2)
from .common import pick_device, split_by_leg


def train_stage2(data_dir, epochs=30, batch_size=256, lr=3e-4, device=None, seed=0,
                 num_workers=0, log_cb=None, eval_every=4, cfg_drop_p=0.1, cov_k=12,
                 eval_k=5, decimate=1.0, source="gauss", src_sigma=1.0,
                 step_weight="integrated", pos_loss_w=0.0, complete_only=False,
                 cfg_drop_end=None, out=None):
    from ..evaluation.stage2 import coverage_eval, point_eval

    device = pick_device(device)
    torch.manual_seed(seed)
    npz = np.load(Path(data_dir) / "windows.npz")
    meta = pd.read_parquet(Path(data_dir) / "windows_meta.parquet")
    is_val = split_by_leg(meta, seed)

    keep = np.ones(len(meta), bool)
    if complete_only:
        # The leak-free CONTROL: every retained window has all H future steps observed, so
        # there is nothing to mask and nothing to noise-fill. Beware the selection bias.
        # At H=60 this drops turns from about 51 to 33 percent of windows and speed clauses
        # from 25 to 5, because a turn or a speed change is usually superseded quickly and
        # that is exactly what truncates the window. Only defensible at a short horizon.
        keep = npz["fut_mask"].astype(bool).all(1)
        print(f"complete-only: {keep.sum():,}/{len(keep):,} ({keep.mean():.1%}) windows; "
              f"check the population shift before reporting this as comparable")
    tr_idx = np.where(~is_val & keep)[0]
    va_idx = np.where(is_val & keep)[0]

    dl = torch.utils.data.DataLoader
    tl = dl(WindowDataset(npz, meta, tr_idx), batch_size, shuffle=True, drop_last=True,
            num_workers=num_workers)
    vl = dl(WindowDataset(npz, meta, va_idx), batch_size, shuffle=False, drop_last=True,
            num_workers=num_workers)

    H, P, NF = npz["fut_ctrl"].shape[1], npz["past"].shape[1], npz["past"].shape[2]
    time_enc = "past_time" in npz
    model = FlowPredictor(source=source, src_sigma=src_sigma, step_weight=step_weight,
                          pos_loss_w=pos_loss_w, cfg_drop_p=cfg_drop_p,
                          cfg_drop_end=cfg_drop_end, h_steps=H, p_steps=P, n_past_f=NF,
                          time_encoding=time_enc).to(device)
    past_kind = f"raw <= {P} tokens (continuous time)" if time_enc else f"grid {P}s x {NF}f"
    print(f"stage2: train {len(tr_idx)} | val {len(va_idx)} | {device}")
    print(f"  horizon H={H}s | past {past_kind} | cfg_drop_p={cfg_drop_p}"
          + (f" | decimate keep={decimate}" if time_enc and decimate < 1.0 else ""))
    print(f"  params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    if model.cfg_drop_end != model.cfg_drop_start:
        print(f"  CFG dropout schedule {model.cfg_drop_start:g} -> {model.cfg_drop_end:g}; "
              f"this changes how well trained the no-instruction baseline is, which every "
              f"reported gain is measured against")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs * len(tl))

    for ep in range(epochs):
        if model.cfg_drop_end != model.cfg_drop_start:
            f = ep / max(1, epochs - 1)
            model.cfg_drop_p = (model.cfg_drop_start
                                + f * (model.cfg_drop_end - model.cfg_drop_start))
        model.train()
        tot = nb = 0
        gsum = 0.0
        pbar = tqdm(tl, desc=f"ep {ep + 1}/{epochs}", leave=False, dynamic_ncols=True)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            if time_enc and decimate < 1.0:
                batch = decimate_past(batch, decimate)
            loss = model.flow_loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
            gsum += float(gnorm)
            pbar.set_postfix(loss=f"{tot / nb:.4f}", gnorm=f"{gsum / nb:.2f}")

        # ODE sampling is the slow part, so the sampled metrics run only every eval_every
        # epochs and on the last. The flow loss is logged every epoch.
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            ev = point_eval(model, vl, device, k=eval_k)
            ev.update(coverage_eval(model, vl, device, k=cov_k))
            ev["grad_norm"] = gsum / nb
            print(f"ep{ep + 1} loss={tot / nb:.4f} | minADE={ev['minADE_nm']:.2f}nm "
                  f"(clearance {ev['clearance_gain_nm']:+.2f}nm/"
                  f"{ev['clearance_gain_vert_ft']:+.0f}ft, aircraft "
                  f"{ev['aircraft_gain_nm']:+.2f}nm) | avgADE={ev['avgADE_nm']:.2f}nm | "
                  f"energy-gain={ev['energy_gain']:+.4f} kde-gain={ev['kde_gain']:+.2f}")
            if log_cb:
                log_cb(ep + 1, tot / nb, ev, sched.get_last_lr()[0])
        else:
            print(f"ep{ep + 1} loss={tot / nb:.4f}")
            if log_cb:
                log_cb(ep + 1, tot / nb, None, sched.get_last_lr()[0])

    if out:
        save_stage2(model, out, epochs=epochs, seed=seed, data_dir=str(data_dir))
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="window directory from build_windows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--eval-k", type=int, default=5)
    ap.add_argument("--cov-k", type=int, default=12)
    ap.add_argument("--cfg-drop-p", type=float, default=0.1)
    ap.add_argument("--cfg-drop-end", type=float, default=None,
                    help="anneal cfg_drop_p linearly to this; a deliberate ablation only")
    ap.add_argument("--source", default="gauss", choices=("gauss", "dr"),
                    help="flow source: standard Gaussian, or centred on dead reckoning")
    ap.add_argument("--src-sigma", type=float, default=1.0)
    ap.add_argument("--step-weight", default="integrated", choices=("integrated", "uniform"))
    ap.add_argument("--pos-loss-w", type=float, default=0.0,
                    help="weight on the auxiliary dead-reckoned position term")
    ap.add_argument("--decimate", type=float, default=1.0,
                    help="keep probability for past-token decimation augmentation")
    ap.add_argument("--complete-only", action="store_true",
                    help="train only on fully observed windows; shifts the population")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="atc-stage2")
    a = ap.parse_args()

    run = None
    if a.wandb:
        import wandb
        run = wandb.init(project=a.wandb_project, config=vars(a))

    def log_cb(ep, loss, ev, lr):
        if run:
            run.log({"flow_loss": loss, "lr": lr, **(ev or {})}, step=ep)

    train_stage2(a.data, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, seed=a.seed,
                 num_workers=a.workers, eval_every=a.eval_every, eval_k=a.eval_k,
                 cov_k=a.cov_k, cfg_drop_p=a.cfg_drop_p, cfg_drop_end=a.cfg_drop_end,
                 source=a.source, src_sigma=a.src_sigma, step_weight=a.step_weight,
                 pos_loss_w=a.pos_loss_w, decimate=a.decimate,
                 complete_only=a.complete_only, log_cb=log_cb if run else None, out=a.out)
    if run:
        run.finish()
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
