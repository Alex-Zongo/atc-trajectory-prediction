"""Train a v2 trajectory model: the CTP or one of its baselines, all on one budget.

    python -m atc_forecast.training.traj_v2 --model ctp --output basis --condition atoms \
        --out runs/v2/ctp_basis_s0.pt --seed 0

    --model ctp      rectified flow (models/ctp.py); --output basis|steps;
                     --condition atoms|none|text (text needs --profile, optional --init-sbm)
    --model mdn      mixture density network on the basis coefficients
    --model anchors  MDN with MultiPath-style fixed anchors (k-means of training coefficients)
    --model prim     design A: physical-kind head + flow over maneuver primitives
    --model kindctp  the same kind head, flow on the PCA weights (ablation); --kinds joint|axis
                     (models/prim.py; needs data/v2/primitives.npz from build_primitives_v2.py)

Every model gets the same fixed number of optimizer steps (default 60k, batch 256),
AdamW with warm-up and cosine decay, gradient clipping, and an EMA of the weights that is
what gets evaluated and saved. Selection is on validation energy score (horizontal + the
vertical in metres) on a fixed subset with seeded noise, every --eval-every steps. A JSON
line per evaluation goes to <out>.log.jsonl, so train/validation gaps are visible.
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ..evaluation.metrics_v2 import coverage, positions_from_velocities, spread_skill, summarize
from ..models.baselines import MDN, MDNConfig, kmeans_anchors, save_mdn
from ..models.ctp import CTP, CTPConfig, save_ctp, seeded_noise
from .common import pick_device
from .data_v2 import MessageStore, WindowStore, ZPolicy
from .tracking import Tracker, add_args


def build_model(a, ws: WindowStore, vocab_size=0):
    if a.model in ("prim", "kindctp"):            # design A, or the kind-conditioned CTP
        from ..models.prim import PrimCTP, load_prim_targets
        cfg = CTPConfig(output="basis", condition=a.condition, vocab_size=vocab_size,
                        time_sampling=a.time_sampling, d=a.d, dec_layers=a.dec_layers,
                        kinds=a.kinds, target="prim" if a.model == "prim" else "pca",
                        kind_sample=a.kind_sample, kind_cond=not a.kind_readout)
        return PrimCTP(cfg, ws.basis, load_prim_targets(a.data, a.prim_file))
    if a.model == "ctp":
        cfg = CTPConfig(output=a.output, condition=a.condition, vocab_size=vocab_size,
                        time_sampling=a.time_sampling, d=a.d, dec_layers=a.dec_layers)
        return CTP(cfg, ws.basis)
    anchors = None
    if a.model == "anchors":
        tr = ws.split("train")
        sel = np.random.default_rng(a.seed).choice(tr, min(50000, len(tr)), False)
        X = ws.coefficients(sel) / ws.basis["coef_std"]
        anchors = kmeans_anchors(X, a.n_comp, seed=a.seed)
    cfg = MDNConfig(d=a.d, n_comp=a.n_comp, condition=a.condition, anchors=a.model == "anchors")
    return MDN(cfg, ws.basis, anchors)


class Batcher:
    """Training and validation batches for the chosen conditioning."""

    def __init__(self, a, ws: WindowStore, device):
        self.a, self.ws, self.device = a, ws, device
        self.rng = np.random.default_rng(a.seed)
        self.policy = ZPolicy(p_null=a.p_null, p_maintain=a.p_maintain)
        self.train = ws.split("train")
        val = ws.split("val")
        self.val = np.sort(np.random.default_rng(123).choice(val, min(a.eval_n, len(val)),
                                                              replace=False))
        if a.condition == "text":
            self.ms = MessageStore(a.data, a.profile, ws)
            r = self.ms.rows("train")
            self.msg_rows, self.msg_w = r, self.ms.weights(r)
            # validation: one fixed message per window, its first own (or other) message
            vm = self.ms.msgs.iloc[self.ms.rows("val")]
            vm = vm[vm["window_id"].isin(self.val)]
            pref = vm.assign(o=vm["variant"].map({"own": 0, "maintain": 1, "own_other": 2,
                                                  "other": 3}))
            first = pref.sort_values(["window_id", "o", "draw"]).groupby("window_id").head(1)
            self.val_msgs = first.index.to_numpy()

    def train_batch(self, n):
        if self.a.condition == "text":
            rows = self.rng.choice(self.msg_rows, n, p=self.msg_w)
            return self.ms.batch(rows, self.device)
        idx = self.rng.choice(self.train, n)
        z = self.policy.draw(self.ws, np.sort(idx), self.rng)
        return self.ws.batch(idx, self.device, z)[0]

    def val_batches(self, n=256):
        if self.a.condition == "text":
            for i in range(0, len(self.val_msgs), n):
                yield self.ms.batch(self.val_msgs[i:i + n], self.device)
            return
        for i in range(0, len(self.val), n):
            idx = self.val[i:i + n]
            z = self.ws.z_true[idx].copy()
            steady = ~self.ws.is_man[idx]
            z[steady, 0] = 1                       # no motion message on steady windows
            yield self.ws.batch(idx, self.device, z)[0]


@torch.no_grad()
def evaluate(model, batcher, a):
    """Validation scores, the training loss on validation windows (fixed noise, so the
    train/val gap is comparable across evaluations), and per-lead calibration curves."""
    model.eval()
    scores, cov, ss, vloss = [], [], [], []
    g = torch.get_rng_state()
    torch.manual_seed(4321)
    for b in batcher.val_batches():
        wid = b["window_id"]
        vloss.append((len(wid), model.loss(b).item()))
        if isinstance(model, CTP):
            noise = seeded_noise(wid, a.eval_samples, model.space.shape, run_seed=1234)
            V = model.sample(b, a.eval_samples, noise=noise, nfe=a.nfe)
        else:
            gens = [torch.Generator().manual_seed(int(w) + 1234) for w in np.asarray(wid)]
            V = model.sample(b, a.eval_samples, generators=gens)
        X = positions_from_velocities(V.cpu().numpy())
        y = positions_from_velocities(b["fut_vel"].cpu().numpy())
        scores.append((len(wid), summarize(X, y)))
        cov.append((len(wid), coverage(X, y)))
        ss.append((len(wid), spread_skill(X, y)))
    torch.set_rng_state(g)
    n = sum(k for k, _ in scores)
    out = {key: sum(k * s[key] for k, s in scores) / n for key in scores[0][1]}
    out["select"] = out["energy_h"] + out["energy_v"] / 3.28084
    out["loss"] = sum(k * v for k, v in vloss) / n
    curves = {"cov": sum(k * c for k, c in cov) / n,             # [2, 60, 3]
              "spread_skill": sum(k * c for k, c in ss) / n}     # [60, 3]
    return out, curves


def log_curves(tracker, curves, step):
    names = ["along", "right", "vertical"]
    for lv, c in zip((50, 90), curves["cov"]):
        tracker.curve(f"cal/coverage{lv}", [c[:, i].tolist() for i in range(3)],
                      names=names, title=f"{lv}% interval coverage by lead", step=step)
    tracker.curve("cal/spread_skill", [curves["spread_skill"][:, i].tolist() for i in range(3)],
                  names=names, title="spread / skill by lead", step=step)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="ctp", choices=("ctp", "mdn", "anchors", "prim", "kindctp"))
    ap.add_argument("--output", default="basis", choices=("basis", "steps"))
    ap.add_argument("--kinds", default="joint", choices=("joint", "axis"),
                    help="prim/kindctp: one head over the joint physical kinds, or one per axis")
    ap.add_argument("--kind-sample", type=float, default=0.0,
                    help="prim/kindctp: share of rows whose flow is conditioned on a kind drawn "
                         "from the head instead of the true kind (scheduled sampling)")
    ap.add_argument("--kind-readout", action="store_true",
                    help="prim/kindctp: train the kind head as an output only, not fed to the flow")
    ap.add_argument("--prim-file", default="primitives.npz",
                    help="prim/kindctp: design-A targets in --data (build_primitives_v2.py --out)")
    ap.add_argument("--basis", default="pca", choices=("pca", "sine"),
                    help="shapes for --output basis: the PCA of basis.npz, or quarter-wave sines "
                         "with the same count per axis (the Fourier ablation)")
    ap.add_argument("--condition", default="atoms", choices=("atoms", "none", "text"))
    ap.add_argument("--profile", default="jacktol")
    ap.add_argument("--init-sbm", default=None, help="SBM checkpoint to initialise the "
                    "text encoder from (direct transcript CTP)")
    ap.add_argument("--time-sampling", default="logit_normal",
                    choices=("logit_normal", "uniform"))
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--dec-layers", type=int, default=4)
    ap.add_argument("--n-comp", type=int, default=8)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--p-null", type=float, default=0.2)
    ap.add_argument("--p-maintain", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--eval-n", type=int, default=5000)
    ap.add_argument("--eval-samples", type=int, default=32)
    ap.add_argument("--nfe", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    add_args(ap)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = pick_device(a.device)
    ws = WindowStore(a.data, a.prim_file)
    ws.swap_basis(a.basis)
    batcher = Batcher(a, ws, device)
    vocab = len(batcher.ms.vocab) if a.condition == "text" else 0
    model = build_model(a, ws, vocab).to(device)
    if a.init_sbm:
        from ..models.sbm import load_sbm
        sbm, _ = load_sbm(a.init_sbm)
        model.text.load_state_dict(sbm.text.state_dict())
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5
                                              * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = open(out.with_suffix(".log.jsonl"), "w")
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{a.model}/{a.output}/{a.condition}: {n_par / 1e6:.2f}M parameters on {device}, "
          f"{len(batcher.train)} train windows", flush=True)
    tracker = Tracker(a.wandb, a.wandb_project, a.run_name or out.stem,
                      {**vars(a), "n_params": n_par}, group=a.wandb_group,
                      tags=[a.model, a.output, a.condition, a.basis])

    best, t0, run = float("inf"), time.time(), []
    for step in range(1, a.steps + 1):
        model.train()
        loss = model.loss(batcher.train_batch(a.batch))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - a.ema)
        run.append(loss.item())
        if step % 100 == 0:
            tracker.log({"train/loss": np.mean(run[-100:]), "train/grad_norm": float(gn),
                         "train/lr": sched.get_last_lr()[0],
                         "train/ms_per_step": (time.time() - t0) / step * 1000}, step)
        if step % 500 == 0:
            print(f"step {step} loss {np.mean(run[-500:]):.4f} |g| {gn:.2f} "
                  f"{(time.time() - t0) / step * 1000:.0f} ms/step", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            sc, curves = evaluate(ema, batcher, a)
            tracker.log({f"val/{k}": v for k, v in sc.items()}, step)
            log_curves(tracker, curves, step)
            rec = {"step": step, "train_loss": float(np.mean(run[-a.eval_every:])),
                   "minutes": (time.time() - t0) / 60, **sc}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps({k: round(v, 3) if isinstance(v, float) else v
                              for k, v in rec.items()}), flush=True)
            if sc["select"] < best:
                best = sc["select"]
                extra = {"args": vars(a), "val": sc, "step": step,
                         "curves": {k: v.tolist() for k, v in curves.items()}}
                (save_ctp if isinstance(ema, CTP) else save_mdn)(ema, out, extra)
                tracker.summary({f"final/{k}": v for k, v in sc.items()} | {"final/step": step})
    print(f"best select {best:.2f} -> {out}")
    tracker.finish()


if __name__ == "__main__":
    main()
