"""Train the semantic belief model on precomputed messages.

    python -m atc_forecast.training.sbm_v2 --profile jacktol --factorization anyorder \
        --out runs/v2/sbm_jacktol_anyorder_s0.pt --seed 0

Batches are drawn from the training messages with weights that give the target variant
mix (own 50 %, own-other 20 %, other aircraft 20 %, maintain 10 %). Selection is on the
validation negative log-likelihood of the full interpretation in the canonical order, on
a fixed message subset drawn with the same mix. After training, one temperature per slot
is fitted on validation (teacher-forced canonical conditionals) and stored in the model.
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..models.sbm import SBM, SLOT_NAMES, SBMConfig, save_sbm
from .common import pick_device
from .data_v2 import MessageStore
from .tracking import Tracker, add_args


def ece(prob, correct, bins=15):
    """Expected calibration error of top-1 confidence."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    return float(sum(abs(prob[idx == k].mean() - correct[idx == k].mean()) * (idx == k).mean()
                     for k in range(bins) if (idx == k).any()))


@torch.no_grad()
def evaluate(model, ms, rows, device, n=512):
    """Joint NLL in the canonical order, and per slot the teacher-forced (true earlier
    slots revealed) top-1 accuracy and ECE. Mixture heads report the joint NLL only."""
    model.eval()
    nll, tot = 0.0, 0
    conf, hit = [[] for _ in range(4)], [[] for _ in range(4)]
    for i in range(0, len(rows), n):
        b = ms.batch(rows[i:i + n], device)
        enc = model.encode(b)
        z = b["z"]
        nll += -model.log_prob(b, z, enc).sum().item()
        tot += len(z)
        if model.cfg.factorization == "mixture":
            continue
        for s in range(4):
            rev = torch.zeros_like(z, dtype=torch.bool)
            rev[:, :s] = model.cfg.factorization == "anyorder"
            p = torch.softmax(model.logits(model.slots(enc, rev, z))[s], -1)
            c, k = p.max(-1)
            conf[s].append(c.cpu().numpy())
            hit[s].append((k == z[:, s]).float().cpu().numpy())
    out = {"nll": nll / tot}
    for s, name in enumerate(SLOT_NAMES):
        if conf[s]:
            c, h = np.concatenate(conf[s]), np.concatenate(hit[s])
            out[f"acc_{name}"] = float(h.mean())
            out[f"ece_{name}"] = ece(c, h)
    out["status_acc"] = out.get("acc_status", float("nan"))
    return out


@torch.no_grad()
def teacher_forced_logits(model, ms, rows, device, n=512):
    """Per slot, the canonical-order conditional logits given the true earlier slots."""
    out = [[] for _ in range(4)]
    zs = []
    for i in range(0, len(rows), n):
        b = ms.batch(rows[i:i + n], device)
        enc = model.encode(b)
        z = b["z"]
        for s in range(4):
            rev = torch.zeros_like(z, dtype=torch.bool)
            rev[:, :s] = model.cfg.factorization == "anyorder"
            out[s].append(model.logits(model.slots(enc, rev, z))[s].cpu())
        zs.append(z.cpu())
    return [torch.cat(o) for o in out], torch.cat(zs)


def fit_temperatures(model, ms, rows, device):
    """One temperature per slot minimizing teacher-forced validation NLL. Mixture heads
    have no per-slot conditionals and are left at 1."""
    model.temperature.fill_(1.0)
    if model.cfg.factorization == "mixture":
        return {}
    logits, z = teacher_forced_logits(model, ms, rows, device)
    temps = []
    for s in range(4):
        lt = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([lt], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(logits[s] / lt.exp(), z[:, s])
            loss.backward()
            return loss
        opt.step(closure)
        temps.append(float(lt.detach().exp()))
    model.temperature.copy_(torch.tensor(temps))
    return dict(zip(SLOT_NAMES, temps))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/v2")
    ap.add_argument("--profile", default="jacktol")
    ap.add_argument("--out", required=True)
    ap.add_argument("--factorization", default="anyorder",
                    choices=("anyorder", "independent", "mixture"))
    ap.add_argument("--no-history", action="store_true")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=4000)
    ap.add_argument("--eval-n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    add_args(ap)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = pick_device(a.device)
    ms = MessageStore(a.data, a.profile)
    tr = ms.rows("train")
    w = ms.weights(tr)
    va = ms.rows("val")
    val = np.sort(np.random.default_rng(123).choice(va, min(a.eval_n, len(va)), replace=False,
                                                     p=ms.weights(va)))
    model = SBM(SBMConfig(vocab_size=len(ms.vocab), d=a.d, factorization=a.factorization,
                          use_history=not a.no_history)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5
                                              * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = open(out.with_suffix(".log.jsonl"), "w")
    print(f"SBM {a.factorization}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
          f"parameters on {device}, {len(tr)} train messages", flush=True)
    n_par = sum(p.numel() for p in model.parameters())
    tracker = Tracker(a.wandb, a.wandb_project, a.run_name or out.stem,
                      {**vars(a), "n_params": n_par}, group=a.wandb_group,
                      tags=["sbm", a.factorization, a.profile])
    best, best_state, t0, run = float("inf"), None, time.time(), []
    for step in range(1, a.steps + 1):
        model.train()
        b = ms.batch(rng.choice(tr, a.batch, p=w), device)
        loss, per = model.loss(b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run.append(loss.item())
        if step % 100 == 0:
            tracker.log({"train/loss": np.mean(run[-100:]), "train/lr": sched.get_last_lr()[0],
                         "train/ms_per_step": (time.time() - t0) / step * 1000,
                         **{f"train/ce_{k}": v.item() for k, v in per.items()}}, step)
        if step % 500 == 0:
            print(f"step {step} loss {np.mean(run[-500:]):.4f} "
                  + " ".join(f"{k} {v.item():.3f}" for k, v in per.items())
                  + f" {(time.time() - t0) / step * 1000:.0f} ms/step", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            sc = evaluate(model, ms, val, device)
            rec = {"step": step, "train_loss": float(np.mean(run[-a.eval_every:])),
                   "minutes": (time.time() - t0) / 60, **sc}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            tracker.log({f"val/{k}": v for k, v in sc.items()}, step)
            if sc["nll"] < best:
                best, best_state = sc["nll"], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    temps = fit_temperatures(model, ms, val, device)
    sc = evaluate(model, ms, val, device)
    cert = None
    if a.factorization != "mixture":
        b = ms.batch(val[:1024], device)
        zb, _, c = model.beam(b, width=16)
        cert = {"beam_certified": c.float().mean().item(),
                "mode_exact": (zb[:, 0] == b["z"]).all(1).float().mean().item()}
    print("temperatures", temps, "calibrated val", sc, cert)
    save_sbm(model, out, {"args": vars(a), "val": sc, "temperatures": temps,
                          "vocab": ms.vocab, "beam": cert})
    tracker.summary({f"final/{k}": v for k, v in sc.items()}
                    | {f"final/T_{k}": v for k, v in temps.items()}
                    | {f"final/{k}": v for k, v in (cert or {}).items()})
    tracker.finish()


if __name__ == "__main__":
    main()
