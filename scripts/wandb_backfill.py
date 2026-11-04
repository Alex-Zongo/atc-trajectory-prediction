"""Push the JSON logs of runs trained without --wandb to Weights & Biases.

    python scripts/wandb_backfill.py runs/v2/*.log.jsonl --group core

Each <run>.log.jsonl (written by the v2 trainers at every evaluation) becomes one W&B run
named after the checkpoint, with its val/* metrics at the logged steps and the
checkpoint's arguments as config. Training-loss curves at finer resolution exist only for
runs started with --wandb.
"""

import argparse
import json
from pathlib import Path

import torch
import wandb


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--project", default="atc-v2")
    ap.add_argument("--group", default=None)
    a = ap.parse_args()
    for p in map(Path, a.logs):
        name = p.name.replace(".log.jsonl", "")
        ck = p.with_name(name + ".pt")
        cfg = {}
        if ck.exists():
            c = torch.load(ck, map_location="cpu", weights_only=False)
            cfg = {**c.get("args", {}), **{f"model/{k}": v for k, v in c.get("cfg", {}).items()}}
        run = wandb.init(project=a.project, name=name, group=a.group, config=cfg,
                         tags=["backfill"], job_type="train")
        for line in p.read_text().splitlines():
            rec = json.loads(line)
            step = rec.pop("step")
            run.log({"train/loss_window": rec.pop("train_loss", None),
                     **{f"val/{k}": v for k, v in rec.items()}}, step=step)
        run.finish()
        print(f"{name}: pushed")


if __name__ == "__main__":
    main()
