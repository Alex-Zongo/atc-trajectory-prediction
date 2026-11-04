"""Optional Weights & Biases logging for the v2 trainers and evaluators.

Off unless --wandb is passed, so runs never depend on a network or a login. With it,
every scalar the trainer prints also goes to W&B under a stable key layout:

    train/*   loss, per-slot cross-entropy, gradient norm, learning rate, ms/step
    val/*     the selection metrics at each evaluation
    cal/*     per-lead-time coverage and spread-skill curves (line plots)
    final/*   the selected checkpoint's scores (run summary)

Set WANDB_MODE=offline to log locally and `wandb sync` later.
"""

import os


class Tracker:
    def __init__(self, enabled, project="atc-v2", name=None, config=None, group=None,
                 tags=None, job_type="train"):
        self.run = None
        if not enabled:
            return
        import wandb
        self.wandb = wandb
        self.run = wandb.init(project=project, name=name, config=config, group=group,
                              tags=tags, job_type=job_type, dir=os.environ.get("WANDB_DIR"),
                              mode=os.environ.get("WANDB_MODE", "online"))

    def log(self, d, step=None):
        if self.run:
            self.run.log(d, step=step)

    def curve(self, key, ys, xs=None, names=None, title=None, step=None):
        """Per-lead-time curves: ys is a list of series (or one series)."""
        if not self.run:
            return
        if ys and not isinstance(ys[0], (list, tuple)):
            ys = [ys]
        xs = xs or list(range(1, len(ys[0]) + 1))
        self.run.log({key: self.wandb.plot.line_series(
            xs=xs, ys=[list(map(float, y)) for y in ys], keys=names or [key],
            title=title or key, xname="lead time (s)")}, step=step)

    def summary(self, d):
        if self.run:
            for k, v in d.items():
                self.run.summary[k] = v

    def finish(self):
        if self.run:
            self.run.finish()


def add_args(ap):
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    ap.add_argument("--wandb-project", default="atc-v2")
    ap.add_argument("--wandb-group", default=None, help="e.g. core, baselines, seeds")
    ap.add_argument("--run-name", default=None, help="defaults to the checkpoint stem")
