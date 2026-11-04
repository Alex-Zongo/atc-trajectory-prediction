"""Gather every v2 result into one JSON: what the web page's Results sections read.

    python scripts/collect_results_v2.py --runs runs/v2 --results results/v2 --out results/v2/results.json

Collects, when present:
  sbm         per SBM checkpoint: final validation scores, temperatures, beam check, and
              the validation curve from its .log.jsonl
  traj        per trajectory checkpoint (CTP, MDN, anchors): its switches, best validation
              scores and step, and the train/validation curves
  eval        results/v2/eval_test_*.json (composition on the test split)
  sweeps      results/v2/compose_*.json (composition sweeps: K x S, sampler, CTP and
              SBM variants)
  composition results/v2/composition.json (scripts/analyze_composition_v2.py)
  addressee   results/v2/addressee_*.json (evaluation/addressee_v2.py: who the message is
              for, over the aircraft on frequency)
  altitude    results/v2/altitude_noise.json (scripts/altitude_noise_v2.py)
  diagnose    results/v2/diagnose_*.json
Missing pieces are simply absent, so the page can show what is still pending.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch


def finite(x):
    """NaN and infinities become null: browsers reject them in JSON."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite(v) for v in x]
    return x


def curve(log, keys):
    if not log.exists():
        return []
    out = []
    for line in log.read_text().splitlines():
        r = json.loads(line)
        out.append({k: r.get(k) for k in ["step", *keys] if k in r})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/v2")
    ap.add_argument("--results", default="results/v2")
    ap.add_argument("--out", default="results/v2/results.json")
    a = ap.parse_args()
    runs, res = Path(a.runs), Path(a.results)
    out = {"collected": time.strftime("%Y-%m-%d %H:%M"), "sbm": {}, "traj": {}, "eval": {},
           "diagnose": {}}
    for ck in sorted(runs.glob("*.pt")):
        c = torch.load(ck, map_location="cpu", weights_only=False)
        log = ck.with_suffix(".log.jsonl")
        name = ck.stem
        if "factorization" in c.get("cfg", {}):          # an SBM checkpoint
            out["sbm"][name] = {
                "factorization": c["cfg"].get("factorization"),
                "use_history": c["cfg"].get("use_history"),
                "profile": c.get("args", {}).get("profile"), "val": c.get("val"),
                "temperatures": c.get("temperatures"), "beam": c.get("beam"),
                "n_params": sum(v.numel() for v in c["state"].values()),
                "curve": curve(log, ["nll", "status_acc", "train_loss"])}
        else:
            args = c.get("args", {})
            out["traj"][name] = {
                "model": args.get("model"), "output": args.get("output"),
                "condition": args.get("condition"), "time_sampling": args.get("time_sampling"),
                "seed": args.get("seed"), "step": c.get("step"), "val": c.get("val"),
                "curves_by_lead": c.get("curves"),
                "curve": curve(log, ["train_loss", "loss", "energy_h", "energy_v", "ade_h",
                                     "minade6_h", "cov50", "cov90", "select"])}
    for f in sorted(res.glob("eval_*.json")):
        out["eval"][f.stem] = json.loads(f.read_text())
    out["sweeps"] = {f.stem.replace("compose_", ""): json.loads(f.read_text())
                     for f in sorted(res.glob("compose_*.json"))}
    if (res / "composition.json").exists():
        out["composition"] = json.loads((res / "composition.json").read_text())
    out["addressee"] = {f.stem.replace("addressee_", ""): json.loads(f.read_text())
                        for f in sorted(res.glob("addressee_*.json"))}
    if (res / "altitude_noise.json").exists():
        out["altitude_noise"] = json.loads((res / "altitude_noise.json").read_text())
    for f in sorted(res.glob("diagnose_*.json")):
        out["diagnose"][f.stem.replace("diagnose_", "")] = json.loads(f.read_text())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(finite(out), allow_nan=False))
    print(f"collected {len(out['sbm'])} SBM, {len(out['traj'])} trajectory runs, "
          f"{len(out['eval'])} evaluations, {len(out['diagnose'])} diagnoses -> {a.out}")


if __name__ == "__main__":
    main()
