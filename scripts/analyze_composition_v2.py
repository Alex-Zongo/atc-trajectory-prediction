"""Where does composition win or lose? Analysis of compose_v2 --per-message records.

    python scripts/analyze_composition_v2.py --pm results/v2/eval_test_s0.npz \
        --out results/v2/composition.json

  by_confidence   messages binned by the SBM's mode probability: per bin the share of
                  correct modes and the mean horizontal energy score of each condition.
                  Composition should matter where the SBM is unsure; where it is sure,
                  composed and mode should coincide.
  by_correct      the same split by whether the SBM mode equals the true Z
  paired          per message differences between conditions (energy, horizontal), with
                  95% bootstrap intervals over messages, per message variant
  distinct        how many distinct interpretations the K draws contain
"""

import argparse
import json
from pathlib import Path

import numpy as np

BINS = [0.0, 0.5, 0.8, 0.95, 0.99, 1.0001]
PAIRS = [("composed", "mode"), ("composed", "oracle"), ("composed", "hist_only"),
         ("composed", "direct"), ("composed", "direct_init"), ("mdn_composed", "composed"),
         ("anchors_composed", "composed"), ("mdn_composed", "mdn_oracle")]


def boot(d, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return [float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pm", default="results/v2/eval_test_s0.npz")
    ap.add_argument("--out", default="results/v2/composition.json")
    a = ap.parse_args()
    d = dict(np.load(a.pm))
    conds = sorted({k.split("/")[0] for k in d if k.endswith("/energy_h")})
    var = d["variant"].astype(str)
    p_mode = np.exp(d["mode_logp"])
    out = {"conditions": conds, "bins": BINS[:-1] + [1.0], "by_confidence": {},
           "by_correct": {}, "paired": {}, "distinct": {}}
    for v in ("own", "own_other", "other", "maintain"):
        s = var == v
        if not s.any():
            continue
        rows = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = s & (p_mode >= lo) & (p_mode < hi)
            rows.append({"lo": lo, "hi": min(hi, 1.0), "n": int(m.sum()),
                         "mode_right": float(d["mode_exact"][m].mean()) if m.any() else None,
                         **{c: float(d[f"{c}/energy_h"][m].mean()) if m.any() else None
                            for c in conds}})
        out["by_confidence"][v] = rows
        out["by_correct"][v] = {
            ok: {"n": int((s & (d["mode_exact"] == (ok == "right"))).sum()),
                 **{c: float(d[f"{c}/energy_h"][s & (d["mode_exact"] == (ok == "right"))].mean())
                    for c in conds if (s & (d["mode_exact"] == (ok == "right"))).any()}}
            for ok in ("right", "wrong")}
        out["paired"][v] = {f"{x} - {y}": boot(d[f"{x}/energy_h"][s] - d[f"{y}/energy_h"][s])
                            for x, y in PAIRS if f"{x}/energy_h" in d and f"{y}/energy_h" in d}
        nd = d["n_distinct"][s]
        out["distinct"][v] = {str(k): int((nd == k).sum()) for k in np.unique(nd)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["paired"], indent=1))


if __name__ == "__main__":
    main()
