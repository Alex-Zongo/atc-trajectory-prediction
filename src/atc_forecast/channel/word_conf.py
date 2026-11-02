"""Word-level confidence fitted to real recognizers.

The channel decides which words are wrong; this module decides what confidence the
recognizer would have printed on each word. Two empirical distributions per recognizer,
from its benchmark hypotheses (data/asr/hypotheses) and the per-utterance error counts
(data/asr/scores/items.jsonl):

  clean   probabilities of every word in utterances with no trajectory-relevant error
  error   the LOWEST word probability of each utterance that has such an error, the best
          available proxy for the probability on the wrong word itself (the benchmark
          labels errors per utterance, not per word)

Stored as 101 quantiles each, so sampling is an inverse-CDF lookup. The fitted file lives
under data/ (gitignored) because it derives from ATCO2, which is for internal research
use; without it, a parametric fallback with the same shape is used and says so.

    python -m atc_forecast.channel.word_conf --out data/v2/word_conf.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

# channel profile -> recognizer whose hypotheses it was fitted from
PROFILE_MODEL = {
    "jacktol": "jacktol_large_v3",
    "jacktol_medium_bench": "jacktol_medium",
    "us_atc_turbo_bench": "us_atc_turbo_v3",
    "whisper_large_v3_bench": "whisper_large_v3",
}
SETS = ("atco2_1h", "atcosim_250")
QS = np.linspace(0, 1, 101)


def fit(asr_dir="data/asr") -> dict:
    asr = Path(asr_dir)
    items = [json.loads(line) for line in
             (asr / "scores" / "items.jsonl").read_text().splitlines()]
    err = {(i["model"], i["set"], i["item_id"]): i["trajectory_errors"] for i in items}
    out = {}
    for model in sorted(set(PROFILE_MODEL.values())):
        clean, low = [], []
        for s in SETS:
            f = asr / "hypotheses" / model / f"{s}.jsonl"
            if not f.exists():
                continue
            for line in f.read_text().splitlines():
                h = json.loads(line)
                e = err.get((model, s, h["item_id"]))
                p = [w["probability"] for w in h.get("words") or []]
                if e is None or not p:
                    continue
                if e == 0:
                    clean += p
                else:
                    low.append(min(p))
        if clean and low:
            out[model] = {"clean": np.quantile(clean, QS).round(4).tolist(),
                          "error": np.quantile(low, QS).round(4).tolist(),
                          "n_clean_words": len(clean), "n_error_utterances": len(low)}
    return out


class WordConf:
    """Sampler of per-word confidences for one recognizer."""

    def __init__(self, profile: str, path="data/v2/word_conf.json"):
        model = PROFILE_MODEL.get(profile, profile)
        p = Path(path)
        d = json.loads(p.read_text()).get(model) if p.exists() else None
        self.fitted = d is not None
        if d is None:
            # Fallback with the shape of the fitted jacktol curves: clean words near 1,
            # wrong words spread over the middle.
            rng = np.random.default_rng(0)
            d = {"clean": np.quantile(rng.beta(30, 1.2, 20000), QS).tolist(),
                 "error": np.quantile(rng.beta(2.5, 2.0, 20000), QS).tolist()}
        self.clean, self.error = np.asarray(d["clean"]), np.asarray(d["error"])

    def sample(self, is_error, rng) -> np.ndarray:
        is_error = np.asarray(is_error, bool)
        u = rng.uniform(0, 1, len(is_error))
        return np.where(is_error, np.interp(u, QS, self.error),
                        np.interp(u, QS, self.clean)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asr", default="data/asr")
    ap.add_argument("--out", default="data/v2/word_conf.json")
    a = ap.parse_args()
    d = fit(a.asr)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(d))
    for m, v in d.items():
        print(f"{m}: {v['n_clean_words']} clean words (median {v['clean'][50]:.3f}), "
              f"{v['n_error_utterances']} error utterances (median min {v['error'][50]:.3f})")


if __name__ == "__main__":
    main()
