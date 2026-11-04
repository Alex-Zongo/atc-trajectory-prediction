"""Fit an observation channel from the labelled recognizer benchmark, not from regexes.

`fit.py` parses reference and hypothesis text with regular expressions. On real ATC that
produced two artefacts: climb/descend "swaps" that were parser misreads, and a 0.51 speed
dropout. This fitter reads the per-recording outcomes written by
`atc_forecast.asr.labelling.score` instead, where an LLM extracted each instruction element
blind and code compared them. Every rate is a count over reference values, stored with its
count and a 95% Wilson interval, so a profile says how much evidence stands behind it.

What it adds to a profile, beyond `fit.py`:

  value_error     the measured PER-VALUE wrong rate. The channel converts it to a per-digit
                  rate for the digits of each value, so the simulated wrong-value rate
                  matches the measured one (`digit_error` is per digit).
  direction.omit  the direction word is lost but the value survives ("heading 270" for
                  "turn left heading 270"). Measured directions are dropped, almost never
                  swapped.
  keyword_dropout the value survives without its instruction word ("three four five").
  insertion       per transmission, an instruction element nobody said: a value with its
                  verb, or a bare direction. Real recognizers do this; the regex channel
                  could not.
  confidence      `conditional`: separate empirical distributions for recordings with and
                  without a trajectory error, so confidence carries per-utterance evidence.

Limits, recorded in each profile. A benchmark recording can hold an instruction and its
readback, while the channel corrupts one clearance; insertion rates are per recording. A
field with fewer than 30 reference values (speed, everywhere) takes the pooled heading and
altitude rates as a prior and is listed under `provisional`. jacktol's models trained on
most ATCO2 clips (atc_forecast.asr.overlap), so a jacktol profile fitted on them is an
optimistic bound.

    python -m atc_forecast.channel.fit_labels --model whisper_large_v3 --name whisper_large_v3_bench --write
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .fit import confidence_model
from .model import PROFILES_DIR

OUTCOMES = ("correct", "wrong", "value_missing", "kind_lost", "dropped")
# channel field -> benchmark kinds pooled into it. A relative turn is not a heading, so it
# only counts towards insertions, which is where it changes what the clearance says.
FIELDS = {"heading": ("heading",), "altitude": ("altitude", "vertical_rate"), "speed": ("speed",)}
MIN_N = 30


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(centre - half, 0.0), 4), round(min(centre + half, 1.0), 4)]


def _rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def fit(items: list[dict], name: str, model: str, sets: list[str]) -> dict:
    c = Counter()
    for it in items:
        c.update(it["counts"])
    n_rec = len(items)

    measured, fields = {}, {}
    for f, kinds in FIELDS.items():
        k = {o: sum(c.get(f"{kind}:{o}", 0) for kind in kinds) for o in OUTCOMES}
        n = sum(k.values())
        measured[f] = {o: _rate(k[o], n) for o in OUTCOMES if o != "correct"}
        fields[f] = {"dropout": k["dropped"] / n if n else 0.0,
                     "number_dropout": k["value_missing"] / n if n else 0.0,
                     "value_error": k["wrong"] / n if n else 0.0}
    provisional = [f for f in FIELDS if measured[f]["dropped"]["n"] < MIN_N]
    solid = [f for f in FIELDS if f not in provisional]
    for f in provisional:                   # prior: pooled rates of the well-measured fields
        fields[f] = {key: float(np.mean([fields[g][key] for g in solid])) if solid else 0.0
                     for key in fields[f]}

    d = {o: c.get(f"direction:{o}", 0) for o in ("correct", "swapped", "dropped", "inserted")}
    n_dir = d["correct"] + d["swapped"] + d["dropped"]
    measured["direction"] = {"swapped": _rate(d["swapped"], n_dir),
                             "dropped": _rate(d["dropped"], n_dir)}
    fields["direction"] = {"dropout": 0.0, "flip": d["swapped"] / n_dir if n_dir else 0.0,
                           "omit": d["dropped"] / n_dir if n_dir else 0.0}

    n_val = sum(measured[f]["dropped"]["n"] for f in FIELDS)
    kind_lost = sum(measured[f]["kind_lost"]["k"] for f in FIELDS)
    measured["keyword_dropout"] = _rate(kind_lost, n_val)

    # insertions per recording; an invented value usually comes with its verb, so the
    # direction-only rate is what the value insertions do not already account for
    ins = {f: sum(c.get(f"{kind}:inserted", 0) for kind in kinds) for f, kinds in FIELDS.items()}
    ins["heading"] += c.get("turn:inserted", 0)
    ins["direction"] = max(d["inserted"] - sum(ins.values()), 0)
    measured["insertion_per_recording"] = {f: _rate(v, n_rec) for f, v in ins.items()}

    def conf(rows):
        return confidence_model([{"confidence_proxy": r["confidence"]} for r in rows])

    clean = [r for r in items if r["confidence"] is not None and r["trajectory_errors"] == 0]
    bad = [r for r in items if r["confidence"] is not None and r["trajectory_errors"] > 0]

    for f in fields:
        fields[f] = {k: round(v, 4) for k, v in fields[f].items()}
    return {
        "name": name, "spoken": True, "scaling": "fixed",
        "source": (f"fitted by channel.fit_labels from the labelled benchmark: recognizer "
                   f"{model}, sets {', '.join(sets)}, {n_rec} recordings. Rates are per "
                   "reference value; insertion is per recording (a recording can hold an "
                   "instruction and its readback). correction_rate is a prior, not measured."
                   + (" OPTIMISTIC: jacktol models trained on most ATCO2 clips."
                      if model.startswith("jacktol") and "atco2_1h" in sets else "")),
        "full_dropout": 0.0,
        "verb_flip": fields["direction"]["flip"],
        "correction_rate": 0.03,
        "keyword_dropout": round(kind_lost / n_val, 4) if n_val else 0.0,
        "fields": fields,
        "insertion": {f: round(v / n_rec, 4) for f, v in ins.items()},
        "confidence": {"kind": "conditional", "clean": conf(clean), "error": conf(bad),
                       "noise": 0.03},
        "surface": {"garble": 0.0, "filler": 0.0, "word_drop": 0.0, "truncation": 0.0},
        "usable_for_sampling": [],
        "provisional": provisional,
        "measured": measured,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="recognizer name in data/asr/scores")
    ap.add_argument("--name", required=True, help="profile name (profiles/<name>.json)")
    ap.add_argument("--sets", nargs="+", default=["atco2_1h", "atcosim_250"])
    ap.add_argument("--unseen-only", action="store_true",
                    help="drop recordings in jacktol's training or validation data")
    ap.add_argument("--scores", default="data/asr/scores/items.jsonl")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    items = [json.loads(l) for l in Path(a.scores).read_text().splitlines() if l.strip()]
    items = [r for r in items if r["model"] == a.model and r["set"] in a.sets
             and not (a.unseen_only and r["in_jacktol_training"])]
    if not items:
        raise SystemExit(f"no scored recordings for {a.model} in {a.sets}")
    profile = fit(items, a.name, a.model, a.sets)
    print(json.dumps({k: profile[k] for k in ("fields", "keyword_dropout", "insertion",
                                              "provisional")}, indent=2))
    if a.write:
        out = PROFILES_DIR / f"{a.name}.json"
        out.write_text(json.dumps(profile, indent=2) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
