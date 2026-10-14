"""Fit an observation channel to a real recognizer, and check that the fit is honest.

Input is a JSONL file with one record per utterance, each carrying at least:

    {"reference_text": "<human transcript>", "hypothesis_text": "<recognizer output>",
     "confidence_proxy": 0.94}

`confidence_proxy` is optional. Both texts are parsed for the fields the channel models,
and the per-field dropout and wrong-value rates become the channel JSON.

Two things matter about this procedure.

Fitting marginal rates is not validation. The rates match the recognizer BY CONSTRUCTION,
so measuring them again proves nothing. What fitting cannot guarantee is the structure
layered on top: the channel corrupts each field with an independent draw, assumes no
dependence on utterance length, and emits a single confidence value. `fidelity` tests all
three against the same records, and a channel should not be used without reporting it.

Altitude parsing is easy to get wrong. "N thousand M hundred" is ONE altitude, not two, and
the digits before "thousand" form a run, so "one four thousand" is 14,000 and not 4,000.
Getting this wrong corrupts the REFERENCE side, which scores a recognizer that transcribed
the utterance perfectly as having missed an altitude; on one recognizer that inflated the
measured altitude error rate from roughly 0.23 to 0.39.

    python -m atc_forecast.channel.fit records.jsonl --name myasr --write
"""

import argparse
import itertools
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .model import PROFILES_DIR

WORD_DIGITS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "tree": "3",
    "four": "4", "for": "4", "five": "5", "fife": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "niner": "9",
}
_RUN = r"(?:\d\s*)+"
FIELDS = ("heading", "altitude", "speed", "direction")


def normalize(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r"\[[^\]]*\]", " ", t)          # annotation tags
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def words_to_digits(text: str) -> str:
    return " ".join(WORD_DIGITS.get(w, w) for w in text.split())


def altitudes_ft(text: str) -> list[int]:
    """Altitudes in feet, handling "N thousand M hundred" as one value."""
    s = words_to_digits(normalize(text))
    out, spans = [], []
    for m in re.finditer(rf"\b({_RUN})thousand(?:\s*({_RUN})hundred)?", s):
        th = int(re.sub(r"\s+", "", m.group(1)))
        hu = int(re.sub(r"\s+", "", m.group(2))) if m.group(2) else 0
        out.append(th * 1000 + hu * 100)
        spans.append((m.start(), m.end()))
    for m in re.finditer(rf"\b(?:flight level|fl)\s+({_RUN})(hundred)?", s):
        if any(a <= m.start() < b for a, b in spans):
            continue
        # "flight level one hundred" is FL100, so the spoken hundred multiplies the LEVEL
        # and the level is itself hundreds of feet
        lvl = int(re.sub(r"\s+", "", m.group(1))) * (100 if m.group(2) else 1)
        out.append(lvl * 100)
    return sorted(set(out))


def headings_deg(text: str) -> list[int]:
    s = words_to_digits(normalize(text))
    out = []
    for m in re.finditer(rf"\bheading\s+({_RUN})", s):
        d = re.sub(r"\s+", "", m.group(1))[:3]
        if d:
            out.append(int(d) % 360)
    return sorted(set(out))


def speeds_kt(text: str) -> list[int]:
    """Commanded speeds only. Requires a speed verb, because a bare "N knots" in real ATC
    is almost always a WIND report, and counting those would fit the channel to noise."""
    s = words_to_digits(normalize(text))
    out = []
    for m in re.finditer(rf"\b(?:speed|reduce|increase|maintain)\s+(?:speed\s+)?({_RUN})", s):
        d = re.sub(r"\s+", "", m.group(1))[:3]
        if d:
            out.append(int(d))
    return sorted(set(out))


def directions(text: str) -> list[str]:
    s = normalize(text)
    return sorted({w for w in ("left", "right") if re.search(rf"\bturn\s+\w*\s*{w}\b|\b{w}\s+turn\b", s)})


def verbs(text: str) -> list[str]:
    s = normalize(text)
    return sorted({v for v in ("climb", "descend", "descent") if v in s})


PARSERS = {"heading": headings_deg, "altitude": altitudes_ft,
           "speed": speeds_kt, "direction": directions}


def classify_altitude_error(missed: set[int], hyp: set[int]) -> str:
    """Name the error mode. A uniform digit substitution essentially never generates the
    first two, so if they dominate the scalar digit_error rate is the wrong model."""
    for m in missed:
        for h in hyp:
            if m >= 10000 and h == m - 10000:
                return "leading_digit_lost"
            if h == m * 10 or h * 10 == m:
                return "factor_of_10"
            if abs(h - m) == 1000:
                return "off_by_1000"
            if 0 < abs(h - m) < 1000:
                return "off_by_<1000"
    return "other"


def load_records(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def field_rates(records: list[dict]) -> dict:
    """Per-field dropout and wrong-value rates, over records where the reference has it."""
    out, modes = {}, Counter()
    for f, parse in PARSERS.items():
        n = miss = wrong = 0
        for r in records:
            ref = set(parse(r.get("reference_text", "")))
            hyp = set(parse(r.get("hypothesis_text", "")))
            if not ref:
                continue
            n += 1
            if not hyp:
                miss += 1
            elif not ref <= hyp:
                wrong += 1
                if f == "altitude":
                    modes[classify_altitude_error(ref - hyp, hyp)] += 1
        out[f] = {"n_reference": n,
                  "dropout": round(miss / max(n, 1), 4),
                  "digit_error" if f != "direction" else "flip": round(wrong / max(n, 1), 4)}
    # Publish an error-mode DISTRIBUTION only with enough genuine errors to estimate one.
    # Below that the counts are recorded but must not be used as weights: fitting a
    # confusion model to a handful of errors is fitting noise.
    total = sum(modes.values())
    if total >= 20:
        out["altitude"]["error_modes"] = {k: round(v / total, 4) for k, v in modes.items()}
    elif total:
        out["altitude"]["error_mode_counts_provisional"] = dict(modes)
    return out


def verb_flip_rate(records: list[dict]) -> float:
    n = flips = 0
    for r in records:
        ref, hyp = set(verbs(r.get("reference_text", ""))), set(verbs(r.get("hypothesis_text", "")))
        ref = {"descend" if v == "descent" else v for v in ref}
        hyp = {"descend" if v == "descent" else v for v in hyp}
        if not ref:
            continue
        n += 1
        if hyp and not ref & hyp:
            flips += 1
    return round(flips / max(n, 1), 4)


def confidence_model(records: list[dict], n_bins: int = 4) -> dict:
    """An EMPIRICAL confidence distribution when the records carry one.

    A fixed value is the honest default when no proxy is available, but it means the
    confidence input carries no per-utterance information: measured on real records the
    bottom confidence quartile is several times as error-prone as the top, and a constant
    throws that signal away. Binning preserves it.
    """
    conf = [float(r["confidence_proxy"]) for r in records if r.get("confidence_proxy") is not None]
    if len(conf) < 40:
        return {"kind": "fixed", "value": round(float(np.mean(conf)), 4) if conf else 0.8,
                "noise": 0.05}
    c = np.asarray(conf)
    edges = np.quantile(c, np.linspace(0, 1, n_bins + 1))
    bins = []
    for i in range(n_bins):
        m = (c >= edges[i]) & (c <= edges[i + 1])
        if m.sum():
            bins.append({"mean_confidence": round(float(c[m].mean()), 4),
                         "count": int(m.sum())})
    return {"kind": "empirical_bins", "bins": bins, "noise": 0.03,
            "mean": round(float(c.mean()), 4)}


def fidelity(records: list[dict]) -> dict:
    """Test the channel's three structural assumptions against the same records.

    independence  the channel draws each field separately, so it predicts that the odds of
                  a second field being wrong are the same whether or not a first one is.
                  A ratio above 1 means real errors CLUSTER, and the synthetic transcripts
                  are then systematically easier than real ones.
    length        the channel has no length term, so WER should not depend on length.
    confidence    a fixed-value channel carries no information about how damaged this
                  particular utterance is; this reports how much is being thrown away.
    """
    wrong_sets = []
    for r in records:
        p = {}
        for f, parse in PARSERS.items():
            ref = set(parse(r.get("reference_text", "")))
            if ref:
                p[f] = not ref <= set(parse(r.get("hypothesis_text", "")))
        if len(p) >= 2:
            wrong_sets.append(p)
    given_bad, given_ok = [], []
    for p in wrong_sets:
        for a, b in itertools.permutations(p, 2):
            (given_bad if p[a] else given_ok).append(p[b])
    ratio = (float(np.mean(given_bad)) / float(np.mean(given_ok))
             if given_bad and given_ok and np.mean(given_ok) > 0 else float("nan"))

    out = {"n_records": len(records), "n_multifield": len(wrong_sets),
           "independence_ratio": round(ratio, 3) if ratio == ratio else None,
           "independence_holds": bool(ratio <= 1.05) if ratio == ratio else None}

    wer = np.array([r["wer"] for r in records if r.get("wer") is not None], float)
    nw = np.array([r.get("reference_words") or len(normalize(r.get("reference_text", "")).split())
                   for r in records if r.get("wer") is not None], float)
    if len(wer) > 10:
        out["wer_mean"] = round(float(wer.mean()), 4)
        out["corr_wer_length"] = round(float(np.corrcoef(wer, nw)[0, 1]), 3)
    conf = np.array([r["confidence_proxy"] for r in records
                     if r.get("confidence_proxy") is not None and r.get("wer") is not None], float)
    if len(conf) > 10 and len(conf) == len(wer):
        out["corr_confidence_wer"] = round(float(np.corrcoef(conf, wer)[0, 1]), 3)
        q = np.quantile(conf, [0.25, 0.75])
        hi = wer[conf >= q[1]].mean()
        out["wer_ratio_low_vs_high_confidence"] = round(
            float(wer[conf <= q[0]].mean() / hi), 2) if hi > 0 else None
    return out


def build_channel(records: list[dict], name: str, empirical_confidence: bool = True) -> dict:
    rates = field_rates(records)
    fields = {f: {k: v for k, v in rates[f].items() if k != "n_reference"} for f in FIELDS}
    return {
        "name": name, "spoken": True, "scaling": "fixed",
        "source": f"fitted from {len(records)} reference/hypothesis pairs",
        "full_dropout": 0.0,
        "verb_flip": verb_flip_rate(records),
        "correction_rate": 0.03,
        "fields": fields,
        "confidence": (confidence_model(records) if empirical_confidence
                       else {"kind": "fixed", "value": 0.8, "noise": 0.05}),
        "surface": {"garble": 0.0, "filler": 0.0, "word_drop": 0.0, "truncation": 0.0},
        "provisional": tuple(f for f in FIELDS if rates[f]["n_reference"] < 30),
        "fidelity": fidelity(records),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("records", help="JSONL with reference_text / hypothesis_text per utterance")
    ap.add_argument("--name", required=True, help="channel name (profiles/<name>.json)")
    ap.add_argument("--fixed-confidence", action="store_true",
                    help="emit one confidence value instead of the empirical distribution")
    ap.add_argument("--write", action="store_true", help="write the profile JSON")
    a = ap.parse_args()

    records = load_records(a.records)
    ch = build_channel(records, a.name, empirical_confidence=not a.fixed_confidence)
    print(json.dumps(ch, indent=2, default=list))
    f = ch["fidelity"]
    if f.get("independence_holds") is False:
        print("\nWARNING: real field errors CLUSTER (independence ratio "
              f"{f['independence_ratio']}). Synthetic transcripts from this channel will be "
              "easier than real ones; report this with any result that uses it.")
    if a.write:
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        out = PROFILES_DIR / f"{a.name}.json"
        out.write_text(json.dumps(ch, indent=2, default=list) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
