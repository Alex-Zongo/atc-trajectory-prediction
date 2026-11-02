"""Score each recognizer's labels against the reference labels: which values survive.

The labels only EXTRACT what each text says; every judgement below is made here, by code,
the same way for every recognizer. A recording is compared as a whole: an ATCO2 clip's
reference turns (instruction and readback) are pooled, since a recognizer output has no
turns, and a value said twice counts once.

For each numeric instruction element of the reference (heading, relative turn, altitude,
speed, vertical rate, frequency, squawk, altimeter setting), the recognizer's labels give:

    correct       the same kind of element with the same value
    wrong         the same kind of element with another value (misheard)
    value_missing the element is there but its value is not (cut off or dropped)
    kind_lost     the value is there but not as that element ("345" without "heading")
    dropped       nothing of that kind

and any value of those kinds the reference does not have is `inserted`. Directions
(left/right, climb/descend, increase/reduce) are read from any clause and are `correct`,
`swapped` (the opposite one only), `dropped` or `inserted`. Callsigns are compared with
spaces removed ("k l m 3 4 6" = "klm 346"), and separately on their digits alone, since
the airline name is often misheard while the flight number survives.

Word error rate is reported too, after a light normalisation (lowercase, corpus markup
removed, digits spelled out one by one). It still penalises a correct "4000" against a
reference "four thousand", so read it as a rough guide; the value counts are the measure.

    uv run python -m atc_forecast.asr.labelling.score
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

NUMERIC = ["heading", "turn", "altitude", "speed", "vertical_rate", "frequency", "squawk",
           "altimeter"]
TRAJECTORY = {"heading", "turn", "altitude", "speed", "vertical_rate"}
AXIS = {"left": "right", "right": "left", "climb": "descend", "descend": "climb",
        "increase": "reduce", "reduce": "increase"}
DIGITS = "zero one two three four five six seven eight nine".split()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _candidates(clause) -> set:
    """The values a clause can stand for. An altitude without a unit may be feet or a level."""
    v, unit = clause["value"], clause["unit"]
    if v is None:
        return set()
    if isinstance(v, str):
        return {v.strip().lower().lstrip("0") or "0"}
    if clause["kind"] == "altitude":
        return {round(v * 100)} if unit == "FL" else {round(v)} if unit == "ft" else {round(v), round(v * 100)}
    return {round(float(v), 3)}


def elements(label: dict) -> dict:
    """Pool a label's turns into {kind: [value sets]}, directions and callsigns."""
    vals, nulls, dirs, callsigns = defaultdict(list), set(), set(), []
    for t in label["turns"]:
        if t.get("callsign"):
            callsigns.append(re.sub(r"\s+", "", t["callsign"]["normalized"].lower()))
        for c in t["clauses"]:
            if c["direction"] in AXIS:
                dirs.add(c["direction"])
            if c["kind"] not in NUMERIC and c["kind"] != "other":
                continue
            cand = _candidates(c)
            if not cand:
                nulls.add(c["kind"])
            elif not any(cand & seen for seen in vals[c["kind"]]):
                vals[c["kind"]].append(cand)
    return {"values": vals, "nulls": nulls, "dirs": dirs, "callsigns": callsigns}


def compare(ref: dict, hyp: dict) -> Counter:
    c = Counter()
    for kind in NUMERIC:
        refs, hyps = list(ref["values"].get(kind, [])), list(hyp["values"].get(kind, []))
        for r in list(refs):
            hit = next((h for h in hyps if r & h), None)
            if hit is not None:
                c[f"{kind}:correct"] += 1
                refs.remove(r)
                hyps.remove(hit)
        anywhere = [v for k, vs in hyp["values"].items() if k != kind for v in vs]
        for r in refs:
            if hyps:
                hyps.pop(0)
                c[f"{kind}:wrong"] += 1
            elif kind in hyp["nulls"]:
                c[f"{kind}:value_missing"] += 1
            elif any(r & v for v in anywhere):
                c[f"{kind}:kind_lost"] += 1
            else:
                c[f"{kind}:dropped"] += 1
        c[f"{kind}:inserted"] += len(hyps)
    for d in ref["dirs"]:
        c["direction:" + ("correct" if d in hyp["dirs"] else
                          "swapped" if AXIS[d] in hyp["dirs"] else "dropped")] += 1
    c["direction:inserted"] += len({d for d in hyp["dirs"] if d not in ref["dirs"]
                                    and AXIS[d] not in ref["dirs"]})
    rc, hc = ref["callsigns"][:1], hyp["callsigns"][:1]
    if rc:
        c["callsign:" + ("missing" if not hc else "correct" if rc == hc else "wrong")] += 1
        c["callsign_number:" + ("correct" if hc and re.sub(r"\D", "", rc[0]) ==
                                re.sub(r"\D", "", hc[0]) else "lost")] += 1
    elif hc:
        c["callsign:inserted"] += 1
    return c


_MARKUP = re.compile(r"\[[^\]]*\]|<[^>]*>|[~@=]")


def words(text: str) -> list[str]:
    t = _MARKUP.sub(" ", text.lower()).replace("niner", "nine")
    t = re.sub(r"\d", lambda m: f" {DIGITS[int(m.group())]} ", t)
    return re.sub(r"[^a-z' ]", " ", t).split()


def edit_distance(a: list, b: list) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/asr")
    ap.add_argument("--reference", default="data/asr/labelling/outputs/claude_v2")
    a = ap.parse_args()
    root = Path(a.root)
    refs = {r["item_id"]: r for f in sorted(Path(a.reference).glob("*.jsonl")) for r in _jsonl(f)}
    texts = {r["item_id"]: " ".join(t["text"] for t in r["turns"])
             for f in (root / "labelling/inputs").glob("*.jsonl") for r in _jsonl(f)}

    # Recordings in jacktol's training or validation data (see atc_forecast.asr.overlap) are
    # scored again without them, as "<set>_unseen", so every model meets them for the first time.
    overlap_file = root / "index/overlap_jacktol.json"
    overlap = json.loads(overlap_file.read_text()) if overlap_file.exists() else {}
    seen = {i for i, splits in overlap.items() if {"train", "validation"} & set(splits)}

    rows, summary = [], {}
    for f in sorted((root / "labelling/outputs/hypotheses").glob("*/*.jsonl")):
        model, set_ = f.parent.name, f.stem
        hyps = {h["item_id"]: h for h in _jsonl(root / "hypotheses" / model / f"{set_}.jsonl")}
        groups = defaultdict(lambda: {"counts": Counter(), "err": 0, "n": 0, "labelled": 0})
        for lab in _jsonl(f):
            iid = lab["item_id"]
            c = compare(elements(refs[iid]), elements(lab))
            ref_words, hyp_words = words(texts[iid]), words(hyps[iid]["hypothesis_text"])
            for g in [set_] + ([f"{set_}_unseen"] if overlap and any(
                    i in seen for i in hyps) and iid not in seen else []):
                groups[g]["counts"] += c
                groups[g]["err"] += edit_distance(ref_words, hyp_words)
                groups[g]["n"] += len(ref_words)
                groups[g]["labelled"] += 1
            errors = sum(v for k, v in c.items()
                         if k.split(":")[0] in TRAJECTORY | {"direction"}
                         and not k.endswith(":correct"))
            rows.append({"model": model, "set": set_, "item_id": iid,
                         "in_jacktol_training": iid in seen,
                         "confidence": hyps[iid]["confidence_proxy"],
                         "trajectory_errors": errors, "counts": dict(c)})
        for g, s in groups.items():
            summary.setdefault(model, {})[g] = {
                "recordings": len(hyps) if g == set_ else len([i for i in hyps if i not in seen]),
                "labelled": s["labelled"],
                "wer": round(s["err"] / s["n"], 3) if s["n"] else None,
                "counts": {k: s["counts"][k] for k in sorted(s["counts"])}}

    out = root / "scores"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    outcomes = ["correct", "wrong", "value_missing", "kind_lost", "dropped", "inserted"]
    print(f"{'model':20} {'set':19} {'n':>4} {'WER':>6}   trajectory values: "
          + " ".join(f"{o:>9}" for o in outcomes) + "   dir swapped/dropped   callsign ok   number ok")
    for model, sets in summary.items():
        for set_, s in sets.items():
            k = s["counts"]
            traj = [sum(k.get(f"{kind}:{o}", 0) for kind in TRAJECTORY) for o in outcomes]
            n_dir = sum(k.get(f"direction:{o}", 0) for o in ("correct", "swapped", "dropped"))
            n_cs = sum(k.get(f"callsign:{o}", 0) for o in ("correct", "wrong", "missing"))
            print(f"{model:20} {set_:19} {s['labelled']:>4} {s['wer']:>6}   {'':19}"
                  + " ".join(f"{v:>9}" for v in traj)
                  + f"   {k.get('direction:swapped', 0):>3}/{k.get('direction:dropped', 0):<3} of {n_dir:<4}"
                  + f"   {k.get('callsign:correct', 0):>4}/{n_cs:<6}"
                  + f"{k.get('callsign_number:correct', 0):>4}/{n_cs}")
    print(f"\nwritten to {out}/summary.json and items.jsonl")


if __name__ == "__main__":
    main()
