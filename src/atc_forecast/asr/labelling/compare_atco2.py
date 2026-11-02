"""Compare the Claude labels of the ATCO2 references with the corpus's own human tags.

The ATCO2 transcripts carry inline tags ([#callsign] [#command] [#value]) written by the
corpus annotators. They were stripped from the labelling inputs, so the labels were made
blind, and they are read back here, straight from the corpus XML, only to measure
agreement. They are never written to disk.

Read the result as AGREEMENT, not accuracy: the corpus flags its own tagging as unverified
(`correct_tagging = 0` on 875 of 877 segments), and its tags are coarse ("starts" and
"flight plan route" are tagged as values). Disagreements are written to a review file so a
person can decide which side is right.

    uv run python -m atc_forecast.asr.labelling.compare_atco2
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from .build_inputs import _ATCO2_TAG

_MARKUP = re.compile(r"\[[^\]]*\]")


def norm(text) -> str:
    t = _MARKUP.sub(" ", str(text or "").lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t)).strip()


def tokens(text) -> set:
    return set(norm(text).split())


def overlaps(a, b, frac=0.5) -> bool:
    """Two spans match if they share at least `frac` of the smaller one's words."""
    ta, tb = tokens(a), tokens(b)
    return bool(ta and tb) and len(ta & tb) >= frac * min(len(ta), len(tb))


def gold_turns(xml_path) -> list[dict]:
    """The corpus tags per non-empty segment, aligned with the labelling turns."""
    out = []
    for seg in ET.parse(xml_path).getroot().iter("segment"):
        raw = seg.findtext("text") or ""
        if not re.sub(r"\s+", " ", _ATCO2_TAG.sub(" ", raw)).strip():
            continue
        raw = raw.replace("[#/value]", "[/#value]")
        out.append({k: re.findall(rf"\[#{k}\](.*?)\[/#{k}\]", raw)
                    for k in ("callsign", "command", "value")})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sidecar", default="../CITRIS-llm-for-atc-dataset/sidecar_real_atc")
    ap.add_argument("--labels", default="data/asr/labelling/outputs/claude")
    ap.add_argument("--inputs", default="data/asr/labelling/inputs/atco2_1h.jsonl")
    ap.add_argument("--out", default="data/asr/labelling/reports")
    a = ap.parse_args()

    xml = {json.loads(l)["id"]: json.loads(l)["xml_path"] for l in
           (Path(a.sidecar) / "data/atco2_1h/manifest.jsonl").read_text().splitlines() if l}
    inputs = {json.loads(l)["item_id"]: json.loads(l) for l in
              Path(a.inputs).read_text().splitlines() if l}
    labels = {}
    for f in sorted(Path(a.labels).glob("atco2_1h_*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                labels[r["item_id"]] = r

    c = Counter()
    review = []
    for iid, lab in labels.items():
        gold = gold_turns(xml[iid])
        if len(gold) != len(lab["turns"]):
            c["turn_count_mismatch"] += 1
            continue
        for g, t, src in zip(gold, lab["turns"], inputs[iid]["turns"]):
            c["turns"] += 1
            clauses = t["clauses"]
            # callsign: detection, then text agreement where both have one
            gcs, lcs = g["callsign"], (t["callsign"] or {}).get("text")
            c[f"callsign_{'g' if gcs else '-'}{'l' if lcs else '-'}"] += 1
            if gcs and lcs:
                ok = any(norm(x) == norm(lcs) for x in gcs)
                c["callsign_text_exact"] += ok
                c["callsign_text_overlap"] += any(overlaps(x, lcs, 0.8) for x in gcs)
                if not ok:
                    review.append({"item_id": iid, "turn": t["turn"], "field": "callsign",
                                   "text": src["text"], "corpus": gcs, "claude": lcs})
            elif bool(gcs) != bool(lcs):
                review.append({"item_id": iid, "turn": t["turn"], "field": "callsign_presence",
                               "text": src["text"], "corpus": gcs, "claude": lcs})
            # values: corpus value spans against the labels' value words
            lvals = [k["value_words"] for k in clauses if k["value_words"]]
            for v in g["value"]:
                hit = any(overlaps(v, x) for x in lvals)
                c["value_gold"] += 1
                c["value_gold_found"] += hit
                if not hit:
                    review.append({"item_id": iid, "turn": t["turn"], "field": "value_missed",
                                   "text": src["text"], "corpus": v, "claude": lvals})
            for x in lvals:
                c["value_label"] += 1
                c["value_label_in_gold"] += any(overlaps(v, x) for v in g["value"])
            # commands: each corpus command phrase should sit inside some labelled clause
            spans = [k["span"] for k in clauses]
            for cmd in g["command"]:
                hit = any(overlaps(cmd, s, 1.0) for s in spans)
                c["command_gold"] += 1
                c["command_gold_found"] += hit
                if not hit:
                    review.append({"item_id": iid, "turn": t["turn"], "field": "command_missed",
                                   "text": src["text"], "corpus": cmd, "claude": spans})
            c[f"type_{t['type']}"] += 1
            for k in clauses:
                c[f"kind_{k['kind']}"] += 1
                c[f"status_{k['status']}"] += 1

    def rate(n, d):
        return round(c[n] / c[d], 3) if c[d] else None

    both = c["callsign_gl"]
    summary = {
        "items_labelled": len(labels), "items_expected": len(inputs),
        "turns_compared": c["turns"], "turn_count_mismatch": c["turn_count_mismatch"],
        "callsign": {
            "both": both, "corpus_only": c["callsign_g-"], "claude_only": c["callsign_-l"],
            "neither": c["callsign_--"],
            "text_exact_when_both": round(c["callsign_text_exact"] / both, 3) if both else None,
            "text_overlap80_when_both": round(c["callsign_text_overlap"] / both, 3) if both else None},
        "value": {"corpus_values": c["value_gold"],
                  "found_in_claude": rate("value_gold_found", "value_gold"),
                  "claude_values": c["value_label"],
                  "claude_values_in_corpus": rate("value_label_in_gold", "value_label")},
        "command": {"corpus_commands": c["command_gold"],
                    "found_in_claude_clauses": rate("command_gold_found", "command_gold")},
        "claude_turn_types": {k[5:]: v for k, v in c.items() if k.startswith("type_")},
        "claude_clause_kinds": {k[5:]: v for k, v in sorted(c.items()) if k.startswith("kind_")},
        "claude_clause_status": {k[7:]: v for k, v in c.items() if k.startswith("status_")},
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "claude_vs_atco2_tags.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "claude_vs_atco2_review.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in review))
    print(json.dumps(summary, indent=2))
    print(f"\n{len(review)} disagreements written to {out / 'claude_vs_atco2_review.jsonl'}")


if __name__ == "__main__":
    main()
