"""Find which benchmark recordings a recognizer may have been trained on.

jacktol's models are fine-tuned on jacktol/ATC-ASR-Dataset, which its card says combines
UWB-ATCC with the ATCO2 1-hour test subset: the same recordings as our ATCO2 benchmark. The
dataset's ids are random, so recordings are matched by transcript: a clip overlaps a split
if its whole transcript, or any of its speaker turns of 3 words or more, appears in it after
normalisation (markup removed, upper case, letters only, "niner" = "nine"). Only the text
column is fetched, through the Hugging Face dataset viewer API; no audio is downloaded.

A text match is evidence, not proof: a short common phrase can recur across recordings.
But here most clips match, which only an overlap explains.

Output: data/asr/index/overlap_jacktol.json, {item_id: [splits]} for every benchmark item.

    python -m atc_forecast.asr.overlap
"""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

DATASET = "jacktol/ATC-ASR-Dataset"
API = "https://datasets-server.huggingface.co"


def norm(text: str) -> str:
    t = re.sub(r"\[NE[^\]]*\].*?\[/NE\]", " ", text)
    t = re.sub(r"\[[^\]]*\]|<[^>]*>", " ", t.upper()).replace("NINER", "NINE")
    return " ".join(re.sub(r"[^A-Z ]", " ", t).split())


def _get(url: str) -> dict:
    for attempt in range(8):
        try:
            return json.load(urllib.request.urlopen(url, timeout=60))
        except Exception:
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}")


def fetch_texts(dataset: str) -> dict:
    """{normalised text: {splits}} for every row of every split."""
    where = {}
    for s in _get(f"{API}/splits?dataset={dataset}")["splits"]:
        off = 0
        while True:
            d = _get(f"{API}/rows?dataset={dataset}&config={s['config']}&split={s['split']}"
                     f"&offset={off}&length=100")
            for r in d["rows"]:
                where.setdefault(norm(r["row"]["text"]), set()).add(s["split"])
            off += 100
            if off >= d["num_rows_total"]:
                break
    return where


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", default="data/asr/labelling/inputs")
    ap.add_argument("--out", default="data/asr/index/overlap_jacktol.json")
    a = ap.parse_args()
    where = fetch_texts(DATASET)
    result = {}
    for f in sorted(Path(a.inputs).glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            found = set(where.get(norm(" ".join(t["text"] for t in item["turns"])), ()))
            for t in item["turns"]:
                n = norm(t["text"])
                if len(n.split()) >= 3:
                    found |= where.get(n, set())
            result[item["item_id"]] = sorted(found)
    Path(a.out).write_text(json.dumps(result, indent=1) + "\n")
    for f in sorted(Path(a.inputs).glob("*.jsonl")):
        ids = [json.loads(l)["item_id"] for l in f.read_text().splitlines() if l.strip()]
        trained = sum("train" in result[i] for i in ids)
        print(f"{f.stem}: {trained} of {len(ids)} recordings match {DATASET}'s train split")


if __name__ == "__main__":
    main()
