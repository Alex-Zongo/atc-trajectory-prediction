"""Build the BLIND labelling inputs for the ATC transcript-labelling pass.

The labellers must see only what the corpus says was spoken, never any earlier labels: not
our parser's slots, and not the ATCO2 corpus's own inline entity tags, which are kept back
to validate the labels afterwards. So each input item holds an id, the corpus name, and its
speaker turns (role + text), and nothing else.

ATCO2: one clip often holds a short exchange (an instruction and its readback), so it is
split into the corpus's own speaker segments. The role comes from the segment's speaker
label ("ATCO ..." is the controller; pilots are labelled with their callsign, which is
deliberately NOT passed on). The inline tags [#callsign] [#command] [#value] are the human
labels and are stripped; transcription markup ([hes], [NE ...]) is kept and documented in
INSTRUCTIONS.md.

ATCOSIM: controller speech only, one turn per utterance, taken verbatim from the corpus's
fulldata.csv. The 250-utterance subset is the one the earlier recognizer runs used.

Audio paths go to a separate index (not given to the labellers), for the transcription runs.

    python -m atc_forecast.asr.labelling.build_inputs \
        --sidecar ../CITRIS-llm-for-atc-dataset/sidecar_real_atc --out data/asr
"""

import argparse
import csv
import hashlib
import json
import re
import wave
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

# Human entity tags inside ATCO2 transcripts, in all the spellings the corpus uses.
_ATCO2_TAG = re.compile(r"\[\s*#?\s*/?\s*#?\s*(callsign|command|value|unnamed)\s*\]", re.I)
CHUNK = 40


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", _ATCO2_TAG.sub(" ", text or "")).strip()


def _audio_meta(path: Path) -> dict:
    with wave.open(str(path)) as w:
        sr, n = w.getframerate(), w.getnframes()
    return {"audio_path": str(path), "sample_rate": sr, "duration_s": round(n / sr, 3),
            "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def atco2(sidecar: Path):
    manifest = [json.loads(line) for line in (sidecar / "data/atco2_1h/manifest.jsonl")
                .read_text().splitlines() if line.strip()]
    items, index, roles = [], [], Counter()
    for m in manifest:
        root = ET.parse(m["xml_path"]).getroot()
        turns = []
        for seg in root.iter("segment"):
            label = (seg.findtext("speaker_label") or "").strip()
            role = "controller" if label.upper().startswith("ATCO") else (
                "pilot" if label else "unknown")
            roles[role] += 1
            text = _clean(seg.findtext("text"))
            if text:
                turns.append({"turn": len(turns), "role": role, "text": text})
        items.append({"item_id": m["id"], "corpus": "atco2", "turns": turns})
        index.append({"item_id": m["id"], "corpus": "atco2", **_audio_meta(Path(m["audio_path"]))})
    return items, index, roles


def atcosim(sidecar: Path):
    data = sidecar / "data/atcosim"
    subset = [json.loads(line)["id"] for line in (data / "manifest_pilot_250.jsonl")
              .read_text().splitlines() if line.strip()]
    rows = {r["filename"]: r for r in csv.DictReader(open(data / "fulldata.csv"))}
    audio = {p.stem: p for p in (data / "audio").rglob("*.wav")}
    items, index = [], []
    for i in subset:
        text = re.sub(r"\s+", " ", rows[i]["transcription"]).strip()
        items.append({"item_id": i, "corpus": "atcosim",
                      "turns": [{"turn": 0, "role": "controller", "text": text}]})
        index.append({"item_id": i, "corpus": "atcosim", **_audio_meta(audio[i])})
    return items, index


def write(items, index, name, out: Path):
    inp, idx, chunks = out / "labelling/inputs", out / "index", out / "labelling/inputs/chunks"
    for d in (inp, idx, chunks):
        d.mkdir(parents=True, exist_ok=True)
    dump = lambda rows: "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (inp / f"{name}.jsonl").write_text(dump(items))
    (idx / f"{name}.jsonl").write_text(dump(index))
    for k in range(0, len(items), CHUNK):
        (chunks / f"{name}_{k // CHUNK:03d}.jsonl").write_text(dump(items[k:k + CHUNK]))
    leaks = sum(bool(_ATCO2_TAG.search(t["text"])) for r in items for t in r["turns"])
    print(f"{name}: {len(items)} items, {sum(len(r['turns']) for r in items)} turns, "
          f"{-(-len(items) // CHUNK)} chunks, residual entity tags: {leaks}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sidecar", default="../CITRIS-llm-for-atc-dataset/sidecar_real_atc")
    ap.add_argument("--out", default="data/asr")
    a = ap.parse_args()
    sidecar, out = Path(a.sidecar), Path(a.out)
    items, index, roles = atco2(sidecar)
    write(items, index, "atco2_1h", out)
    print(f"  ATCO2 speaker roles: {dict(roles)}")
    items, index = atcosim(sidecar)
    write(items, index, "atcosim_250", out)


if __name__ == "__main__":
    main()
