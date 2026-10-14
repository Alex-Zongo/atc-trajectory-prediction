"""Transcribe ATC audio, if you have any.

The pipeline does not require audio: the observation channel synthesises transcripts from
clearances, which is what makes the noise level a controlled variable. This module is for
two other things.

  1. FITTING a channel to your own recognizer. Transcribe a corpus for which you have human
     reference transcripts, write the pairs to JSONL, and pass that to
     `atc_forecast.channel.fit`. That is how a channel profile gets its rates.
  2. RUNNING Stage 1 on real recognizer output at inference time.

Use an ATC-finetuned model. A general-purpose recognizer is much worse on this domain,
mostly on exactly the tokens that matter: spoken digit strings and callsigns.

Caveat worth knowing before you compare synthetic against real performance. A corpus of
real utterances matched to trajectories by callsign and time does NOT give you aligned
(transcript, clearance) pairs: the matched utterance is usually a different instruction
from the one the trajectory shows, so such a set cannot serve as a real-transcript test
set for Stage 1. Measuring real-transcript performance needs utterances whose clearance
label comes from the SAME utterance.

    python -m atc_forecast.asr.transcribe audio/ --out transcripts.jsonl \
        --model jlvdoorn/whisper-large-v3-atco2-asr
"""

import argparse
import json
from pathlib import Path

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}


def transcribe_file(model, path, language="en", beam_size=5, word_timestamps=True) -> dict:
    """One audio file to a record.

    `confidence_proxy` is the mean per-word probability. It is worth keeping: a real
    recognizer's confidence predicts its own error rate strongly, and that signal is
    exactly what a fixed-value channel throws away.
    """
    segments, info = model.transcribe(str(path), language=language, beam_size=beam_size,
                                      word_timestamps=word_timestamps)
    segs, words = [], []
    for s in segments:
        segs.append({"start": round(s.start, 2), "end": round(s.end, 2),
                     "text": s.text.strip()})
        for w in (s.words or []):
            words.append({"word": w.word.strip(), "prob": round(float(w.probability), 4),
                          "start": round(w.start, 2), "end": round(w.end, 2)})
    text = " ".join(s["text"] for s in segs).strip()
    probs = [w["prob"] for w in words]
    return {"id": Path(path).stem, "audio": str(path), "hypothesis_text": text,
            "segments": segs, "words": words,
            "confidence_proxy": round(sum(probs) / len(probs), 4) if probs else None,
            "duration_s": round(float(info.duration), 2),
            "language": info.language}


def load_model(name: str, device: str = "auto", compute_type: str = "default"):
    from faster_whisper import WhisperModel

    return WhisperModel(name, device=device, compute_type=compute_type)


def transcribe_dir(audio_dir, out_path, model_name, references=None, **kw):
    """Transcribe every audio file under `audio_dir` to a JSONL.

    `references` is an optional {id: reference_text} mapping. Supplying it makes the output
    directly usable by `atc_forecast.channel.fit`, which needs both sides of the pair.
    """
    model = load_model(model_name, device=kw.pop("device", "auto"),
                       compute_type=kw.pop("compute_type", "default"))
    files = sorted(p for p in Path(audio_dir).rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        raise SystemExit(f"no audio found under {audio_dir} "
                         f"(looked for {sorted(AUDIO_SUFFIXES)})")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as fh:
        for p in files:
            try:
                rec = transcribe_file(model, p, **kw)
            except Exception as e:                    # one bad file should not kill the run
                print(f"  !! {p.name}: {e}")
                continue
            if references and rec["id"] in references:
                rec["reference_text"] = references[rec["id"]]
            fh.write(json.dumps(rec) + "\n")
            n += 1
            print(f"  {p.name}: {rec['hypothesis_text'][:70]}")
    print(f"\n{n}/{len(files)} transcribed -> {out_path}")
    if references:
        print("pairs include reference_text, so this can be passed to "
              "`python -m atc_forecast.channel.fit`")
    return out_path


def load_references(path) -> dict:
    """{id: text} from a JSON mapping, or from a JSONL with id and reference_text."""
    p = Path(path)
    if p.suffix == ".json":
        return json.loads(p.read_text())
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("reference_text"):
                out[r["id"]] = r["reference_text"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio_dir")
    ap.add_argument("--out", default="transcripts.jsonl")
    ap.add_argument("--model", default="jlvdoorn/whisper-large-v3-atco2-asr",
                    help="an ATC-finetuned recognizer; a general one is much worse here")
    ap.add_argument("--references", default=None,
                    help="JSON {id: text} or JSONL with reference_text, to make fittable pairs")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compute-type", default="default")
    ap.add_argument("--language", default="en")
    ap.add_argument("--beam-size", type=int, default=5)
    a = ap.parse_args()
    transcribe_dir(a.audio_dir, a.out, a.model,
                   references=load_references(a.references) if a.references else None,
                   device=a.device, compute_type=a.compute_type,
                   language=a.language, beam_size=a.beam_size)


if __name__ == "__main__":
    main()
