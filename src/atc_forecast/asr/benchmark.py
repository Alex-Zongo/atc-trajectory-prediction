"""Transcribe the benchmark recordings with several recognizers under one fixed setting.

Every Whisper-family model runs through faster-whisper with the SAME decoding settings
(CPU, int8 weights, greedy decoding, English, word timestamps, no voice-activity filter),
so differences between models come from the models, not from how they were decoded. The
wav2vec2 model is decoded greedily from its CTC output, without its optional language
model, for the same reason. Every model receives the same audio: decoded and resampled to
16 kHz by faster-whisper's own loader.

Models that are not published in faster-whisper's format are converted once into
data/asr/models/: jacktol's large-v3 directly, and the US model by merging its LoRA adapter
into whisper-large-v3-turbo first. Downloads made only for a conversion are deleted
afterwards (the disk is nearly full); a model already in the Hugging Face cache before the
run is never touched.

Output: data/asr/hypotheses/<model>/<set>.jsonl, one line per recording. A rerun skips
recordings already written, so an interrupted run resumes where it stopped.

    python -m atc_forecast.asr.benchmark --models jacktol_medium --sets atco2_1h atcosim_250
    python -m atc_forecast.asr.benchmark --models all --limit 24     # timing check
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

# How each model is obtained. "source" is a faster-whisper model id; "convert" builds one.
MODELS = {
    "jacktol_medium": {"kind": "whisper",
                       "source": "jacktol/whisper-medium.en-fine-tuned-for-ATC-faster-whisper"},
    "jzuluaga_wav2vec2": {"kind": "ctc",
                          "source": "Jzuluaga/wav2vec2-large-960h-lv60-self-en-atc-uwb-atcc-and-atcosim"},
    "whisper_large_v3": {"kind": "whisper", "source": "Systran/faster-whisper-large-v3"},
    "jacktol_large_v3": {"kind": "whisper",
                         "convert": {"base": "jacktol/whisper-large-v3-finetuned-for-ATC"}},
    "us_atc_turbo_v3": {"kind": "whisper",
                        "convert": {"base": "openai/whisper-large-v3-turbo",
                                    "adapter": "thomaseibner/whisper-large-v3-turbo-us-atc-v3"}},
}
DECODE = {"device": "cpu", "compute_type": "int8", "beam_size": 1, "language": "en",
          "word_timestamps": True, "vad_filter": False}
CPU_THREADS = 8


def _cached_repos() -> set:
    from huggingface_hub import scan_cache_dir
    try:
        return {r.repo_id for r in scan_cache_dir().repos}
    except Exception:
        return set()


def _drop_from_cache(repo_ids: set):
    from huggingface_hub import scan_cache_dir
    cache = scan_cache_dir()
    revs = [rev.commit_hash for r in cache.repos if r.repo_id in repo_ids for rev in r.revisions]
    if revs:
        strategy = cache.delete_revisions(*revs)
        print(f"  freeing {strategy.expected_freed_size_str} of downloads: {sorted(repo_ids)}")
        strategy.execute()


def prepare(name: str, root: Path) -> str:
    """The path or id faster-whisper/transformers should load for `name`."""
    spec = MODELS[name]
    if "source" in spec:
        return spec["source"]
    out = root / "models" / name
    if (out / "model.bin").exists():
        return str(out)
    import ctranslate2.converters as conv

    before = _cached_repos()
    base, adapter = spec["convert"]["base"], spec["convert"].get("adapter")
    copy = ["tokenizer.json", "preprocessor_config.json"]
    print(f"[{name}] converting {base}" + (f" + {adapter}" if adapter else ""), flush=True)
    if adapter:
        import torch
        from huggingface_hub import hf_hub_download
        from peft import PeftModel
        from transformers import WhisperForConditionalGeneration

        merged = root / "models" / f"{name}_merged_tmp"
        if not (merged / "model.safetensors").exists():
            model = WhisperForConditionalGeneration.from_pretrained(base, dtype=torch.float32)
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
            model.save_pretrained(merged)
            del model
        # the base's own files: transformers 5 saves the processor under another name
        for f in copy:
            shutil.copy(hf_hub_download(base, f), merged / f)
        conv.TransformersConverter(str(merged), copy_files=copy).convert(
            str(out), quantization="int8", force=True)
        shutil.rmtree(merged)
    else:
        conv.TransformersConverter(base, copy_files=copy).convert(
            str(out), quantization="int8", force=True)
    _drop_from_cache({base, adapter} - before - {None})
    return str(out)


def _load_rows(index_dir: Path, sets, limit):
    rows = []
    for s in sets:
        part = [json.loads(line) for line in (index_dir / f"{s}.jsonl").read_text().splitlines()
                if line.strip()]
        rows += [(s, r) for r in (part[:limit] if limit else part)]
    return rows


class Whisper:
    def __init__(self, path):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(path, device=DECODE["device"],
                                  compute_type=DECODE["compute_type"], cpu_threads=CPU_THREADS)

    def __call__(self, audio):
        segs, _ = self.model.transcribe(audio, language=DECODE["language"],
                                        beam_size=DECODE["beam_size"],
                                        word_timestamps=DECODE["word_timestamps"],
                                        vad_filter=DECODE["vad_filter"])
        segs = list(segs)
        words = [{"word": w.word.strip(), "start": round(w.start, 2), "end": round(w.end, 2),
                  "probability": round(float(w.probability), 4)}
                 for s in segs for w in (s.words or [])]
        conf = float(np.mean([w["probability"] for w in words])) if words else None
        return {"hypothesis_text": " ".join(s.text.strip() for s in segs).strip(),
                "confidence_proxy": conf, "words": words,
                "avg_logprob": float(np.mean([s.avg_logprob for s in segs])) if segs else None,
                "no_speech_prob": float(np.mean([s.no_speech_prob for s in segs])) if segs else None}


class CTC:
    def __init__(self, path):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.processor = Wav2Vec2Processor.from_pretrained(path)
        self.model = Wav2Vec2ForCTC.from_pretrained(path).to(self.device).eval()
        self.blank = self.processor.tokenizer.pad_token_id

    def __call__(self, audio):
        torch = self.torch
        inp = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_values
        with torch.no_grad():
            logits = self.model(inp.to(self.device)).logits[0].float().cpu()
        probs = logits.softmax(-1)
        best = probs.max(-1)
        speech = best.indices != self.blank
        text = self.processor.batch_decode(best.indices[None])[0].lower().strip()
        conf = float(best.values[speech].mean()) if speech.any() else None
        return {"hypothesis_text": text, "confidence_proxy": conf, "words": [],
                "avg_logprob": None, "no_speech_prob": None}


def run(name: str, sets, root: Path, limit=None):
    from faster_whisper.audio import decode_audio

    path = prepare(name, root)
    rows = _load_rows(root / "index", sets, limit)
    out_dir = root / "hypotheses" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".limit{limit}" if limit else ""
    done = {}
    for s in sets:
        f = out_dir / f"{s}{suffix}.jsonl"
        done[s] = ({json.loads(line)["item_id"] for line in f.read_text().splitlines() if line}
                   if f.exists() else set())
    todo = [(s, r) for s, r in rows if r["item_id"] not in done[s]]
    print(f"[{name}] {len(rows) - len(todo)} already done, {len(todo)} to transcribe")
    if not todo:
        return
    t0 = time.time()
    recognizer = (Whisper if MODELS[name]["kind"] == "whisper" else CTC)(path)
    print(f"[{name}] model loaded in {time.time() - t0:.0f}s")
    settings = {"model": name, "loaded_from": path, **(DECODE if MODELS[name]["kind"] == "whisper"
                else {"decoder": "ctc_greedy", "language_model": None})}
    audio_s = run_s = 0.0
    handles = {s: (out_dir / f"{s}{suffix}.jsonl").open("a") for s in sets}
    try:
        for n, (s, r) in enumerate(todo, 1):
            audio = decode_audio(r["audio_path"], sampling_rate=16000)
            t = time.time()
            hyp = recognizer(audio)
            dt = time.time() - t
            audio_s += r["duration_s"]
            run_s += dt
            handles[s].write(json.dumps({"item_id": r["item_id"], "corpus": r["corpus"],
                                         "audio_sha256": r["audio_sha256"], **hyp,
                                         "elapsed_s": round(dt, 3), "settings": settings}) + "\n")
            handles[s].flush()
            if n % 50 == 0 or n == len(todo):
                eta = (len(todo) - n) * run_s / n
                print(f"[{name}] {n}/{len(todo)}  {run_s / audio_s:.2f}x real time  "
                      f"eta {eta / 60:.0f} min", flush=True)
    finally:
        for h in handles.values():
            h.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--sets", nargs="+", default=["atco2_1h", "atcosim_250"])
    ap.add_argument("--root", default="data/asr")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N recordings per set (written to separate files)")
    a = ap.parse_args()
    names = list(MODELS) if a.models == ["all"] else a.models
    for name in names:
        run(name, a.sets, Path(a.root), a.limit)


if __name__ == "__main__":
    main()
