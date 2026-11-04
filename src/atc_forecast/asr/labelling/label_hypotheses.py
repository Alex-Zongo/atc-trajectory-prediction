"""Label the recognizer outputs with Claude, under the same instructions as the references.

Each transcript is labelled in its own request, BLIND: the request holds INSTRUCTIONS.md and
one machine transcript, never the recognizer's name, the reference text or the reference
labels, and never another recognizer's output for the same recording. A recognizer output
has no speaker turns, so the whole recording is one turn with role "unknown".

The requests go through the Message Batches API (half price); the instructions are a cached
system prompt. Labels are checked against schema.json before they are kept; a label that
fails is simply not written, so the next `submit` sends it again.

    # always from a login shell, so ANTHROPIC_API_KEY is set:
    zsh -ic 'uv run python -m atc_forecast.asr.labelling.label_hypotheses probe -n 2'
    zsh -ic 'uv run python -m atc_forecast.asr.labelling.label_hypotheses submit'
    zsh -ic 'uv run python -m atc_forecast.asr.labelling.label_hypotheses collect'

Output: data/asr/labelling/outputs/hypotheses/<model>/<set>.jsonl, in the reference labels'
format plus a `provenance` field (labeller model, effort, instructions hash, batch id).
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

HERE = Path(__file__).parent
INSTRUCTIONS = (HERE / "INSTRUCTIONS.md").read_text()
INSTRUCTIONS_SHA = hashlib.sha256(INSTRUCTIONS.encode()).hexdigest()[:12]
MODEL = "claude-opus-5-5"
# USD per million tokens for MODEL; the batch tier is half of each.
PRICE = {"input": 4.0, "output": 20.0, "cache_write": 5.0, "cache_read": 0.4}
PREAMBLE = """
## This run: machine transcripts

The texts you label in this run are automatic speech-recognizer output for ATC radio
recordings, not human transcripts. So:

- There is no corpus markup and no speaker segmentation: the whole recording is one turn,
  role "unknown". It may hold several speakers (an instruction and its readback); choose
  `type` from the main content and label every clause.
- Recognizers make errors. Label exactly what the text says, errors included: a misheard
  word or number is labelled as written, never corrected to what was probably said. Words
  that make no sense are not content; mention them in `notes` if they look like a garbled
  instruction element.
- Reply with the output JSON object for the one input item, on one line, and nothing else.
"""


def _system():
    return [{"type": "text", "text": INSTRUCTIONS + PREAMBLE, "cache_control": {"type": "ephemeral"}}]


def _params(item: dict, effort: str) -> dict:
    return {"model": MODEL, "max_tokens": 16000, "system": _system(),
            "thinking": {"type": "adaptive"}, "output_config": {"effort": effort},
            "messages": [{"role": "user", "content": json.dumps(item, ensure_ascii=False)}]}


def _custom_id(model: str, set_: str, item_id: str) -> str:
    return "h" + hashlib.sha1(f"{model}|{set_}|{item_id}".encode()).hexdigest()[:20]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


class Store:
    def __init__(self, root: Path):
        self.hyp = root / "hypotheses"
        self.out = root / "labelling/outputs/hypotheses"
        self.batches = root / "labelling/batches/hypotheses.jsonl"
        self.batches.parent.mkdir(parents=True, exist_ok=True)

    def todo(self, models=None) -> list[dict]:
        """Every transcript with no kept label and no request still in flight."""
        pending = {cid for b in _jsonl(self.batches) if not b.get("collected")
                   for cid in b["requests"]}
        rows = []
        for f in sorted(self.hyp.glob("*/*.jsonl")):
            model, set_ = f.parent.name, f.stem
            if "." in set_ or (models and model not in models):
                continue                        # .limitN files are timing checks
            done = {r["item_id"] for r in _jsonl(self.out / model / f"{set_}.jsonl")}
            for h in _jsonl(f):
                cid = _custom_id(model, set_, h["item_id"])
                if h["item_id"] not in done and cid not in pending:
                    rows.append({"custom_id": cid, "model": model, "set": set_,
                                 "item_id": h["item_id"], "corpus": h["corpus"],
                                 "text": h["hypothesis_text"]})
        return rows

    def keep(self, req: dict, text: str, provenance: dict) -> str | None:
        """Write the label if it parses and validates; else return why not."""
        from jsonschema import Draft202012Validator
        try:
            label = json.loads(text[text.index("{"):text.rindex("}") + 1])
        except ValueError as e:
            return f"not JSON: {e}"
        errors = [e.message for e in Draft202012Validator(
            json.loads((HERE / "schema.json").read_text())).iter_errors(label)]
        if label.get("item_id") != req["item_id"]:
            errors.append(f"item_id {label.get('item_id')!r}")
        if [t.get("turn") for t in label.get("turns", [])] != [0]:
            errors.append("turns are not [0]")
        if errors:
            return "; ".join(errors[:3])
        path = self.out / req["model"] / f"{req['set']}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps({**label, "provenance": provenance}, ensure_ascii=False) + "\n")
        return None


def _item(req: dict) -> dict:
    return {"item_id": req["item_id"], "corpus": req["corpus"],
            "turns": [{"turn": 0, "role": "unknown", "text": req["text"]}]}


def _cost(usage, batch: bool) -> float:
    u = {"input": usage.input_tokens, "output": usage.output_tokens,
         "cache_write": usage.cache_creation_input_tokens or 0,
         "cache_read": usage.cache_read_input_tokens or 0}
    return sum(PRICE[k] * v for k, v in u.items()) / 1e6 * (0.5 if batch else 1.0)


def _text(message) -> str:
    return "".join(b.text for b in message.content if b.type == "text")


def probe(store: Store, n: int, effort: str):
    """Label n transcripts directly (full price) to measure tokens and cost per request."""
    import anthropic
    client = anthropic.Anthropic()
    reqs = store.todo()
    total = 0.0
    for req in reqs[:n]:
        msg = client.messages.create(**_params(_item(req), effort))
        c = _cost(msg.usage, batch=False)
        total += c
        why = store.keep(req, _text(msg), {"labeller": MODEL, "effort": effort,
                                           "instructions_sha": INSTRUCTIONS_SHA, "batch": None})
        print(f"{req['model']}/{req['item_id']}: {msg.usage}  ${c:.4f}  "
              f"{'kept' if why is None else 'REJECTED: ' + why}")
    if n:
        per = total / n / 2                    # batch price per request
        print(f"\n~${per:.4f} per request at batch price; {len(reqs) - n} left -> "
              f"~${per * (len(reqs) - n):.0f}")


def submit(store: Store, effort: str, limit, models):
    import anthropic
    client = anthropic.Anthropic()
    reqs = store.todo(models)[:limit]
    if not reqs:
        print("nothing to label")
        return
    for k in range(0, len(reqs), 10000):
        part = reqs[k:k + 10000]
        batch = client.messages.batches.create(requests=[
            {"custom_id": r["custom_id"], "params": _params(_item(r), effort)} for r in part])
        with store.batches.open("a") as f:
            f.write(json.dumps({"batch_id": batch.id, "effort": effort, "submitted": time.time(),
                                "requests": {r["custom_id"]: r for r in part}}) + "\n")
        print(f"submitted {batch.id}: {len(part)} requests")


def collect(store: Store):
    import anthropic
    client = anthropic.Anthropic()
    records = _jsonl(store.batches)
    for b in records:
        if b.get("collected"):
            continue
        status = client.messages.batches.retrieve(b["batch_id"])
        if status.processing_status != "ended":
            print(f"{b['batch_id']}: {status.processing_status} {status.request_counts}")
            continue
        kept, rejected, cost = 0, [], 0.0
        prov = {"labeller": MODEL, "effort": b["effort"], "instructions_sha": INSTRUCTIONS_SHA,
                "batch": b["batch_id"]}
        for res in client.messages.batches.results(b["batch_id"]):
            req = b["requests"][res.custom_id]
            if res.result.type != "succeeded":
                rejected.append((res.custom_id, res.result.type))
                continue
            cost += _cost(res.result.message.usage, batch=True)
            why = store.keep(req, _text(res.result.message), prov)
            if why is None:
                kept += 1
            else:
                rejected.append((res.custom_id, why))
        b.update(collected=time.time(), kept=kept, rejected=rejected, cost_usd=round(cost, 2))
        print(f"{b['batch_id']}: kept {kept}, rejected {len(rejected)}, ${cost:.2f}")
        for cid, why in rejected[:10]:
            print(f"  {b['requests'][cid]['model']}/{b['requests'][cid]['item_id']}: {why}")
    store.batches.write_text("".join(json.dumps(b) + "\n" for b in records))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["probe", "submit", "collect", "status"])
    ap.add_argument("-n", type=int, default=2, help="probe: how many transcripts")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--limit", type=int, default=None, help="submit: at most this many")
    ap.add_argument("--models", nargs="+", default=None, help="submit: only these recognizers")
    ap.add_argument("--root", default="data/asr")
    a = ap.parse_args()
    store = Store(Path(a.root))
    if a.command == "probe":
        probe(store, a.n, a.effort)
    elif a.command == "submit":
        submit(store, a.effort, a.limit, a.models)
    elif a.command == "collect":
        collect(store)
    else:
        rows = store.todo()
        by = {}
        for r in rows:
            by[r["model"]] = by.get(r["model"], 0) + 1
        print(f"{len(rows)} transcripts waiting for a label: {by}")


if __name__ == "__main__":
    main()
