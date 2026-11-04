"""Check a labelling output file against its input batch and schema.json.

    uv run --with jsonschema python -m atc_forecast.asr.labelling.validate INPUT.jsonl OUTPUT.jsonl

Exits non-zero, listing every problem, if an output line is not valid JSON, breaks the
schema, has an item id the input does not have (or misses one), or does not keep every
input turn in order.
"""

import json
import sys
from pathlib import Path

SCHEMA = Path(__file__).with_name("schema.json")


def check(inp: Path, out: Path) -> list[str]:
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(json.loads(SCHEMA.read_text()))
    items = {r["item_id"]: r for r in map(json.loads, inp.read_text().splitlines()) if r}
    errors, seen = [], []
    for n, line in enumerate(out.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {n}: not valid JSON ({e})")
            continue
        for err in validator.iter_errors(row):
            path = "/".join(map(str, err.absolute_path))
            errors.append(f"line {n} [{row.get('item_id')}] {path}: {err.message}")
        iid = row.get("item_id")
        if iid not in items:
            errors.append(f"line {n}: item_id {iid!r} is not in the input")
            continue
        seen.append(iid)
        want = [t["turn"] for t in items[iid]["turns"]]
        got = [t.get("turn") for t in row.get("turns", [])]
        if got != want:
            errors.append(f"line {n} [{iid}]: turns {got} but the input has {want}")
    missing = [i for i in items if i not in seen]
    if missing:
        errors.append(f"{len(missing)} input items have no output: {missing[:5]}")
    dupes = {i for i in seen if seen.count(i) > 1}
    if dupes:
        errors.append(f"duplicated item_ids: {sorted(dupes)[:5]}")
    return errors


def main():
    inp, out = map(Path, sys.argv[1:3])
    errors = check(inp, out)
    if errors:
        print(f"INVALID: {len(errors)} problem(s)")
        print("\n".join(errors[:60]))
        sys.exit(1)
    print(f"VALID: {out} matches {inp}")


if __name__ == "__main__":
    main()
