"""Assemble the Hugging Face Space folder for the demo. Nothing is uploaded.

    uv run python demo/build_bundle.py              # once, or after data changes
    uv run python demo/deploy.py [--out demo/space] [--models runs/v2]

The folder holds the Dockerfile and Space card (from demo/space_files), the package source,
the demo service and front end, the data bundle and the checkpoints the demo can run. To
publish it, create the Space (Docker SDK, private at first) and upload the folder yourself:

    hf repo create <owner>/atc-trajectory-demo --repo-type space --space-sdk docker --private
    hf upload <owner>/atc-trajectory-demo demo/space . --repo-type space

Then test it while private, and make it public from the Space settings.
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo.server import variants as V

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "demo/space"))
    ap.add_argument("--models", default=str(ROOT / "runs/v2"))
    a = ap.parse_args()
    out, models = Path(a.out), Path(a.models)
    bundle = ROOT / "demo/bundle"
    missing = [f for f in V.files() if not (models / f).exists()]
    if missing:
        sys.exit(f"missing checkpoints in {models}: {missing}")
    if not (bundle / "windows.npz").exists():
        sys.exit("no bundle: run demo/build_bundle.py first")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in (ROOT / "demo/space_files").iterdir():
        shutil.copy2(f, out / f.name)
    shutil.copy2(ROOT / "pyproject.toml", out / "pyproject.toml")
    shutil.copytree(ROOT / "src", out / "src", ignore=IGNORE)
    for d in ("server", "web", "bundle"):
        shutil.copytree(ROOT / "demo" / d, out / "demo" / d, ignore=IGNORE)
    (out / "demo/__init__.py").write_text("")
    (out / "models").mkdir()
    for f in V.files():
        shutil.copy2(models / f, out / "models" / f)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"Space folder {out}: {size / 1e6:.0f} MB, {len(V.files())} checkpoints")


if __name__ == "__main__":
    main()
