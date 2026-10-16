"""Share the trained weights and prepared data through a GitHub Release.

`runs/` and `data/` are gitignored, since they are far too large for git. They travel as
assets of one release (by default `data-v1`) on the same repository, so anyone who can see
the repository can fetch them, from any branch, with one command:

    python scripts/artifacts.py fetch            # weights + windows + meta (~420 MB)

and the owner publishes them with

    python scripts/artifacts.py publish          # creates or updates the release

Raw OpenSky ADS-B is NOT published by default: OpenSky's terms restrict redistributing
it, and on a public repository every release asset is public. `publish --adsb` includes it
for a private repository; `fetch --adsb` installs it from such a release or, with
`--from`, from a folder shared privately.

Every asset is listed in SHA256SUMS and verified after download, so a truncated download
fails loudly instead of loading a broken checkpoint. Fetching needs the GitHub CLI (`gh`),
logged in with access to the repository; without it, download the assets from the release
page by hand and run `fetch --from <folder>`.

Bundles:

  runs.tar.gz       runs/stage1.pt, runs/stage2.pt, runs/lambda.json
  dataset.tar       data/dataset/meta*.parquet, data/windows_raw/*
  adsb_<day>.parquet  raw ADS-B the windows were cut from, one asset per day (opt-in)

`data/cache/` is not shipped: the aircraft database and the airport features are public
and are downloaded on first use.
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / ".artifacts"
RELEASE = "data-v1"
BUNDLES = {
    "runs.tar.gz": ["runs/stage1.pt", "runs/stage2.pt", "runs/lambda.json"],
    "dataset.tar": ["data/dataset/meta.parquet", "data/dataset/meta_grounded.parquet",
                    "data/windows_raw/windows.npz", "data/windows_raw/windows_meta.parquet"],
}
ADSB_GLOB = "data/adsb/adsb_*.parquet"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gh(*args):
    if shutil.which("gh") is None:
        sys.exit("the GitHub CLI is not installed: `brew install gh`, then `gh auth login`.\n"
                 "Or download the release assets by hand and use `fetch --from <folder>`.")
    subprocess.run(["gh", *args], cwd=ROOT, check=True)


def publish(a):
    STAGE.mkdir(exist_ok=True)
    assets = []
    for name, members in BUNDLES.items():
        missing = [m for m in members if not (ROOT / m).exists()]
        if missing:
            sys.exit(f"cannot build {name}, missing: {', '.join(missing)}")
        out = STAGE / name
        with tarfile.open(out, "w:gz" if name.endswith(".gz") else "w") as tar:
            for m in members:
                tar.add(ROOT / m, arcname=m)
        assets.append(out)
        print(f"built {name} ({out.stat().st_size / 1e6:.0f} MB)")
    if a.adsb:
        days = sorted(ROOT.glob(ADSB_GLOB))
        if not days:
            print(f"no raw ADS-B under {ADSB_GLOB}; publishing without it")
        assets += days
    sums = STAGE / "SHA256SUMS"
    sums.write_text("".join(f"{sha256(p)}  {p.name}\n" for p in assets))
    assets.append(sums)
    if a.dry_run:
        print("dry run, nothing uploaded:\n" + "\n".join(f"  {p}" for p in assets))
        return
    exists = subprocess.run(["gh", "release", "view", a.release], cwd=ROOT,
                            capture_output=True).returncode == 0
    if exists:
        gh("release", "upload", a.release, *map(str, assets), "--clobber")
    else:
        gh("release", "create", a.release, *map(str, assets), "--title",
           f"Trained weights and data ({a.release})", "--notes",
           "Model weights and prepared data for this repository. Install them with "
           "`python scripts/artifacts.py fetch`.")
    print(f"published {len(assets)} assets to release {a.release}")


def fetch(a):
    src = Path(a.src) if a.src else STAGE
    want = list(BUNDLES) + (["adsb_*.parquet"] if a.adsb else [])
    if not a.src:
        STAGE.mkdir(exist_ok=True)
        for pattern in want + ["SHA256SUMS"]:
            gh("release", "download", a.release, "--pattern", pattern, "--dir", str(STAGE),
               "--clobber")
    sums = {}
    if (src / "SHA256SUMS").exists():
        for line in (src / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split()
            sums[name] = digest
    else:
        print("warning: no SHA256SUMS next to the assets, so nothing is verified")

    def verified(p: Path) -> Path:
        if p.name in sums and sha256(p) != sums[p.name]:
            sys.exit(f"{p.name} does not match its checksum; delete it and fetch again")
        return p

    for name in BUNDLES:
        p = src / name
        if not p.exists():
            sys.exit(f"missing {p}")
        with tarfile.open(verified(p)) as tar:
            tar.extractall(ROOT, filter="data")
        print(f"installed {name}")
    if a.adsb:
        days = sorted(src.glob("adsb_*.parquet"))
        if not days:
            sys.exit(f"no adsb_*.parquet in {src}")
        (ROOT / "data/adsb").mkdir(parents=True, exist_ok=True)
        for p in days:
            shutil.copy2(verified(p), ROOT / "data/adsb" / p.name)
        print(f"installed {len(days)} ADS-B days into data/adsb/")
    print("done. `uv run pytest` and the commands in the README now run against these.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish", help="pack runs/ and data/ and upload them to the release")
    p.add_argument("--release", default=RELEASE)
    p.add_argument("--adsb", action="store_true",
                   help="also publish the raw ADS-B; only on a private repository")
    p.add_argument("--dry-run", action="store_true", help="build the assets, upload nothing")
    f = sub.add_parser("fetch", help="download, verify and unpack the release assets")
    f.add_argument("--release", default=RELEASE)
    f.add_argument("--adsb", action="store_true", help="also install the raw ADS-B days")
    f.add_argument("--from", dest="src", default=None,
                   help="install from a folder of already-downloaded assets instead")
    a = ap.parse_args()
    publish(a) if a.cmd == "publish" else fetch(a)


if __name__ == "__main__":
    main()
