"""Share the trained weights and prepared data through a GitHub Release.

`runs/` and `data/` are gitignored, since they are far too large for git. They travel as
assets of one release (by default `data-v1`) on the same repository, so anyone who can see
the repository can fetch them, from any branch, with one command:

    python scripts/artifacts.py fetch --models-only        # v2: main model and baselines (~38 MB)
    python scripts/artifacts.py fetch --hub <owner>        # v2: + dataset (~2 GB, gated)
    python scripts/artifacts.py fetch --all-runs           # v2: + every training run (~110 MB), if published
    python scripts/artifacts.py fetch --models-only --demo # v2: + the web demo's extra models and data (~45 MB)
    python scripts/artifacts.py fetch --set v1             # v1 (release data-v1, ~420 MB)

and the owner publishes them with

    python scripts/artifacts.py publish [--set v1]         # creates or updates the release

The same bundles can also live on the Hugging Face Hub (models in a model repository, the
dataset in a dataset repository; both created private, the owner makes them public). The v2
dataset is derived from OpenSky data and is published only there, gated with manual approval;
the GitHub release carries the v2 models only:

    python scripts/artifacts.py publish --hub <owner>      # needs `hf auth login` once
    python scripts/artifacts.py fetch --hub <owner> [--models-only]

Hub repositories are <owner>/atc-trajectory-prediction (model) and
<owner>/atc-trajectory-prediction-data (dataset); --hub-model and --hub-dataset override
them. Their README cards come from docs/hub/. huggingface_hub is imported only for --hub.

Raw OpenSky ADS-B is NOT published by default: OpenSky's terms restrict redistributing
it, and on a public repository every release asset is public. `publish --adsb` includes it
for a private repository; `fetch --adsb` installs it from such a release or, with
`--from`, from a folder shared privately.

Every asset is listed in SHA256SUMS and verified after download, so a truncated download
fails loudly instead of loading a broken checkpoint. Fetching needs the GitHub CLI (`gh`),
logged in with access to the repository; without it, download the assets from the release
page by hand and run `fetch --from <folder>`.

Bundles, v2 (release data-v2):

  runs-v2.tar.gz      the main model and its baselines, first run: the final SBM (callsign
                      contrast and handoff phrases), CTP, history-only CTP, transcript-
                      conditioned CTP, anchors head
  runs-v2-demo.tar.gz the web demo (demo/): the extra checkpoints its selectors offer (initial
                      and no-history SBMs, sine-basis flow, trajectory-cluster flow, bank-angle
                      primitives) and its data bundle (demo/bundle: held-out windows with airline
                      callsigns, positions relative to the aircraft or scene, synthetic
                      transmissions; built by demo/build_bundle.py)
  runs-v2-all.tar.gz  every training run (seeds 0-2) of the final, callsign-contrast and
                      base SBMs, the CTP and the anchors head: the paper's run statistics
  dataset-v2.tar      data/v2: 1 Hz windows (arrays, windows.parquet), segments, the PCA
                      basis, the manifest, and the base and final message sets (tokens,
                      confidences)

  The fitted word-confidence file (data/v2/word_conf.json) derives from ATCO2, which is for
  internal research use, and is not shipped; message builds without it use a parametric
  fallback with the same shape (channel/word_conf.py).

Bundles, v1 (release data-v1):

  runs.tar.gz       runs/stage1.pt, runs/stage2.pt, runs/lambda.json
  dataset.tar       data/dataset/meta*.parquet, data/windows_raw/*

Both: adsb_<day>.parquet, the raw ADS-B the windows were cut from, one asset per day
(opt-in, private repositories only).

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
SETS = {
    "v2": ("data-v2", {
        "runs-v2.tar.gz": ["runs/v2/sbm_jacktol_ho_s0.pt", "runs/v2/ctp_basis_atoms_zfix_s0.pt",
                           "runs/v2/ctp_basis_none_s0.pt", "runs/v2/anchors_atoms_zfix_s0.pt",
                           "runs/v2/ctp_basis_text_s0.pt"],
        "runs-v2-all.tar.gz": [f"runs/v2/{m}_s{s}.pt" for m in ("sbm_jacktol_ho", "sbm_jacktol_cs",
                                                                "sbm_jacktol_anyorder", "ctp_basis_atoms_zfix",
                                                                "anchors_atoms_zfix") for s in (0, 1, 2)],
        "runs-v2-demo.tar.gz": ["runs/v2/sbm_jacktol_anyorder_s0.pt", "runs/v2/sbm_jacktol_nohist_s0.pt",
                                "runs/v2/ctp_sine_atoms_s0.pt", "runs/v2/kindctp_joint_atoms_s0.pt",
                                "runs/v2/prim_v3_atoms_s0.pt", "demo/bundle"],
        "dataset-v2.tar": ["data/v2/arrays", "data/v2/windows.parquet", "data/v2/segments.parquet",
                           "data/v2/basis.npz", "data/v2/manifest.json", "data/v2/messages_jacktol",
                           "data/v2/messages_jacktol_ho"],
    }),
    "v1": ("data-v1", {
        "runs.tar.gz": ["runs/stage1.pt", "runs/stage2.pt", "runs/lambda.json"],
        "dataset.tar": ["data/dataset/meta.parquet", "data/dataset/meta_grounded.parquet",
                        "data/windows_raw/windows.npz", "data/windows_raw/windows_meta.parquet"],
    }),
}
MODELS = {"v2": "runs-v2.tar.gz", "v1": "runs.tar.gz"}
OPTIONAL = {"runs-v2-all.tar.gz": "all_runs",     # built and fetched only with --all-runs
            "runs-v2-demo.tar.gz": "demo"}       # ... and with --demo
HUB_ONLY = {"v2": {"dataset-v2.tar"}}            # OpenSky-derived: gated Hub dataset, never a release
HUB_CARDS = {"model": ROOT / "docs/hub/model_card.md", "dataset": ROOT / "docs/hub/dataset_card.md"}
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


def bundles(a) -> dict:
    b = {k: v for k, v in SETS[a.set][1].items() if k not in OPTIONAL or getattr(a, OPTIONAL[k], False)}
    if getattr(a, "models_only", False):
        return {k: v for k, v in b.items() if k == MODELS[a.set] or k in OPTIONAL}
    return b


def hub_repos(a):
    model = a.hub_model or f"{a.hub}/atc-trajectory-prediction"
    dataset = a.hub_dataset or f"{a.hub}/atc-trajectory-prediction-data"
    return model, dataset


def hub_api():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("--hub needs huggingface_hub: `uv pip install huggingface_hub`, then `hf auth login`")
    return HfApi()


def hub_kind(name: str) -> str:
    """Model bundles go to the model repository, everything else to the dataset repository."""
    return "model" if name.startswith("runs") else "dataset"


def hub_publish(a, assets):
    api, (model, dataset) = hub_api(), hub_repos(a)
    repo = {"model": model, "dataset": dataset}
    for kind in ("model", "dataset"):
        mine = [p for p in assets if p.name != "SHA256SUMS" and hub_kind(p.name) == kind]
        if not mine:
            continue
        api.create_repo(repo[kind], repo_type=kind, private=True, exist_ok=True)
        if kind == "dataset":                       # access needs the owner's approval
            api.update_repo_settings(repo[kind], gated="manual", repo_type=kind)
        if HUB_CARDS[kind].exists():
            api.upload_file(path_or_fileobj=str(HUB_CARDS[kind]), path_in_repo="README.md",
                            repo_id=repo[kind], repo_type=kind)
        sums = STAGE / f"SHA256SUMS.{kind}"
        sums.write_text("".join(f"{sha256(p)}  {p.name}\n" for p in mine))
        for p in mine + [sums]:
            api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name if p != sums else "SHA256SUMS",
                            repo_id=repo[kind], repo_type=kind)
        print(f"uploaded {len(mine)} assets to https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo[kind]}")


def hub_fetch(a, want):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("--hub needs huggingface_hub: `uv pip install huggingface_hub`")
    model, dataset = hub_repos(a)
    lines = []
    for kind, repo in (("model", model), ("dataset", dataset)):
        names = [n for n in want if hub_kind(n) == kind]
        if not names:
            continue
        for n in names + ["SHA256SUMS"]:
            p = Path(hf_hub_download(repo, n, repo_type=kind, local_dir=STAGE / kind))
            if n == "SHA256SUMS":
                lines += p.read_text().splitlines()
            else:
                shutil.copy2(p, STAGE / n)
    (STAGE / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def publish(a):
    if not a.hub and HUB_ONLY.get(a.set) and not a.models_only:
        print("the v2 dataset goes only to the gated Hub dataset (--hub); publishing the models only")
        a.models_only = True
    STAGE.mkdir(exist_ok=True)
    assets = []
    for name, members in bundles(a).items():
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
    if a.hub:
        if a.adsb:
            sys.exit("raw ADS-B is never uploaded to the Hub")
        return hub_publish(a, assets)
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
    if not a.hub and not a.src and not a.models_only and HUB_ONLY.get(a.set):
        sys.exit("the v2 dataset is on the Hugging Face Hub, gated: request access on the dataset page,\n"
                 "then `hf auth login` and `fetch --hub <owner>`. The models alone: `fetch --models-only`.")
    want = list(bundles(a)) + (["adsb_*.parquet"] if a.adsb else [])
    if a.hub and not a.src:
        STAGE.mkdir(exist_ok=True)
        hub_fetch(a, [w for w in want if not w.startswith("adsb")])
    elif not a.src:
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

    for name in bundles(a):
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
    p.add_argument("--set", default="v2", choices=tuple(SETS))
    p.add_argument("--release", default=None, help="default: data-v2 or data-v1")
    p.add_argument("--adsb", action="store_true",
                   help="also publish the raw ADS-B; only on a private repository")
    p.add_argument("--dry-run", action="store_true", help="build the assets, upload nothing")
    p.add_argument("--all-runs", action="store_true", help="also every training run (runs-v2-all.tar.gz)")
    p.add_argument("--demo", action="store_true", help="also the web demo's models and data (runs-v2-demo.tar.gz)")
    p.add_argument("--models-only", action="store_true", help="only the model checkpoints, no dataset")
    f = sub.add_parser("fetch", help="download, verify and unpack the release assets")
    f.add_argument("--set", default="v2", choices=tuple(SETS))
    f.add_argument("--release", default=None, help="default: data-v2 or data-v1")
    f.add_argument("--models-only", action="store_true", help="only the model checkpoints")
    f.add_argument("--all-runs", action="store_true", help="also every training run (runs-v2-all.tar.gz)")
    f.add_argument("--demo", action="store_true", help="also the web demo's models and data (runs-v2-demo.tar.gz)")
    f.add_argument("--adsb", action="store_true", help="also install the raw ADS-B days")
    f.add_argument("--from", dest="src", default=None,
                   help="install from a folder of already-downloaded assets instead")
    for q in (p, f):
        q.add_argument("--hub", default=None, metavar="OWNER",
                       help="use the Hugging Face Hub (user or organization) instead of the GitHub release")
        q.add_argument("--hub-model", default=None, help="model repository id (default OWNER/atc-trajectory-prediction)")
        q.add_argument("--hub-dataset", default=None,
                       help="dataset repository id (default OWNER/atc-trajectory-prediction-data)")
    a = ap.parse_args()
    a.release = a.release or SETS[a.set][0]
    publish(a) if a.cmd == "publish" else fetch(a)


if __name__ == "__main__":
    main()
