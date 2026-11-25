"""Static version of the demo: the same front end, reading precomputed predictions instead of a
model service, for a website (GitHub Pages) with no server and no cost.

    uv run python demo/build_static.py [--out output/static_demo] [--scenes 15]

Precomputed, with the default sampler (12 Euler steps, K = 12, S = 6, seed 0):
  - every case of "Try a transmission" under every shown SBM × trajectory predictor pair
    (composed, most probable instruction, semantic label, kinematic rule), plus the ADS-B-only
    and transcript-conditioned references once per case;
  - the first --scenes live-sector scenes with the main models, recipient identification across
    the aircraft, with and without ADS-B-only routing.
Typed transmissions, word edits and other settings need the model service (run locally).
Trajectories are thinned to 36 per prediction (half of each instruction group) and rounded to
10 m; scores are computed on all 72 before thinning.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo.server import variants as V
from demo.server.engine import Engine

KEEP = 36


def thin(item, grouped):
    """Keep 36 of 72 tracks: every other trajectory within each instruction group."""
    T = item["tracks"]
    if len(T) <= KEEP:
        idx = list(range(len(T)))
    elif grouped:
        g = item["group_of_track"]
        seen, idx = {}, []
        for i, k in enumerate(g):
            j = seen.get(k, 0)
            seen[k] = j + 1
            if j % 2 == 0:
                idx.append(i)
    else:
        idx = list(range(0, len(T), max(1, len(T) // KEEP)))[:KEEP]
    out = dict(item)
    out["tracks"] = (np.round(np.asarray(T)[idx] / 10) * 10).astype(int).tolist()
    if grouped:
        out["group_of_track"] = [item["group_of_track"][i] for i in idx]
    return out


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "output/static_demo"))
    ap.add_argument("--scenes", type=int, default=15)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    e = Engine(ROOT / "demo/bundle", ROOT / "runs/v2")
    avail = {f.name for f in (ROOT / "runs/v2").glob("*.pt")}
    var = V.public(available=avail)
    var["samplers"] = {"euler12": var["samplers"]["euler12"]}
    var["static"] = True
    write(out / "data/variants.json", var)

    scenes = e.scenes[:a.scenes]
    used = {c["window"] for c in e.cases} | {x["window"] for s in scenes for x in s["aircraft"]}
    used |= {t["recipient"] for s in scenes for t in s["transmissions"] if t["recipient"] is not None}
    meta = [m if i in used else None for i, m in enumerate(e.meta)]
    write(out / "data/cases.json", {"cases": e.cases, "scenes": scenes, "windows": meta})

    # ---------------------------------------------------------------- try a transmission
    for ci, c in enumerate(e.cases):
        w, words = c["window"], c["message"]["words"]
        ref = e.predict(w, words, conditions=("adsb_only", "transcript"))
        ref["predictions"] = {k: thin(v, False) for k, v in ref["predictions"].items()}
        write(out / f"data/try/{ci}/ref.json", ref)
        for sbm in var["sbm"]:
            for ctp in var["ctp"]:
                r = e.predict(w, words, sbm=sbm, ctp=ctp,
                              conditions=("composed", "most_probable", "label", "kinematic"))
                r["predictions"] = {k: thin(v, k == "composed") for k, v in r["predictions"].items()}
                write(out / f"data/try/{ci}/{sbm}__{ctp}.json", r)
        print(f"case {ci + 1}/{len(e.cases)}", flush=True)

    # ---------------------------------------------------------------- live sector
    for si, s in enumerate(scenes):
        wids = [x["window"] for x in s["aircraft"]]
        for ti, t in enumerate(s["transmissions"]):
            for routing in (False, True):
                r = e.scene(wids, t["message"]["words"], routing=routing)
                for x in r["aircraft"]:
                    keep = KEEP if x["p_addressed"] >= 0.5 else 12
                    tr = np.asarray(x["tracks"])
                    idx = np.linspace(0, len(tr) - 1, keep).astype(int) if keep < KEEP else None
                    if idx is None:
                        th = thin(x, True)
                        x["tracks"], x["group_of_track"] = th["tracks"], th["group_of_track"]
                    else:
                        x["tracks"] = (np.round(tr[idx] / 10) * 10).astype(int).tolist()
                        x["group_of_track"] = [x["group_of_track"][i] for i in idx]
                write(out / f"data/scene/{si}/{ti}_{int(routing)}.json", r)
        print(f"scene {si + 1}/{len(scenes)}", flush=True)

    # ---------------------------------------------------------------- front end
    web = ROOT / "demo/web"
    html = (web / "index.html").read_text().replace('href="/static/style.css"', 'href="style.css"') \
        .replace('src="/static/app.js"', 'src="app.js"').replace("<body>", '<body class="static">')
    (out / "index.html").write_text(html)
    shutil.copy2(web / "app.js", out / "app.js")
    shutil.copy2(web / "style.css", out / "style.css")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"static demo: {out} ({size / 1e6:.1f} MB, {sum(1 for _ in out.rglob('*.json'))} files)")


if __name__ == "__main__":
    main()
