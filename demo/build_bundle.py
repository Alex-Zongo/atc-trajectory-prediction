"""Build the demo's data bundle from the v2 dataset: held-out windows, their synthetic
transmissions, and multi-aircraft scenes.

    uv run python demo/build_bundle.py [--scenes 40] [--cases 24] [--out demo/bundle]

What goes in, and what does not:
  - windows of the test split (recipients and single-aircraft cases) and of the test or
    validation split (other aircraft in a scene), so no prediction is made on training data;
  - airline callsigns only (no N-numbers, which identify private owners);
  - the derived 1 Hz model inputs and the recorded 60 s future, as in the gated dataset;
  - positions only relative to the aircraft, or for scenes relative to the scene centre
    (km), with times relative to the scene; no absolute position, time, day or icao24;
  - synthetic transmissions from the final message set (no recorded speech of any kind).

Scene positions are read from the raw ADS-B day files at build time, which stay local.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from atc_forecast.data.frames import to_track
from atc_forecast.data.text import Vocab
from atc_forecast.inference.transcript import encode, words_to_tokens
from atc_forecast.training.data_v2 import MessageStore, WindowStore
from demo.server.engine import readable

AIRLINE = re.compile(r"^[A-Z]{3}\d")
R_EARTH = 6_371_000.0
PAPER_CASES = ROOT / "figures/v2/data/cases.json"
CONSTRUCTION_WINDOW = 47337          # the flight of Fig. 13 and Table 9


def itos_of(ms):
    v = ms.vocab
    return v if isinstance(v, list) else v.get("itos", v)


def message(ms, itos, row):
    ids = np.asarray(ms.tokens[row])
    kap = np.asarray(ms.kappa[row]) / 255.0
    words = [{"w": itos[i], "c": round(float(k), 3)} for i, k in zip(ids, kap) if i > 0]
    return {"row": int(row), "variant": str(ms.variant[row]),
            "said": str(ms.msgs.loc[row, "said"]).strip() if "said" in ms.msgs else "",
            "words": words, "label": readable(ms.z[row])}


def roundtrip_ok(vocab, ms, row):
    ids = np.asarray(ms.tokens[row])
    itos = vocab.itos
    toks = [itos[i] for i in ids if i > 0]
    conf = (np.asarray(ms.kappa[row])[:len(toks)] / 255.0).tolist()
    t2, c2 = words_to_tokens(list(zip(toks, conf)), 1.0)
    e, _, _ = encode(vocab, t2, c2)
    return np.array_equal(e[:len(ids)], ids[:len(e)])


def positions(day, members, ws):
    """Per member window: (lat, lon, alt_ft) at t0 and the 30 s history every 2 s, from the
    raw reports, linearly interpolated."""
    m = ws.meta
    ic = {m.icao24[w] for w in members}
    raw = pd.read_parquet(ROOT / f"data/adsb/adsb_{day}.parquet",
                          columns=["time_position", "icao24", "latitude", "longitude", "baro_altitude"],
                          filters=[("icao24", "in", sorted(ic))])
    raw = raw.dropna().sort_values("time_position").drop_duplicates(["icao24", "time_position"])
    out = {}
    for w in members:
        r = raw[raw.icao24 == m.icao24[w]]
        t0 = m.t0[w]
        ts = t0 + np.arange(-30, 1, 2.0)
        tp = r.time_position.to_numpy(float)
        if len(tp) < 5 or tp[0] > ts[0] + 4 or tp[-1] < t0 - 4:
            continue
        lat = np.interp(ts, tp, r.latitude.to_numpy())
        lon = np.interp(ts, tp, r.longitude.to_numpy())
        alt = np.interp(ts, tp, r.baro_altitude.to_numpy()) * 3.28084
        out[w] = (lat, lon, alt)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data/v2"))
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--cases", type=int, default=24)
    ap.add_argument("--max-dt", type=float, default=30.0)
    ap.add_argument("--radius-km", type=float, default=35.0, help="scene members within this distance of the recipient")
    ap.add_argument("--out", default=str(ROOT / "demo/bundle"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ws = WindowStore(a.data)
    ms = MessageStore(a.data, "jacktol_ho", ws)
    itos = itos_of(ms)
    vocab = Vocab.from_itos(itos)
    m = ws.meta
    airline = m.callsign.fillna("").str.match(AIRLINE).to_numpy()
    split = m.split.to_numpy()
    held = ws.ok & airline & np.isin(split, ["test", "val"])
    test = held & (split == "test")
    M = ms.msgs
    first = M[M.draw == 0]
    by = {(int(w), v): int(r) for r, w, v in zip(first.index, first.window_id, first.variant)}
    own = np.array([(w, "own") in by for w in range(len(m))])

    chosen = {}                        # original window id -> bundle index
    order = []

    def add(w):
        w = int(w)
        if w not in chosen:
            chosen[w] = len(order)
            order.append(w)
        return chosen[w]

    def msgs_of(w):
        return [message(ms, itos, by[(w, v)]) for v in ("own", "maintain", "own_other", "other", "swap")
                if (w, v) in by]

    # ---------------------------------------------------------------- scenes
    day, t0, ic = m.day.to_numpy(), m.t0.to_numpy(), m.icao24.to_numpy()
    seeds = np.where(test & own & (m.kind == "maneuver").to_numpy())[0]
    rng.shuffle(seeds)
    scenes, used = [], set()
    for s in seeds:
        if len(scenes) >= a.scenes:
            break
        if ic[s] in used:
            continue
        cand = np.where(held & (day == day[s]) & (np.abs(t0 - t0[s]) <= a.max_dt))[0]
        cand = cand[np.argsort(np.abs(t0[cand] - t0[s]))][:60]
        pos = positions(day[s], [s] + [c for c in cand if c != s], ws)
        if s not in pos:
            continue
        la, lo = pos[s][0][-1], pos[s][1][-1]
        def near(w, pos=pos, la=la, lo=lo):
            return np.hypot((pos[w][0][-1] - la) * 111.2,
                            (pos[w][1][-1] - lo) * 111.2 * np.cos(np.radians(la))) <= a.radius_km
        members, seen = [s], {ic[s]}
        for c in cand:
            if c in pos and ic[c] not in seen and near(c) and \
                    (own[c] or (c, "maintain") in by or (c, "own_other") in by):
                seen.add(ic[c])
                members.append(c)
            if len(members) == 8:
                break
        if len(members) < 4:
            continue
        lat0 = np.mean([pos[w][0][-1] for w in members])
        lon0 = np.mean([pos[w][1][-1] for w in members])
        kx = np.radians(1) * R_EARTH * np.cos(np.radians(lat0)) / 1000
        ky = np.radians(1) * R_EARTH / 1000
        tref = t0[s]
        sc = {"id": f"scene{len(scenes) + 1:02d}", "aircraft": [], "transmissions": []}
        for w in members:
            lat, lon, alt = pos[w]
            e, n = (lon - lon0) * kx, (lat - lat0) * ky
            sc["aircraft"].append({"window": add(w), "t": round(float(t0[w] - tref), 1),
                                   "hist_km": np.round(np.stack([e, n, alt], 1), 3).tolist()})
            v = "own" if own[w] else ("maintain" if (w, "maintain") in by else "own_other")
            sc["transmissions"].append({"t": round(float(t0[w] - tref), 1), "recipient": chosen[w],
                                        "message": message(ms, itos, by[(w, v)])})
        if (s, "other") in by:                     # one transmission for an aircraft not in the scene
            sc["transmissions"].append({"t": round(float(t0[s] - tref) + 4, 1), "recipient": None,
                                        "message": message(ms, itos, by[(s, "other")])})
        sc["transmissions"].sort(key=lambda x: x["t"])
        used |= {ic[w] for w in members}
        scenes.append(sc)
    print(f"{len(scenes)} scenes, {np.mean([len(x['aircraft']) for x in scenes]):.1f} aircraft on average")

    # ---------------------------------------------------------------- single-aircraft cases
    cases = []
    if PAPER_CASES.exists():
        names = {"rescue": "Fig. 6(a): misheard callsign, lost turn direction",
                 "confident": "Fig. 6(b): self-correction, read with confidence",
                 "hedge": "Fig. 6(c): own instruction, callsign lost",
                 "addressee": "Fig. 6(d): another aircraft's instruction, callsign lost"}
        ms0 = MessageStore(a.data, "jacktol", ws)          # the paper's cases index the initial set
        itos0 = itos_of(ms0)
        for c in json.loads(PAPER_CASES.read_text()):
            w = int(ms0.window_id[c["row"]])
            if not airline[w] or not test[w]:
                continue
            if not roundtrip_ok(Vocab.from_itos(itos0), ms0, c["row"]):
                print("case", c["case"], "does not round-trip; skipped")
                continue
            cases.append({"id": c["case"], "title": names.get(c["case"], c["case"]), "window": add(w),
                          "message": message(ms0, itos0, c["row"])})
    if test[CONSTRUCTION_WINDOW]:
        for v, title in (("own", "Construction flight (Fig. 13): left turn to 070"),
                         ("swap", "Construction flight: callsign contrast"),
                         ("own_other", "Construction flight: handoff, no motion")):
            if (CONSTRUCTION_WINDOW, v) in by:
                cases.append({"id": f"construction_{v}", "title": title, "window": add(CONSTRUCTION_WINDOW),
                              "message": message(ms, itos, by[(CONSTRUCTION_WINDOW, v)])})
    plan = (("own", 12, "Own-aircraft maneuver"), ("maintain", 4, "Maintain instruction"),
            ("own_other", 4, "No motion instruction"), ("other", 4, "Another aircraft"))
    pool = np.where(test)[0]
    for v, n, title in plan:
        ok = [w for w in rng.permutation(pool) if (w, v) in by][:max(1, n * a.cases // 24)]
        for w in ok:
            msg = message(ms, itos, by[(w, v)])
            cases.append({"id": f"{v}_{w}", "title": f"{title}: {msg['label']}", "window": add(w), "message": msg})
    print(f"{len(cases)} cases")

    # ---------------------------------------------------------------- windows
    idx = np.array(order)
    arr = {"hist": ws.hist[idx], "ctx": ws.ctx[idx], "center": ws.center[idx], "state": ws.state[idx],
           "wake": ws.wake[idx], "engine": ws.engine[idx], "n_engines": ws.n_engines[idx],
           "fut_vel": ws.fut_vel[idx], "z_true": ws.z_true[idx], "seed_id": idx.astype(np.int64)}
    np.savez_compressed(out / "windows.npz", **{k: np.asarray(v) for k, v in arr.items()})
    meta = []
    bad = 0
    for i, w in enumerate(idx):
        P = np.cumsum(ws.fut_vel[w], 0)
        E = to_track(P[None], np.array([np.radians(m.trk_deg[w])]))[0]
        E = np.concatenate([[[0, 0, 0]], E])[::2]
        mm = msgs_of(w)
        bad += sum(not roundtrip_ok(vocab, ms, x["row"]) for x in mm)
        meta.append({"i": i, "callsign": m.callsign[w], "kind": m.kind[w], "split": split[w],
                     "trk_deg": round(float(m.trk_deg[w]), 1), "alt_ft": round(float(m.alt_ft[w])),
                     "gs_kt": round(float(m.gs_kt[w])), "label": readable(ws.z_true[w]),
                     "flown_m": np.round(E, 1).tolist(), "messages": mm})
    (out / "windows.json").write_text(json.dumps(meta))
    (out / "scenes.json").write_text(json.dumps(scenes))
    (out / "cases.json").write_text(json.dumps(cases))
    tv = json.loads((Path(a.data) / "messages_jacktol/vocab.json").read_text())
    (out / "text_vocab.json").write_text(json.dumps(tv if isinstance(tv, list) else tv.get("itos", tv)))
    print(f"{len(idx)} windows; token round-trip mismatches: {bad}")
    print("bundle size", sum(f.stat().st_size for f in out.iterdir()) // 1024, "KB ->", out)


if __name__ == "__main__":
    main()
