"""Model execution for the demo: one aircraft window and one transcript in, belief and
predicted trajectories out; or a scene of aircraft competing for one transmission.

Inputs come from the demo bundle (build_bundle.py): derived 1 Hz windows of held-out
flights, never raw ADS-B. A transcript is a list of words with recognizer confidences.
Each semantic model reads it with its own vocabulary, so the same text can be sent to any
SBM variant. Every random draw is seeded by (seed, window, interpretation), as in the
paper's evaluation, so a result can be reproduced and shared.

Positions are returned in metres relative to the aircraft at the prediction time, east,
north and up, every 2 s from 0 to 60 s.
"""

import json
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch

from atc_forecast.data import atoms as A
from atc_forecast.data.frames import to_track
from atc_forecast.data.text import UNK, Vocab
from atc_forecast.evaluation.addressee_v2 import gated_status, resolve
from atc_forecast.evaluation.compose_v2 import per_message
from atc_forecast.evaluation.metrics_v2 import coverage, nested
from atc_forecast.inference.transcript import encode, encode_callsign, words_to_tokens
from atc_forecast.models.baselines import kinematic_rule, load_mdn
from atc_forecast.models.ctp import load_ctp, seeded_noise
from atc_forecast.models.sbm import load_sbm, window_generators

from . import variants as V

FT = 3.28084
EVERY = 2                                  # output every 2 s
MAX_K, MAX_S = 48, 12


class BadRequest(ValueError):
    pass


def readable(z):
    """'turn left heading 270, descend 5,000 ft' style text for a representation."""
    st = A.STATUS[int(z[0])]
    if st == "other":
        return "another aircraft"
    parts = []
    for slot, a in zip(("lateral", "vertical", "speed"), z[1:]):
        a = int(a)
        if a == A.NONE:
            continue
        if a == A.MAINTAIN:
            parts.append({"lateral": "maintain heading", "vertical": "maintain altitude",
                          "speed": "maintain speed"}[slot])
            continue
        tab = A.SLOTS[slot]
        v, s = int(tab["value"][a]), tab["sign"][a]
        if slot == "lateral":
            parts.append(f"{'left' if s < 0 else 'right'} {v:03d}")
        elif slot == "vertical":
            parts.append(f"{'climb' if s > 0 else 'descend'} {v:,} ft")
        else:
            parts.append(f"{'increase' if s > 0 else 'reduce'} {v} kt")
    if not parts:
        return "this aircraft, no motion instruction"
    return ", ".join(parts)


def branch(z):
    """Short name of a sampled representation for labelling its branch of the fan: the
    lateral and vertical instruction (speed only when it is the only content), so that
    representations differing in a small speed target share one branch."""
    if A.STATUS[int(z[0])] == "other":
        return "another aircraft"
    lat_ver = [int(a) != A.NONE for a in z[1:3]]
    if any(lat_ver):
        return readable([z[0], z[1], z[2], A.NONE])
    return readable(z)


class Engine:
    def __init__(self, bundle_dir, models_dir, device="cpu", cache_size=12):
        self.dir = Path(bundle_dir)
        self.models_dir = Path(models_dir)
        self.device = device
        f = np.load(self.dir / "windows.npz")
        self.arr = {k: f[k] for k in f.files}
        self.meta = json.loads((self.dir / "windows.json").read_text())
        self.scenes = json.loads((self.dir / "scenes.json").read_text())
        self.cases = json.loads((self.dir / "cases.json").read_text())
        self.text_vocab = Vocab.from_itos(json.loads((self.dir / "text_vocab.json").read_text()))
        self._models = OrderedDict()
        self.cache_size = cache_size

    # ------------------------------------------------------------------ models
    def _load(self, file, kind):
        key = (file, kind)
        if key in self._models:
            self._models.move_to_end(key)
            return self._models[key]
        path = self.models_dir / file
        if not path.exists():
            raise BadRequest(f"checkpoint {file} is not installed")
        if kind == "sbm":
            m, ck = load_sbm(path, self.device)
            extra = {"vocab": Vocab.from_itos(ck["vocab"]),
                     "prior": float(ck.get("addressed_prior", 0.8))}
        elif kind == "mdn":
            m, _ = load_mdn(path, self.device)
            extra = {}
        else:
            m, _ = load_ctp(path, self.device)
            extra = {}
        self._models[key] = (m, extra)
        while len(self._models) > self.cache_size:
            self._models.popitem(last=False)
        return self._models[key]

    def sbm(self, name):
        if name not in V.SBM:
            raise BadRequest(f"unknown SBM variant {name}")
        return self._load(V.SBM[name]["file"], "sbm")

    def ctp(self, name):
        if name in V.CTP:
            e = V.CTP[name]
            return self._load(e["file"], e["kind"])[0], e["kind"]
        if name in V.REFERENCE:
            return self._load(V.REFERENCE[name]["file"], "ctp")[0], "ctp"
        raise BadRequest(f"unknown trajectory variant {name}")

    # ------------------------------------------------------------------ inputs
    def window(self, wid):
        wid = int(wid)
        if not 0 <= wid < len(self.meta):
            raise BadRequest(f"unknown window {wid}")
        return wid

    def tokens(self, words, vocab):
        pairs = [(str(w.get("w", "")), w.get("c")) for w in words]
        toks, conf = words_to_tokens(pairs, 1.0)
        ids, kap, n_unk = encode(vocab, toks, conf)
        unknown = sorted({t for t in toks if vocab.stoi.get(t, UNK) == UNK})
        return ids, kap, n_unk, toks, unknown

    def batch(self, wids, words, vocab):
        wids = np.asarray(wids, int)
        a = self.arr
        t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt).to(self.device)
        ids, kap, n_unk, toks, unknown = self.tokens(words, vocab)
        n = len(wids)
        cs = np.stack([encode_callsign(vocab, self.meta[w]["callsign"]) for w in wids])
        b = {"hist": t(a["hist"][wids]), "ctx": t(a["ctx"][wids]), "center": t(a["center"][wids]),
             "state": t(a["state"][wids]), "wake": t(a["wake"][wids], torch.long),
             "engine": t(a["engine"][wids], torch.long), "n_engines": t(a["n_engines"][wids]),
             "tokens": t(np.repeat(ids[None], n, 0), torch.long),
             "kappa": t(np.repeat(kap[None], n, 0)), "cs_tokens": t(cs, torch.long),
             "z": t(a["z_true"][wids], torch.long), "fut_vel": t(a["fut_vel"][wids]),
             "window_id": a["seed_id"][wids]}
        info = {"n_tokens": int((ids > 0).sum()), "n_unknown": int(n_unk), "unknown": unknown,
                "tokens": toks}
        return b, info

    # ------------------------------------------------------------------ sampling
    def futures(self, model, kind, b, z, S, seed, nfe, method):
        """Velocities [B, K*S, 60, 3] for representations z [B, K, 4]."""
        K = z.shape[1]
        if kind == "mdn":
            gens = lambda k: [torch.Generator().manual_seed(hash((seed, int(w), k + 1)) & 0x7FFFFFFF)
                              for w in np.asarray(b["window_id"])]
            return torch.cat([model.sample({**b, "z": z[:, k]}, S, generators=gens(k))
                              for k in range(K)], 1).cpu().numpy()
        wid, shape = np.asarray(b["window_id"]), model.space.shape
        noise = torch.stack([seeded_noise(wid, S, shape, seed, k=k) for k in range(K)], 1)
        out = model.sample(b, S, noise=noise.reshape(-1, S, *shape), z=z, nfe=nfe, method=method)
        return out.reshape(out.shape[0], K * S, 60, 3).cpu().numpy()

    def unconditioned(self, model, b, n, seed, nfe, method):
        noise = seeded_noise(b["window_id"], n, model.space.shape, seed)
        return model.sample(b, n, noise=noise, nfe=nfe, method=method).cpu().numpy()

    def most_probable(self, sbm, b, enc, seed, K):
        """(z [B, 4], readings [B][(text, p, z)]) from beam search, or for the mixture SBM
        the most frequent of 4K draws with their frequencies."""
        if sbm.cfg.factorization == "mixture":
            zm = sbm.sample(b, 4 * K, window_generators(b["window_id"], seed, salt=2), enc=enc)
            zb, reads = [], []
            for x in zm.cpu().numpy():
                c = Counter(map(tuple, x)).most_common(5)
                zb.append(c[0][0])
                reads.append([(readable(z), n / len(x), list(z)) for z, n in c])
            return torch.as_tensor(np.array(zb)).to(self.device), reads, False
        z, lp, _ = sbm.beam(b, width=16, enc=enc)
        p = lp.exp().cpu().numpy()
        zn = z.cpu().numpy()
        reads = [[(readable(zn[i, j]), float(p[i, j]), zn[i, j].tolist()) for j in range(min(5, z.shape[1]))]
                 for i in range(len(zn))]
        return z[:, 0], reads, True

    # ------------------------------------------------------------------ outputs
    def enu(self, wid, V):
        """Velocities [N, 60, 3] in the track frame to ENU positions [N, 31, 3] (m)."""
        P = np.cumsum(V, axis=-2)
        psi = np.radians(self.meta[wid]["trk_deg"])
        E = to_track(P.reshape(1, -1, 3), np.array([psi])).reshape(P.shape)
        E = np.concatenate([np.zeros((E.shape[0], 1, 3)), E], 1)[:, ::EVERY]
        return np.round(E, 1)

    def scores(self, V, wid, groups=None):
        y = np.cumsum(self.arr["fut_vel"][wid], 0)[None]
        X = np.cumsum(V, axis=-2)[None]
        pm = per_message(X, y, groups)
        out = {"es_h_m": float(pm["energy_h"][0]) if "energy_h" in pm else None,
               "es_v_ft": float(pm["energy_v"][0]) if "energy_v" in pm else None,
               "ade_h_m": float(pm["ade_h"][0])}
        if X.shape[1] > 4:
            out["cov90"] = float(coverage(X, y, groups=groups)[1].mean())
        return out

    def groups(self, z):
        """Sampled representations grouped by value: [(text, count, z)] largest first."""
        c = Counter(map(tuple, np.asarray(z).tolist()))
        return [{"text": readable(k), "n": n, "z": list(k)} for k, n in c.most_common()]

    # ------------------------------------------------------------------ requests
    def check(self, K, S):
        if not (1 <= K <= MAX_K and 1 <= S <= MAX_S):
            raise BadRequest(f"K must be 1..{MAX_K} and S 1..{MAX_S}")

    @torch.no_grad()
    def predict(self, wid, words, sbm="final", ctp="flow_pca", sampler="euler12", K=12, S=6,
                seed=0, conditions=("composed", "most_probable", "adsb_only")):
        """One aircraft, one transcript."""
        t_start = time.perf_counter()
        wid = self.window(wid)
        self.check(K, S)
        if sampler not in V.SAMPLERS:
            raise BadRequest(f"unknown sampler {sampler}")
        sp = V.SAMPLERS[sampler]
        nfe, method = sp["nfe"], sp["method"]
        sm, sx = self.sbm(sbm)
        b, info = self.batch([wid], words, sx["vocab"])
        enc = sm.encode(b)
        zs = sm.sample(b, K, window_generators(b["window_id"], seed), enc=enc)
        zb, reads, exact = self.most_probable(sm, b, enc, seed, K)
        q = sm.status_probs(b, enc=enc).cpu().numpy()[0]
        cm, kind = self.ctp(ctp)
        g = nested(K, S)
        out = {"window": wid, "input": info,
               "status": dict(zip(A.STATUS, map(float, q))),
               "readings": [{"text": r[0], "p": r[1], "z": [int(x) for x in r[2]]} for r in reads[0]],
               "readings_exact": exact,
               "samples": self.groups(zs[0].cpu().numpy()),
               "branch_of_sample": [branch(x) for x in zs[0].cpu().numpy()],
               "label": readable(self.arr["z_true"][wid]),
               "flown": self.enu(wid, self.arr["fut_vel"][wid][None])[0].tolist(),
               "predictions": {}}
        n = K * S
        for c in conditions:
            if c == "composed":
                Vc = self.futures(cm, kind, b, zs, S, seed, nfe, method)[0]
                gr = g
            elif c == "most_probable":
                Vc = self.futures(cm, kind, b, zb[:, None].expand(-1, K, -1), S, seed, nfe, method)[0]
                gr = None
            elif c == "label":
                zt = b["z"][:, None].expand(-1, K, -1)
                Vc = self.futures(cm, kind, b, zt, S, seed, nfe, method)[0]
                gr = None
            elif c == "adsb_only":
                Vc = self.unconditioned(self.ctp("adsb_only")[0], b, n, seed, nfe, method)[0]
                gr = None
            elif c in ("transcript", "transcript_sbm_init"):
                bt, _ = self.batch([wid], words, self.text_vocab)
                Vc = self.unconditioned(self.ctp(c)[0], bt, n, seed, nfe, method)[0]
                gr = None
            elif c == "kinematic":
                Vc = kinematic_rule(zb.cpu().numpy(), b["state"].cpu().numpy(),
                                    b["center"].cpu().numpy())[0][None]
                gr = None
            else:
                raise BadRequest(f"unknown condition {c}")
            item = {"tracks": self.enu(wid, Vc).tolist(), "scores": self.scores(Vc, wid, gr)}
            if c == "composed":
                item["group_of_track"] = g.tolist()
            out["predictions"][c] = item
        out["ms"] = round(1000 * (time.perf_counter() - t_start))
        return out

    @torch.no_grad()
    def scene(self, wids, words, sbm="final", ctp="flow_pca", sampler="euler12", K=12, S=6,
              seed=0, mode="normalized", routing=False, out_weight=1.0, adsb=True):
        """Several aircraft hear one transmission. mode 'normalized': recipient probabilities
        across the aircraft (Appendix H); 'independent': each aircraft alone. routing: samples
        with no motion content use the ADS-B-only predictor (Appendix H.3). adsb: also return
        each aircraft's ADS-B-only prediction, the baseline that ignores the radio."""
        t_start = time.perf_counter()
        wids = [self.window(w) for w in wids]
        if not 1 <= len(wids) <= 16:
            raise BadRequest("a scene has 1 to 16 aircraft")
        self.check(K, S)
        sp = V.SAMPLERS.get(sampler) or V.SAMPLERS["euler12"]
        sm, sx = self.sbm(sbm)
        b, info = self.batch(wids, words, sx["vocab"])
        enc = sm.encode(b)
        q = sm.status_probs(b, enc=enc).cpu().numpy()
        r = q[:, :2].sum(1)
        if mode == "normalized":
            if sm.cfg.factorization == "mixture":
                raise BadRequest("the mixture-over-fields SBM has no status-first order, so it cannot take "
                                 "recipient probabilities across aircraft; untick that option or pick another SBM")
            p = resolve(r, out_weight, sx["prior"])
            p_status = torch.as_tensor(gated_status(q, p), dtype=torch.float32)
        elif mode == "independent":
            p, p_status = r, None
        else:
            raise BadRequest(f"unknown mode {mode}")
        zs = sm.sample(b, K, window_generators(b["window_id"], seed), enc=enc, p_status=p_status)
        cm, kind = self.ctp(ctp)
        Vc = self.futures(cm, kind, b, zs, S, seed, sp["nfe"], sp["method"])
        zn = zs.cpu().numpy()
        routed = np.zeros(zn.shape[:2], bool)
        H = None
        if adsb or routing:
            H = self.unconditioned(self.ctp("adsb_only")[0], b, K * S, seed, sp["nfe"], sp["method"])
        if routing:
            routed = (zn[..., 1:] == A.NONE).all(-1)
            if routed.any():
                mask = np.repeat(routed, S, 1)
                Vc = np.where(mask[..., None, None], H, Vc)
        g = nested(K, S)
        aircraft = []
        for i, w in enumerate(wids):
            aircraft.append({"window": w, "callsign": self.meta[w]["callsign"],
                             "p_addressed": float(p[i]), "status": dict(zip(A.STATUS, map(float, q[i]))),
                             "samples": self.groups(zn[i]), "routed": int(routed[i].sum()),
                             "branch_of_sample": [branch(x) for x in zn[i]],
                             "tracks": self.enu(w, Vc[i]).tolist(), "group_of_track": g.tolist(),
                             "scores": self.scores(Vc[i], w, g),
                             **({"adsb_only": {"tracks": self.enu(w, H[i]).tolist(), "scores": self.scores(H[i], w)}}
                                if adsb else {})})
        return {"mode": mode, "routing": routing, "input": info, "aircraft": aircraft,
                "p_outside": float(max(0.0, 1 - p.sum())) if mode == "normalized" else None,
                "ms": round(1000 * (time.perf_counter() - t_start))}
