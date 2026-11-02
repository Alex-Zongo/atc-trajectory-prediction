"""The v2 model on real inputs: ADS-B plus a recognizer transcript to forecast futures.

    from atc_forecast.inference import Forecaster, load_states
    fc = Forecaster.load("runs/v2/sbm_jacktol_anyorder_s0.pt",
                         "runs/v2/ctp_basis_atoms_zfix_s0.pt")
    states = load_states("adsb/*.parquet")                   # OpenSky or adsb.lol
    out = fc.forecast(states, icao24="a1b2c3", t0=1767225600.0,
                      words=[{"word": "united", "probability": 0.98}, ...])
    out["readings"]      # the SBM's most likely interpretations, with probabilities
    out["lat"], out["lon"], out["alt_ft"]    # [K*S, 60] composed future positions

The steps are the training pipeline's, in order:

  window    data/resample.py on the aircraft's raw state vectors at t0 (the end of the
            transmission): 30 s of filtered history; with future=True also the recorded
            60 s, for evaluation. frames.make_features turns it into model inputs.
  aircraft  wake and engine class from the OpenSky aircraft database by icao24 (unknown
            when the aircraft is not listed; the models have an "unknown" embedding).
  message   inference/transcript.py: spoken-form tokens with per-word confidence, and
            the aircraft's own ADS-B callsign as the SBM's exact callsign input.
  compose   K interpretations from the SBM, S futures from the CTP under each (12 x 6).
            Optional: who the message is for over several aircraft (addressee), with
            evaluation/addressee_v2.resolve.

Futures come back as 1 s velocities in the track frame [B, N, 60, 3] (what the metrics
read) and as positions; to_geo turns positions into latitude, longitude and altitude.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..collect.aircraft import DEFAULT_CACHE, load_aircraft_db
from ..data import atoms as A
from ..data.frames import make_features, to_track
from ..data.resample import WindowSpec, windows_from_adsb
from ..data.text import Vocab
from ..models.ctp import load_ctp, seeded_noise
from ..models.sbm import load_sbm, window_generators
from ..training.common import pick_device
from ..units import DEG_TO_M, ENGINE_VOCAB, M_TO_FT, WAKE_VOCAB
from .adsb import callsign_at
from .transcript import encode, encode_callsign, words_to_tokens

SLOT_NAMES = ("status", "lateral", "vertical", "speed")


def describe(z) -> str:
    """Readable interpretation: 'own_motion | R270 | descend 5,000 | -'."""
    return " | ".join(A.describe(s, int(a)) for s, a in zip(SLOT_NAMES, z))


@dataclass
class Windows:
    """Model inputs for a set of (aircraft, t0) requests, as numpy arrays."""
    hist: np.ndarray
    ctx: np.ndarray
    center: np.ndarray
    state: np.ndarray            # track deg, altitude ft, ground speed kt
    wake: np.ndarray
    engine: np.ndarray
    n_engines: np.ndarray
    lat0: np.ndarray
    lon0: np.ndarray
    z0: np.ndarray               # pressure altitude at t0 (m)
    psi: np.ndarray              # track at t0 (rad)
    icao24: np.ndarray
    callsign: np.ndarray
    request: np.ndarray          # row of the request table each window answers
    fut_vel: np.ndarray | None   # recorded future, when built with future=True
    raw_future: list

    def __len__(self):
        return len(self.request)


class Forecaster:
    def __init__(self, sbm, sbm_ck, ctp, hist_only=None, device="cpu",
                 aircraft_db=DEFAULT_CACHE, spec: WindowSpec = WindowSpec()):
        self.sbm, self.ctp, self.hist_only = sbm, ctp, hist_only
        self.vocab = Vocab.from_itos(sbm_ck["vocab"])
        self.device, self.spec = device, spec
        self._db_path, self._db = aircraft_db, None

    @classmethod
    def load(cls, sbm, ctp, hist_only=None, device=None, **kw):
        dev = pick_device(device)
        s, ck = load_sbm(sbm, dev)
        return cls(s, ck, load_ctp(ctp, dev)[0],
                   load_ctp(hist_only, dev)[0] if hist_only else None, dev, **kw)

    # ------------------------------------------------------------------ inputs
    def aircraft(self, icao24s) -> pd.DataFrame:
        df = pd.DataFrame({"icao24": pd.Series(icao24s, dtype=str).str.lower()})
        if self._db is None and self._db_path is not None:
            self._db = load_aircraft_db(self._db_path).drop_duplicates("icao24")
        if self._db is not None:
            df = df.merge(self._db, on="icao24", how="left")
        for c in ("wake", "engine_class"):
            df[c] = df[c].fillna("unknown") if c in df else "unknown"
        df["n_engines"] = df["n_engines"].fillna(1.0) if "n_engines" in df else 1.0
        return df

    def windows(self, states: pd.DataFrame, requests: pd.DataFrame, future: bool = True):
        """Windows for requests (columns icao24, t0). Returns (Windows or None, report),
        report holding one row per request with its rejection reason (None if kept)."""
        w, report = windows_from_adsb(states, requests[["icao24", "t0"]], self.spec, future)
        if w is None:
            return None, report
        f = make_features(w)
        req = requests.reset_index(drop=True).iloc[w["request_index"]]
        ac = self.aircraft(req["icao24"].to_numpy())
        wv, ev = {v: i for i, v in enumerate(WAKE_VOCAB)}, {v: i for i, v in enumerate(ENGINE_VOCAB)}
        cs = [c if isinstance(c, str) and c else callsign_at(states, ic, t)
              for c, ic, t in zip(req.get("callsign", pd.Series([None] * len(req))).to_numpy(),
                                  req["icao24"].to_numpy(), req["t0"].to_numpy())]
        st = f["state"]
        v0 = w["hist_vel"][:, -1]
        return Windows(
            hist=f["hist"], ctx=f["ctx"], center=f["center"],
            state=np.stack([st["trk_deg"], st["alt_ft"], st["gs_kt"]], 1).astype(np.float32),
            wake=ac["wake"].map(wv).fillna(wv["unknown"]).to_numpy(np.int64),
            engine=ac["engine_class"].map(ev).fillna(ev["unknown"]).to_numpy(np.int64),
            n_engines=ac["n_engines"].to_numpy(np.float32),
            lat0=w["lat0"], lon0=w["lon0"], z0=w["z0"], psi=np.arctan2(v0[:, 0], v0[:, 1]),
            icao24=req["icao24"].to_numpy().astype(str), callsign=np.array(cs, dtype=object),
            request=np.asarray(w["request_index"]),
            fut_vel=f["fut_vel"] if future else None, raw_future=f["raw_future"]), report

    def batch(self, win: Windows, rows, messages, seed_ids=None):
        """Torch batch for windows `rows`, each hearing messages[i] = (words or tokens)."""
        rows = np.asarray(rows)
        t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt).to(self.device)
        toks, kaps = [], []
        for m in messages:
            ids, kap, _ = self.encode(m)
            toks.append(ids)
            kaps.append(kap)
        b = {"hist": t(win.hist[rows]), "ctx": t(win.ctx[rows]), "center": t(win.center[rows]),
             "state": t(win.state[rows]), "wake": t(win.wake[rows], torch.long),
             "engine": t(win.engine[rows], torch.long), "n_engines": t(win.n_engines[rows]),
             "tokens": t(np.stack(toks), torch.long), "kappa": t(np.stack(kaps)),
             "cs_tokens": t(np.stack([encode_callsign(self.vocab, c) for c in win.callsign[rows]]),
                            torch.long),
             "z": t(np.zeros((len(rows), 4)), torch.long),
             "window_id": np.asarray(rows if seed_ids is None else seed_ids)}
        if win.fut_vel is not None:
            b["fut_vel"] = t(win.fut_vel[rows])
        return b

    def encode(self, message, default_conf: float = 1.0):
        """(ids, kappa, n_unknown) for recognizer words (dicts or (word, prob) pairs) or
        plain text (every word at default_conf)."""
        return encode(self.vocab, *words_to_tokens(message, default_conf))

    # ------------------------------------------------------------------ models
    @torch.no_grad()
    def readings(self, b, top: int = 5, width: int = 16):
        """The SBM's most likely interpretations: [B] lists of (text, probability, z)."""
        z, lp, _ = self.sbm.beam(b, width=width)
        z, p = z.cpu().numpy(), lp.exp().cpu().numpy()
        return [[(describe(z[i, j]), float(p[i, j]), z[i, j].tolist()) for j in range(min(top, z.shape[1]))]
                for i in range(len(z))]

    @torch.no_grad()
    def status(self, b):
        """q(status | O, X) [B, 3]: own motion, own non-motion, other."""
        return self.sbm.status_probs(b).cpu().numpy()

    @torch.no_grad()
    def interpretations(self, b, K: int = 12, seed: int = 0, p_status=None):
        p = None if p_status is None else torch.as_tensor(p_status, dtype=torch.float32)
        return self.sbm.sample(b, K, window_generators(b["window_id"], seed), p_status=p)

    @torch.no_grad()
    def futures(self, b, z, S: int = 6, seed: int = 0, nfe: int = 12, method: str = "euler"):
        """CTP futures [B, K*S, 60, 3] (velocities, m/s, track frame) for z [B, K, 4]."""
        K, wid, shape = z.shape[1], np.asarray(b["window_id"]), self.ctp.space.shape
        noise = torch.stack([seeded_noise(wid, S, shape, seed, k=k) for k in range(K)], 1)
        V = self.ctp.sample(b, S, noise=noise.reshape(-1, S, *shape), z=z, nfe=nfe, method=method)
        return V.reshape(V.shape[0], K * S, 60, 3).cpu().numpy()

    @torch.no_grad()
    def history_only(self, b, n: int = 72, seed: int = 0, nfe: int = 12, method: str = "euler"):
        m = self.hist_only
        noise = seeded_noise(b["window_id"], n, m.space.shape, seed)
        return m.sample(b, n, noise=noise, nfe=nfe, method=method).cpu().numpy()

    def compose(self, b, K=12, S=6, seed=0, nfe=12, method="euler", p_status=None):
        """The main prediction: K interpretations, S futures each. Returns (V, z)."""
        z = self.interpretations(b, K, seed, p_status)
        return self.futures(b, z, S, seed, nfe, method), z.cpu().numpy()

    # ------------------------------------------------------------------ outputs
    @staticmethod
    def positions(V):
        return np.cumsum(V, axis=-2)

    @staticmethod
    def to_geo(P, win: Windows, rows):
        """Track-frame positions [B, N, T, 3] (m, relative to the present) to latitude,
        longitude (deg) and pressure altitude (ft), each [B, N, T]."""
        rows = np.asarray(rows)
        B = P.shape[0]
        flat = P.reshape(B, -1, 3)
        enu = to_track(flat, win.psi[rows]).reshape(P.shape)       # the rotation is its own inverse
        lat0, lon0 = win.lat0[rows][:, None, None], win.lon0[rows][:, None, None]
        lat = lat0 + enu[..., 1] / DEG_TO_M
        lon = lon0 + enu[..., 0] / (DEG_TO_M * np.cos(np.radians(lat0)))
        alt = (win.z0[rows][:, None, None] + enu[..., 2]) * M_TO_FT
        return lat, lon, alt

    def forecast(self, states, icao24, t0, words, K=12, S=6, seed=0, nfe=12, method="euler",
                 future=False, top=5):
        """One transmission heard by one aircraft: interpretations and composed futures."""
        req = pd.DataFrame({"icao24": [str(icao24).lower()], "t0": [float(t0)]})
        win, report = self.windows(states, req, future=future)
        if win is None:
            return {"accepted": False, "reason": report["reason"].iloc[0]}
        b = self.batch(win, [0], [words])
        V, z = self.compose(b, K, S, seed, nfe, method)
        lat, lon, alt = self.to_geo(self.positions(V), win, [0])
        ids, _, n_unk = self.encode(words)
        return {"accepted": True, "readings": self.readings(b, top)[0],
                "status": dict(zip(A.STATUS, self.status(b)[0].tolist())),
                "interpretations": [describe(x) for x in z[0]],
                "velocities": V[0], "lat": lat[0], "lon": lon[0], "alt_ft": alt[0],
                "n_tokens": int((ids > 0).sum()), "n_unknown": int(n_unk),
                "callsign": win.callsign[0]}


def concat(parts: list[Windows]) -> Windows | None:
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    out = {}
    for k in Windows.__dataclass_fields__:
        vals = [getattr(p, k) for p in parts]
        if k == "raw_future":
            out[k] = [r for v in vals for r in v]
        elif vals[0] is None:
            out[k] = None
        else:
            out[k] = np.concatenate(vals)
    return Windows(**out)


__all__ = ["Forecaster", "Windows", "describe", "concat"]
