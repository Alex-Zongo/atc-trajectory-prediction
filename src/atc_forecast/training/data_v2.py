"""Batches for the v2 models, gathered straight from the prebuilt arrays.

Everything is precomputed (windows_v2.py, messages.py), so a batch is a fancy-index into
memory-mapped arrays and a transfer to the device; there is no per-item Python. Random
choices that remain at training time (which interpretation a CTP window gets, which
message an SBM step sees) are drawn from a numpy Generator owned by the sampler, so a run
is reproducible from its seed.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data import atoms as A
from ..units import ENGINE_VOCAB, WAKE_VOCAB

SLOTS = ("atom_lateral", "atom_vertical", "atom_speed")


class WindowStore:
    """The trajectory windows: arrays (memory-mapped) plus the per-window table."""

    def __init__(self, data_dir, prim_file="primitives.npz"):
        d = Path(data_dir)
        self.dir = d
        self.meta = pd.read_parquet(d / "windows.parquet")
        a = d / "arrays"
        self.hist = np.load(a / "hist.npy", mmap_mode="r")
        self.ctx = np.load(a / "ctx.npy", mmap_mode="r")
        self.center = np.load(a / "center.npy", mmap_mode="r")
        self.fut_vel = np.load(a / "fut_vel.npy", mmap_mode="r")
        self.raw_future = np.load(a / "raw_future.npy", mmap_mode="r")
        self.basis = dict(np.load(d / "basis.npz"))
        self.manifest = json.loads((d / "manifest.json").read_text())
        m = self.meta
        self.wake = m["wake"].map({w: i for i, w in enumerate(WAKE_VOCAB)}).fillna(
            WAKE_VOCAB.index("unknown")).to_numpy(np.int64)
        self.engine = m["engine_class"].map({w: i for i, w in enumerate(ENGINE_VOCAB)}).fillna(
            ENGINE_VOCAB.index("unknown")).to_numpy(np.int64)
        self.n_engines = m["n_engines"].fillna(1.0).to_numpy(np.float32)
        self.state = m[["trk_deg", "alt_ft", "gs_kt"]].to_numpy(np.float32)
        self.z_true = np.stack([np.zeros(len(m), np.int64)]
                               + [m[c].to_numpy(np.int64) for c in SLOTS], 1)
        self.is_man = (m["kind"] == "maneuver").to_numpy()
        self.ok = ((m["offset0_h_m"] <= 50) & (m["offset0_v_m"].abs() <= 30)).to_numpy()
        self.vr = m["vr_fpm"].to_numpy(np.float32)
        lat_m = np.isin(m["lateral"], ["L", "R"]) & self.is_man
        ver_m = np.isin(m["vertical"], ["climb", "descend"]) & self.is_man
        spd_m = np.isin(m["speed"], ["accel", "decel"]) & self.is_man
        # axes on which a MAINTAIN is truthful (mirrors messages._steady_axes)
        self.steady = np.stack([~lat_m, ~ver_m & (np.abs(self.vr) < 300), ~spd_m], 1)
        pf = d / prim_file                            # design-A targets (build_primitives_v2.py)
        if pf.exists():
            f = np.load(pf)
            X = np.concatenate([f["params"].reshape(len(m), -1), f["coef"].reshape(len(m), -1)], 1)
            self.prim_x = ((X - f["mu"]) / f["sd"]).astype(np.float32)
            self.kind = f["kind"].astype(np.int64)
            if "kind_axis" in f.files:
                self.kind_axis = f["kind_axis"].astype(np.int64)
        if "coef_std" not in self.basis:
            self.basis["coef_std"] = self.coefficients(self.split("train")[::7]).std(0)

    def swap_basis(self, family):
        """Replace the PCA shapes by a fixed family with the same number of shapes per axis
        (the Fourier ablation). "sine": quarter-wave sines sin((k - 1/2) pi t / 60), zero at
        t = 0 and free at the horizon, orthonormalized on the 60-step grid. The per-step scale
        and mean are kept; the coefficient standardization is recomputed."""
        if family == "pca":
            return
        if family != "sine":
            raise ValueError(family)
        t = np.arange(1, 61)
        for k in ("lateral_along", "lateral_right", "vertical"):
            n = len(self.basis[k])
            S = np.stack([np.sin((j - .5) * np.pi * t / 60) for j in range(1, n + 1)], 1)
            self.basis[k] = np.linalg.qr(S)[0].T.astype(self.basis[k].dtype)
        self.basis.pop("coef_std", None)
        self.basis["coef_std"] = self.coefficients(self.split("train")[::7]).std(0)

    def coefficients(self, idx):
        """Basis coefficients [N, D] of the scaled residual targets (before the per-
        coefficient standardization)."""
        b = self.basis
        idx = np.sort(idx)
        R = (self.fut_vel[idx] - self.center[idx][:, None]) / b["scale"] - b["mean"]
        return np.concatenate([R[..., i] @ b[k].T for i, k in
                               enumerate(("lateral_along", "lateral_right", "vertical"))], 1)

    def split(self, name, quality=True):
        """Window indices of a split. quality=True drops the 1.2 % of windows whose two
        estimates of the present (history filter vs future smoother) disagree by more than
        50 m horizontally or 30 m vertically: a sign of bad raw samples around t0."""
        m = self.meta["split"].to_numpy() == name
        if quality:
            m &= self.ok
        return np.where(m)[0]

    def batch(self, idx, device, z=None, sort=True):
        idx = np.asarray(idx)
        if sort:
            idx = np.sort(idx)                         # sorted reads are faster on mmap
        t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt).to(device)
        b = {"hist": t(self.hist[idx]), "ctx": t(self.ctx[idx]),
             "center": t(self.center[idx]), "fut_vel": t(self.fut_vel[idx]),
             "wake": t(self.wake[idx], torch.long), "engine": t(self.engine[idx], torch.long),
             "n_engines": t(self.n_engines[idx]), "state": t(self.state[idx]),
             "window_id": idx}
        if hasattr(self, "prim_x"):
            b["prim_x"] = t(self.prim_x[idx])
            b["kind"] = t(self.kind[idx], torch.long)
            if hasattr(self, "kind_axis"):
                b["kind_axis"] = t(self.kind_axis[idx], torch.long)
        if z is not None:
            b["z"] = t(z, torch.long)
        return b, idx


@dataclass
class ZPolicy:
    """Which clean interpretation a CTP training window is shown. Mirrors the message
    generator (data/messages.py), so a Z means the same thing to the CTP as to the SBM:

    Any window, with p_null: a message that is not about this aircraft's motion (status
    own_other or other, every motion slot NONE). Drawn independently of the window's
    kind, so "not ours" is followed by maneuvers at their base rate and the CTP learns
    the history-only answer for it, not "steady flight".
    Maneuver windows otherwise: the true atoms, with p_maintain a MAINTAIN clause on a
    steady axis (the generator's own + maintain clause).
    Steady windows otherwise: one MAINTAIN on a steady axis alone (the maintain message),
    or a no-motion message when no axis is steady.
    """
    p_null: float = 0.2
    p_maintain: float = 0.2

    def draw(self, store: WindowStore, idx, rng):
        z = store.z_true[idx].copy()
        man, st = store.is_man[idx], store.steady[idx]
        null = (rng.random(len(idx)) < self.p_null) | (~man & ~st.any(1))
        z[~man, 1:] = A.NONE
        clause = man & (rng.random(len(idx)) < self.p_maintain)
        for i in np.where(~null & (~man | clause) & st.any(1))[0]:
            z[i, 1 + rng.choice(np.where(st[i])[0])] = A.MAINTAIN
        z[null, 1:] = A.NONE
        z[:, 0] = np.where(null, rng.choice([1, 2], len(idx)), 0)
        return z


class MessageStore:
    """Precomputed messages for one channel profile, joined to the windows."""

    MIX = {"own": 0.5, "own_other": 0.2, "other": 0.2, "maintain": 0.1}

    def __init__(self, data_dir, profile="jacktol", windows: WindowStore | None = None):
        d = Path(data_dir) / f"messages_{profile}"
        self.msgs = pd.read_parquet(d / "messages.parquet")
        self.tokens = np.load(d / "tokens.npy", mmap_mode="r")
        self.kappa = np.load(d / "kappa.npy", mmap_mode="r")
        self.own_cs = np.load(d / "own_callsign.npy")
        self.vocab = json.loads((d / "vocab.json").read_text())
        self.windows = windows or WindowStore(data_dir)
        self.window_id = self.msgs["window_id"].to_numpy()
        self.z = self.msgs[["status"] + list(SLOTS)].to_numpy(np.int64)
        self.variant = self.msgs["variant"].to_numpy()
        self.split = np.where(self.windows.ok[self.window_id],
                              self.windows.meta["split"].to_numpy()[self.window_id], "excluded")

    def weights(self, rows):
        """Sampling weights giving the target variant mix within `rows`."""
        v = self.variant[rows]
        w = np.zeros(len(rows))
        for name, share in self.MIX.items():
            m = v == name
            if m.any():
                w[m] = share / m.sum()
        return w / w.sum()

    def rows(self, split):
        return np.where(self.split == split)[0]

    def batch(self, rows, device):
        rows = np.sort(np.asarray(rows))
        wid = self.window_id[rows]
        b, _ = self.windows.batch(wid, device, sort=False)
        t = lambda x, dt: torch.as_tensor(np.asarray(x), dtype=dt).to(device)
        b["tokens"] = t(self.tokens[rows].astype(np.int64), torch.long)
        b["kappa"] = t(self.kappa[rows].astype(np.float32) / 255.0, torch.float32)
        b["cs_tokens"] = t(self.own_cs[wid].astype(np.int64), torch.long)
        b["z"] = t(self.z[rows], torch.long)
        b["variant"] = self.variant[rows]
        b["msg_row"] = rows
        return b
