"""Trajectory baselines that share the CTP's inputs and output space.

MDN        mixture density network (Bishop, 1994) on the 16 basis coefficients: K Gaussian
           components with diagonal covariance, from the same history and instruction
           encoders. The classic probabilistic trajectory head.
Anchors    MultiPath-style (Chai et al., CoRL 2019): the K component means sit on fixed
           anchors (k-means of training coefficients) plus a learned offset, so modes
           cannot collapse onto each other.
Kinematic  instruction-aware rule, no learning: after the mean message-to-onset delay,
           turn at standard rate (3 deg/s) toward the commanded track in the commanded
           direction, climb at 1,800 fpm or descend at 1,500 fpm to the target level,
           change ground speed at 1 kt/s; otherwise keep the present velocity.

The MDN family is compared with the coefficient flow at equal encoders, so the comparison
isolates the output distribution: K Gaussians against a learned flow.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..data import atoms as A
from ..units import KT_TO_MS
from .ctp import H_STEPS, TargetSpace
from .encoders import HistoryEncoder, InstructionTokens


@dataclass
class MDNConfig:
    d: int = 128
    hist_layers: int = 2
    dropout: float = 0.1
    n_comp: int = 8
    condition: str = "atoms"         # atoms | none
    anchors: bool = False


class MDN(nn.Module):
    def __init__(self, cfg: MDNConfig, basis: dict, anchors: np.ndarray | None = None):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.space = TargetSpace(basis, "basis")
        self.dim = self.space.shape[0]
        self.hist = HistoryEncoder(d, cfg.hist_layers, cfg.dropout)
        if cfg.condition == "atoms":
            self.instr = InstructionTokens(d)
            self.mix_layer = nn.TransformerEncoderLayer(d, 4, 4 * d, cfg.dropout,
                                                        batch_first=True, norm_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, 4, batch_first=True)
        K = cfg.n_comp
        self.head = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                  nn.Linear(2 * d, K * (1 + 2 * self.dim)))
        if cfg.anchors:
            assert anchors is not None and anchors.shape == (K, self.dim)
        self.register_buffer("anchors", torch.as_tensor(
            anchors if anchors is not None else np.zeros((K, self.dim)), dtype=torch.float32))

    def params(self, b):
        mem = self.hist(b["hist"], b["ctx"], b["wake"], b["engine"], b["n_engines"])
        if self.cfg.condition == "atoms":
            mem = self.mix_layer(torch.cat([mem, self.instr(b["z"], b["state"])], 1))
        h = self.pool(self.query.expand(mem.shape[0], -1, -1), mem, mem)[0][:, 0]
        K, D = self.cfg.n_comp, self.dim
        out = self.head(h)
        logit = out[:, :K]
        mu = out[:, K:K + K * D].view(-1, K, D) + self.anchors
        log_sig = out[:, K + K * D:].view(-1, K, D).clamp(-5, 3)
        return logit, mu, log_sig

    def loss(self, b):
        x = self.space.encode(b["fut_vel"] - b["center"][:, None])[..., 0]    # [B, D]
        logit, mu, log_sig = self.params(b)
        lp = (-0.5 * ((x[:, None] - mu) / log_sig.exp()) ** 2 - log_sig
              - 0.5 * np.log(2 * np.pi)).sum(-1)
        return -(torch.logsumexp(F.log_softmax(logit, -1) + lp, -1)).mean()

    @torch.no_grad()
    def sample(self, b, n_samples, noise=None, generators=None):
        logit, mu, log_sig = self.params(b)
        B, K, D = mu.shape
        if generators is not None:
            u = torch.stack([torch.rand(n_samples, generator=g) for g in generators])
            eps = torch.stack([torch.randn(n_samples, D, generator=g) for g in generators])
        else:
            u, eps = torch.rand(B, n_samples), torch.randn(B, n_samples, D)
        c = F.softmax(logit, -1).cumsum(-1)
        comp = torch.searchsorted(c, u.to(c.device) * c[:, -1:]).clamp(max=K - 1)
        m = mu.gather(1, comp[..., None].expand(-1, -1, D))
        s = log_sig.exp().gather(1, comp[..., None].expand(-1, -1, D))
        x = m + s * eps.to(m.device)
        R = self.space.decode(x[..., None])
        return R + b["center"][:, None, None]


def kmeans_anchors(X, k=8, iters=50, seed=0):
    """k-means++ on training coefficients, [k, D]."""
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d = np.min(((X[:, None] - np.array(C)[None]) ** 2).sum(-1), 1)
        C.append(X[rng.choice(len(X), p=d / d.sum())])
    C = np.array(C)
    for _ in range(iters):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    return C.astype(np.float32)


def kinematic_rule(z, state, center, delay_s=7.0, turn_rate=3.0, climb_fpm=1800.0,
                   descend_fpm=1500.0, accel_kt_s=1.0):
    """Deterministic future velocities [B, 60, 3] (m/s, track frame) from atoms.

    z [B, 4] atoms; state [B, 3] (track deg, altitude ft, ground speed kt); center [B, 3].
    """
    z, state, center = (np.asarray(x) for x in (z, state, center))
    B = len(z)
    t = np.arange(1, H_STEPS + 1) - 0.5                       # mid-step
    act = np.clip(t - delay_s, 0, None)
    out = np.zeros((B, H_STEPS, 3), np.float32)
    for i in range(B):
        gs0, vz0 = center[i, 0], center[i, 2]
        # lateral: signed turn in the commanded direction, as the instruction tokens do
        lat = A.LATERAL
        psi = np.zeros(H_STEPS)
        if lat["kind"][z[i, 1]] == 2:
            d = float(A.relative_features("lateral", [z[i, 1]], state[i, 0], 0, 0)[0, 2] * 180)
            psi = np.sign(d) * np.minimum(turn_rate * act, abs(d))
        # speed
        gs = np.full(H_STEPS, gs0)
        if A.SPEED["kind"][z[i, 3]] == 2:
            tgt = A.SPEED["value"][z[i, 3]] * KT_TO_MS
            step = accel_kt_s * KT_TO_MS * act
            gs = gs0 + np.clip(tgt - gs0, -step, step)
        # vertical: integrate altitude toward the target
        vz = np.full(H_STEPS, vz0)
        if A.VERTICAL["kind"][z[i, 2]] == 2:
            tgt_m = A.VERTICAL["value"][z[i, 2]] * 0.3048
            alt = state[i, 1] * 0.3048
            for k in range(H_STEPS):
                if t[k] >= delay_s:
                    rem = tgt_m - alt
                    rate = (climb_fpm if rem > 0 else descend_fpm) * 0.00508     # fpm -> m/s
                    vz[k] = np.sign(rem) * min(rate, abs(rem))                   # stop at target
                alt += vz[k]
        elif A.VERTICAL["kind"][z[i, 2]] == 1:
            vz = np.zeros(H_STEPS)
        r = np.radians(psi)
        out[i, :, 0], out[i, :, 1], out[i, :, 2] = gs * np.cos(r), gs * np.sin(r), vz
    return out


def save_mdn(model: MDN, path, extra=None):
    torch.save({"cfg": asdict(model.cfg), "state": model.state_dict(),
                "anchors": model.anchors.cpu().numpy(),
                "basis": {k: v.cpu().numpy() for k, v in model.space.state_dict().items()},
                **(extra or {})}, path)


def load_mdn(path, device="cpu"):
    """An MDN or anchors checkpoint written by save_mdn; returns (model, checkpoint)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    bs = ck["basis"]
    basis = {"scale": bs["scale"], "mean": bs["mean"], "lateral_along": bs["comp0"],
             "lateral_right": bs["comp1"], "vertical": bs["comp2"], "coef_std": bs["cstd"]}
    m = MDN(MDNConfig(**ck["cfg"]), basis, ck["anchors"]).to(device)
    m.load_state_dict(ck["state"])
    return m.eval(), ck
