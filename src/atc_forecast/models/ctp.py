"""Conditional trajectory predictor (CTP): p(Y | X, Z) by rectified flow.

One class covers the main model and its trajectory-side competitors, so they differ only
in what the switches say:

  output      "basis"  flow on 18 PCA coefficients of the scaled residuals (6 along,
                       4 right, 8 vertical); smooth by construction, cheap to sample
              "steps"  flow on all 60 x 3 scaled residuals
  condition   "atoms"  Z as 4 instruction tokens (the composed model)
              "none"   history only: a separate model, same budget
              "text"   transcript tokens with word confidence (the direct competitor)

Memory = history tokens (8) + condition tokens. The flow decoder is a small transformer
over output tokens (16 coefficients or 60 steps) with cross-attention to the memory and
adaLN modulation by flow time. Memory is computed once per window and interpretation and
reused at every sampling step.

Targets: R = fut_vel - center (m/s), scaled per step by s_k (basis.npz). Training draws
flow time s from a logit-normal (Esser et al., 2024) or uniform, x_s = (1 - s) x0 + s x1,
and regresses the velocity x1 - x0. Sampling integrates with Euler or Heun from seeded
noise: sample k of window w always gets the same noise, whatever the batch or device.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .encoders import HistoryEncoder, InstructionTokens, TextEncoder

H_STEPS = 60
BASIS_K = (6, 4, 8)


@dataclass
class CTPConfig:
    d: int = 128
    hist_layers: int = 2
    dec_layers: int = 4
    heads: int = 4
    dropout: float = 0.1
    output: str = "basis"
    condition: str = "atoms"
    vocab_size: int = 0
    text_layers: int = 4
    time_sampling: str = "logit_normal"
    kinds: str = "joint"         # models/prim.py: one head over joint kinds, or "axis": one per axis
    target: str = "prim"         # models/prim.py: flow on primitives, or "pca" (kind-conditioned CTP)
    kind_sample: float = 0.0     # models/prim.py: share of training rows whose flow sees a kind drawn
                                 # from the head instead of the true one (scheduled sampling)
    kind_cond: bool = True       # models/prim.py: False = the kind head is a readout only (trained
                                 # alongside, never fed to the flow)


class TargetSpace(nn.Module):
    """Scaled residuals <-> model space (per-step or PCA coefficients). Fixed buffers."""

    def __init__(self, basis: dict, output: str):
        super().__init__()
        self.output = output
        self.register_buffer("scale", torch.as_tensor(basis["scale"]))       # [60, 3]
        self.register_buffer("mean", torch.as_tensor(basis["mean"]))         # [60, 3]
        for i, k in enumerate(("lateral_along", "lateral_right", "vertical")):
            self.register_buffer(f"comp{i}", torch.as_tensor(basis[k]))      # [k_a, 60]
        # per-coefficient std over training data, so every coefficient enters the flow at
        # unit variance (the leading component otherwise carries ~50x the variance)
        n = sum(basis[k].shape[0] for k in ("lateral_along", "lateral_right", "vertical"))
        self.register_buffer("cstd", torch.as_tensor(basis.get("coef_std", np.ones(n)),
                                                     dtype=torch.float32))

    @property
    def shape(self):
        return (sum(getattr(self, f"comp{i}").shape[0] for i in range(3)), 1) \
            if self.output == "basis" else (H_STEPS, 3)

    def encode(self, R):
        x = R / self.scale
        if self.output == "steps":
            return x
        x = x - self.mean
        c = [torch.einsum("bt,kt->bk", x[..., i], getattr(self, f"comp{i}")) for i in range(3)]
        return (torch.cat(c, -1) / self.cstd)[..., None]

    def decode(self, x):
        if self.output == "steps":
            return x * self.scale
        c = x[..., 0] * self.cstd
        out, j = [], 0
        for i in range(3):
            comp = getattr(self, f"comp{i}")
            out.append(torch.einsum("...k,kt->...t", c[..., j:j + comp.shape[0]], comp))
            j += comp.shape[0]
        return (torch.stack(out, -1) + self.mean) * self.scale


def time_embedding(s, dim=64):
    half = dim // 2
    f = torch.exp(-math.log(1000.0) * torch.arange(half, device=s.device) / half)
    a = s[:, None] * 1000.0 * f[None]
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class FlowBlock(nn.Module):
    """Self-attention, cross-attention to memory, MLP; each modulated by flow time."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.n1, self.n2, self.n3 = (nn.LayerNorm(d, elementwise_affine=False) for _ in range(3))
        self.sa = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ca = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * d, d))
        self.mod = nn.Linear(d, 9 * d)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, mem, mem_pad, temb):
        m = self.mod(temb)[:, None].chunk(9, -1)
        h = self.n1(x) * (1 + m[0]) + m[1]
        x = x + (1 + m[2]) * self.sa(h, h, h, need_weights=False)[0]
        h = self.n2(x) * (1 + m[3]) + m[4]
        x = x + (1 + m[5]) * self.ca(h, mem, mem, key_padding_mask=mem_pad,
                                     need_weights=False)[0]
        h = self.n3(x) * (1 + m[6]) + m[7]
        return x + (1 + m[8]) * self.mlp(h)


class CTP(nn.Module):
    def __init__(self, cfg: CTPConfig, basis: dict):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.space = TargetSpace(basis, cfg.output)
        n_tok, tdim = self.space.shape
        self.hist = HistoryEncoder(d, cfg.hist_layers, cfg.dropout)
        if cfg.condition == "atoms":
            self.instr = InstructionTokens(d)
        elif cfg.condition == "text":
            self.text = TextEncoder(cfg.vocab_size, d, cfg.text_layers, dropout=cfg.dropout)
        self.inp = nn.Linear(tdim, d)
        self.pos = nn.Parameter(torch.randn(n_tok, d) * 0.02)
        self.temb = nn.Sequential(nn.Linear(64, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([FlowBlock(d, cfg.heads, cfg.dropout)
                                     for _ in range(cfg.dec_layers)])
        self.out_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, tdim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    # ---------------------------------------------------------------- memory
    def memory(self, b, z=None):
        """Memory tokens and padding mask for a batch. `z` [B, 4] overrides b["z"]; with
        [B, K, 4] it builds one memory per (window, interpretation), flattened to B*K."""
        mem = self.hist(b["hist"], b["ctx"], b["wake"], b["engine"], b["n_engines"])
        B = mem.shape[0]
        if self.cfg.condition == "atoms":
            z = b["z"] if z is None else z
            if z.dim() == 3:
                K = z.shape[1]
                mem = mem[:, None].expand(-1, K, -1, -1).reshape(B * K, *mem.shape[1:])
                state = b["state"][:, None].expand(-1, K, -1).reshape(B * K, 3)
                z = z.reshape(B * K, 4)
            else:
                state = b["state"]
            mem = torch.cat([mem, self.instr(z, state)], 1)
            pad = torch.zeros(mem.shape[:2], dtype=torch.bool, device=mem.device)
        elif self.cfg.condition == "text":
            t, tpad = self.text(b["cs_tokens"], b["tokens"], b["kappa"])
            pad = torch.cat([torch.zeros(mem.shape[:2], dtype=torch.bool, device=mem.device),
                             tpad], 1)
            mem = torch.cat([mem, t], 1)
        else:
            pad = torch.zeros(mem.shape[:2], dtype=torch.bool, device=mem.device)
        return mem, pad

    # ---------------------------------------------------------------- velocity field
    def velocity(self, x, s, mem, pad):
        h = self.inp(x) + self.pos
        temb = self.temb(time_embedding(s))
        for blk in self.blocks:
            h = blk(h, mem, pad, temb)
        return self.out(self.out_norm(h))

    def sample_time(self, n, device):
        if self.cfg.time_sampling == "logit_normal":
            return torch.sigmoid(torch.randn(n, device=device))
        return torch.rand(n, device=device)

    def loss(self, b):
        mem, pad = self.memory(b)
        R = b["fut_vel"] - b["center"][:, None]
        x1 = self.space.encode(R)
        x0 = torch.randn_like(x1)
        s = self.sample_time(x1.shape[0], x1.device)
        xs = (1 - s)[:, None, None] * x0 + s[:, None, None] * x1
        return F.mse_loss(self.velocity(xs, s, mem, pad), x1 - x0)

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def integrate(self, x0, mem, pad, nfe=12, method="euler"):
        x = x0
        ts = torch.linspace(0, 1, nfe + 1, device=x.device)
        for i in range(nfe):
            s0, s1 = ts[i], ts[i + 1]
            sv = s0.expand(x.shape[0])
            v = self.velocity(x, sv, mem, pad)
            if method == "heun" and i < nfe - 1:
                xe = x + (s1 - s0) * v
                v2 = self.velocity(xe, s1.expand(x.shape[0]), mem, pad)
                x = x + (s1 - s0) * 0.5 * (v + v2)
            else:
                x = x + (s1 - s0) * v
        return x

    @torch.no_grad()
    def sample(self, b, n_samples, noise=None, z=None, nfe=12, method="euler"):
        """Future velocities [B, (K,) S, 60, 3] in m/s (track frame).

        `noise` [B*(K), S, *space.shape] from seeded_noise; z [B, K, 4] samples one batch of
        S per interpretation.
        """
        mem, pad = self.memory(b, z)
        M = mem.shape[0]
        S = n_samples
        shape = self.space.shape
        if noise is None:
            noise = torch.randn(M, S, *shape, device=mem.device)
        x0 = noise.reshape(M * S, *shape).to(mem.device)
        mem_r = mem[:, None].expand(-1, S, -1, -1).reshape(M * S, *mem.shape[1:])
        pad_r = pad[:, None].expand(-1, S, -1).reshape(M * S, -1)
        x = self.integrate(x0, mem_r, pad_r, nfe, method)
        R = self.space.decode(x).reshape(M, S, H_STEPS, 3)
        center = b["center"]
        if z is not None and z.dim() == 3:
            K = z.shape[1]
            R = R.reshape(-1, K, S, H_STEPS, 3)
            return R + center[:, None, None, None]
        return R + center[:, None, None]


def seeded_noise(window_ids, n_samples, shape, run_seed=0, k=None):
    """Noise that depends only on (run seed, window id, interpretation k, sample index):
    the same window gets the same samples whatever the batch size, order or device."""
    out = torch.empty(len(window_ids), n_samples, *shape)
    for i, w in enumerate(np.asarray(window_ids).tolist()):
        g = torch.Generator().manual_seed(
            hash((int(run_seed), int(w), -1 if k is None else int(k))) & 0x7FFFFFFF)
        out[i] = torch.randn(n_samples, *shape, generator=g)
    return out


def save_ctp(model: CTP, path, extra=None):
    if hasattr(model, "basis_np"):                               # design A family (models/prim.py)
        from .prim import save_prim
        return save_prim(model, path, extra)
    torch.save({"cfg": asdict(model.cfg), "state": model.state_dict(),
                "basis": {k: v.cpu().numpy() for k, v in model.space.state_dict().items()},
                **(extra or {})}, path)


def load_ctp(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("model") == "prim":
        from .prim import load_prim
        return load_prim(ck, device)
    b = ck["basis"]
    basis = {"scale": b["scale"], "mean": b["mean"], "lateral_along": b["comp0"],
             "lateral_right": b["comp1"], "vertical": b["comp2"], "coef_std": b["cstd"]}
    m = CTP(CTPConfig(**ck["cfg"]), basis).to(device)
    m.load_state_dict(ck["state"])
    return m.eval(), ck
