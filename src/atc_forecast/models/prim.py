"""Design A: an explicit maneuver kind, then a flow over maneuver primitives.

p(Y | X, Z) = sum_k  P(kind = k | X, Z)  p_flow(theta | kind = k, X, Z),   Y = rule(theta)

  kind head   a classifier on the pooled memory (history + instruction tokens) over the K
              physical kinds of scripts/build_primitives_v2.py: explicit, readable
              probabilities, what a mixture head does well
  flow        rectified flow, as in the CTP, on 21 standardized numbers given the kind:
              per axis (heading, altitude, ground speed) the primitive (v0, d, r, tau, e)
              -- present rate, delay, new rate, ramp time, capture time -- plus 2 residual
              weights per axis for what the primitive misses
  rule        data/primitives.py: each axis's rate follows present rate -> ramp -> new
              rate -> ramp to zero; heading and ground speed give the horizontal velocity,
              altitude the vertical one

Variants (CTPConfig fields):
  kinds  "joint"  one head over the K physical kinds (k-means on all three axes)
         "axis"   one head per axis, chained turn -> vertical -> speed, each over 4 kinds:
                  keeps its rate / stops (rolls out, levels off, settles) / new rate positive
                  (right, climb, accelerate) / negative; combined maneuvers by construction
  target "prim"   the flow carries the 21 primitive numbers (design A)
         "pca"    the flow carries the CTP's 18 PCA weights (a kind-conditioned CTP): the
                  ablation that separates the effect of kinds from that of primitives

  kind_cond False  the kind head is a readout: trained alongside the flow (shared memory),
                   never fed to it; each sample's kind is read off its own primitives

Everything else (encoders, flow decoder, sampler, seeded noise) is the CTP's. The noise of
one sample has one extra coordinate per kind head, turned into a uniform by the normal CDF,
which draws the kind: a sample's kind is as reproducible as its flow noise.
"""
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .ctp import CTP, CTPConfig, H_STEPS

N_PRIM = 15
KT, FT = 1.94384, 3.28084


class PrimSpace(nn.Module):
    """Standardized design-A state <-> track-frame velocities. Fixed buffers."""

    output = "prim"

    def __init__(self, prim: dict):
        super().__init__()
        for k in ("mu", "sd", "res_comp", "res_scale", "centers"):
            self.register_buffer(k, torch.as_tensor(np.asarray(prim[k]), dtype=torch.float32))
        self.register_buffer("bank", torch.as_tensor(float(np.asarray(prim.get("bank", 0.0))),
                                                     dtype=torch.float32))
        self.n = self.mu.shape[0]                                   # 15 + 3 * n_res
        self.n_kinds = self.centers.shape[0]

    extra = 1                           # noise coordinates that draw kinds, set by PrimCTP

    @property
    def shape(self):
        return (self.n + self.extra, 1)

    def decode(self, x, center):
        """x [M, n] standardized -> velocities [M, 60, 3] (m/s)."""
        X = x * self.sd + self.mu
        P = X[:, :N_PRIM].reshape(-1, 3, 5)
        v0, d, r, tau, e = P.unbind(-1)                             # [M, 3]
        d = d.clamp(0, 60)
        tau = tau.clamp(0, 30)
        e = torch.maximum(e, d + tau).clamp(max=61)
        tc = torch.arange(1, H_STEPS + 1, device=x.device, dtype=x.dtype) - 0.5
        up = ((tc - d[..., None]) / tau[..., None].clamp(min=0.5)).clamp(0, 1)
        down = torch.where(e[..., None] >= 61, torch.zeros_like(up),
                           ((tc - e[..., None]) / tau[..., None].clamp(min=0.5)).clamp(0, 1))
        rate = v0[..., None] * (1 - up) + r[..., None] * up * (1 - down)   # [M, 3, 60]
        cum = rate.cumsum(-1)
        psi, z, gs = cum[:, 0], cum[:, 1], cum[:, 2]
        V = center[:, 0:1] + gs / KT
        if self.bank > 0:                       # rate[:, 0] is bank (deg): omega = g tan(phi) / V
            phi = torch.deg2rad(rate[:, 0].clamp(-60, 60))
            psi = torch.rad2deg(9.81 * torch.tan(phi) / V.clamp(min=1.0)).cumsum(-1)
        h = torch.deg2rad(psi)
        vz = rate[:, 1] / FT
        vel = torch.stack([V * torch.cos(h), V * torch.sin(h), vz], -1)
        coef = X[:, N_PRIM:].reshape(-1, 3, self.res_comp.shape[1])
        res = torch.einsum("mak,akt->mta", coef, self.res_comp) * self.res_scale
        return vel + res


class PCASpace(nn.Module):
    """The CTP's 18 PCA weights, with the same decode signature as PrimSpace."""

    output = "pca"

    def __init__(self, base, n_kinds):
        super().__init__()
        self.base = base
        self.n = base.shape[0]
        self.n_kinds = n_kinds
        self.extra = 1

    @property
    def shape(self):
        return (self.n + self.extra, 1)

    def decode(self, x, center):
        return self.base.decode(x[..., None]) + center[:, None]


class PrimCTP(CTP):
    def __init__(self, cfg: CTPConfig, basis: dict, prim: dict):
        super().__init__(cfg, basis)
        self.basis_np = {k: np.asarray(v) for k, v in basis.items()}   # kept for saving only
        d = cfg.d
        self.prim = PrimSpace(prim)
        self.space = self.prim if cfg.target == "prim" else PCASpace(self.space, self.prim.n_kinds)
        n = self.space.n
        self.pos = nn.Parameter(torch.randn(n, d) * 0.02)
        self.sizes = [self.prim.n_kinds] if cfg.kinds == "joint" else [4, 4, 4]
        mlp = lambda k: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, k))
        self.kind_emb = nn.ModuleList([nn.Embedding(k, d) for k in self.sizes])
        self.kind_head = nn.ModuleList([mlp(k) for k in self.sizes])
        self.space.extra = len(self.sizes)

    @property
    def n_heads(self):
        return len(self.sizes)

    def noise_shape(self):
        return self.space.shape

    def pooled(self, mem, pad):
        keep = (~pad).float()[..., None]
        return (mem * keep).sum(1) / keep.sum(1).clamp(min=1)

    def kind_logits(self, mem, pad, kinds):
        """Logits of every head given the earlier heads' kinds [M, H] (teacher forcing)."""
        h, out = self.pooled(mem, pad), []
        for i, head in enumerate(self.kind_head):
            out.append(head(h))
            if i + 1 < self.n_heads:
                h = h + self.kind_emb[i](kinds[:, i])
        return out

    def draw_kinds(self, mem, pad, u):
        """Kinds [M, H] by inverse CDF of each head's probabilities at uniforms u [M, H]."""
        h, ks = self.pooled(mem, pad), []
        for i, head in enumerate(self.kind_head):
            p = F.softmax(head(h), -1)
            k = torch.searchsorted(p.cumsum(-1).contiguous(), u[:, i:i + 1].contiguous())[:, 0]
            k = k.clamp(max=p.shape[1] - 1)
            ks.append(k)
            h = h + self.kind_emb[i](k)
        return torch.stack(ks, 1)

    def labels(self, b):
        return b["kind"][:, None] if self.cfg.kinds == "joint" else b["kind_axis"]

    def with_kind(self, mem, pad, kinds):
        tok = torch.stack([e(kinds[:, i]) for i, e in enumerate(self.kind_emb)], 1)
        mem = torch.cat([mem, tok], 1)
        pad = torch.cat([pad, torch.zeros_like(pad[:, :tok.shape[1]])], 1)
        return mem, pad

    def target(self, b):
        if self.cfg.target == "prim":
            return b["prim_x"][..., None]
        return self.space.base.encode(b["fut_vel"] - b["center"][:, None])

    def loss(self, b):
        mem, pad = self.memory(b)
        kinds = self.labels(b)
        logits = self.kind_logits(mem, pad, kinds)
        ce = sum(F.cross_entropy(lg, kinds[:, i]) for i, lg in enumerate(logits))
        if self.training and self.cfg.kind_sample > 0:
            # scheduled sampling: some rows condition the flow on the head's own draw, as at
            # inference, so the flow learns to cope with a wrong kind
            drawn = torch.stack([torch.multinomial(F.softmax(lg.detach(), -1), 1)[:, 0] for lg in logits], 1)
            swap = torch.rand(kinds.shape[0], 1, device=kinds.device) < self.cfg.kind_sample
            kinds = torch.where(swap, drawn, kinds)
        memk, padk = self.with_kind(mem, pad, kinds) if self.cfg.kind_cond else (mem, pad)
        x1 = self.target(b)
        x0 = torch.randn_like(x1)
        s = self.sample_time(x1.shape[0], x1.device)
        xs = (1 - s)[:, None, None] * x0 + s[:, None, None] * x1
        return F.mse_loss(self.velocity(xs, s, memk, padk), x1 - x0) + ce

    @torch.no_grad()
    def kind_probs(self, b, z=None):
        """First head's probabilities (the joint kinds, or the turn kinds); with per-axis
        kinds, later heads are conditioned on the true earlier kinds when b has them."""
        mem, pad = self.memory(b, z)
        kinds = self.labels(b) if ("kind" in b and "kind_axis" in b) else \
            torch.zeros(mem.shape[0], self.n_heads, dtype=torch.long, device=mem.device)
        return [F.softmax(lg, -1) for lg in self.kind_logits(mem, pad, kinds)]

    @torch.no_grad()
    def sample(self, b, n_samples, noise=None, z=None, nfe=12, method="euler"):
        """Future velocities [B, (K,) S, 60, 3] (m/s, track frame), as CTP.sample."""
        mem, pad = self.memory(b, z)
        M, S = mem.shape[0], n_samples
        shape = self.noise_shape()
        if noise is None:
            noise = torch.randn(M, S, *shape, device=mem.device)
        noise = noise.reshape(M * S, *shape).to(mem.device)
        H = self.n_heads
        u = torch.special.ndtr(noise[:, :H, 0])
        mem_r = mem[:, None].expand(-1, S, -1, -1).reshape(M * S, *mem.shape[1:])
        pad_r = pad[:, None].expand(-1, S, -1).reshape(M * S, -1)
        if self.cfg.kind_cond:
            memk, padk = self.with_kind(mem_r, pad_r, self.draw_kinds(mem_r, pad_r, u))
        else:                                   # readout: the flow never sees a kind
            memk, padk = mem_r, pad_r
        x = self.integrate(noise[:, H:], memk, padk, nfe, method)[..., 0]
        center = b["center"]
        if z is not None and z.dim() == 3:
            K = z.shape[1]
            c = center[:, None].expand(-1, K, -1).reshape(-1, 3)
        else:
            c = center
        c = c[:, None].expand(-1, S, -1).reshape(-1, 3)
        V = self.space.decode(x, c).reshape(M, S, H_STEPS, 3)
        if z is not None and z.dim() == 3:
            return V.reshape(-1, z.shape[1], S, H_STEPS, 3)
        return V


def load_prim_targets(data_dir, name="primitives.npz"):
    f = np.load(f"{data_dir}/{name}")
    out = {k: f[k] for k in ("mu", "sd", "res_comp", "res_scale", "centers")}
    out["bank"] = f["bank"] if "bank" in f.files else np.float32(0)
    return out


def save_prim(model: PrimCTP, path, extra=None):
    torch.save({"model": "prim", "cfg": asdict(model.cfg), "state": model.state_dict(),
                "basis_np": model.basis_np,
                "prim": {k: v.cpu().numpy() for k, v in model.prim.state_dict().items()},
                **(extra or {})}, path)


def load_prim(ck, device="cpu"):
    m = PrimCTP(CTPConfig(**ck["cfg"]), ck["basis_np"], ck["prim"]).to(device)
    st = dict(ck["state"])
    if "kind_emb.weight" in st:                     # first design-A checkpoint: single head
        st["kind_emb.0.weight"] = st.pop("kind_emb.weight")
        for k in [k for k in st if k.startswith("kind_head.")]:
            st["kind_head.0." + k[len("kind_head."):]] = st.pop(k)
    if "prim.mu" not in st:                         # first checkpoint: space was the PrimSpace
        for k in [k for k in st if k.startswith("space.")]:
            st["prim." + k[len("space."):]] = st[k]
    for k, v in m.state_dict().items():           # checkpoints from before the bank option
        if k.endswith(".bank") and k not in st:
            st[k] = v
    m.load_state_dict(st)
    return m.eval(), ck
