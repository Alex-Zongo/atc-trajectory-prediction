"""Shared building blocks for both stages."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..units import ENGINE_VOCAB, HEAD_BIN_DEG, N_HEAD_BINS, WAKE_VOCAB


@dataclass
class EncoderConfig:
    d_model: int = 128
    hidden: int = 256
    ac_emb: int = 16
    vocab_size: int = 64
    max_len: int = 48
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1


class TextBackbone(nn.Module):
    """Transformer over the heard transcript.

    Returns the full PER-TOKEN contextual states rather than one pooled vector, so
    downstream slot attention can let each clearance field read the tokens it needs. A
    confidence token is prepended, because confidence is the observable cue that lets the
    reported uncertainty become input-adaptive, and every slot has to be able to see it.

    Deliberately small and trained from scratch. The clearance vocabulary is tiny and
    templated, so a pretrained language model adds domain gap without adding coverage.
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab_size, cfg.hidden, padding_idx=0)
        self.pos = nn.Embedding(cfg.max_len + 1, cfg.hidden)       # +1 for the confidence slot
        self.conf_proj = nn.Linear(1, cfg.hidden)
        layer = nn.TransformerEncoderLayer(cfg.hidden, cfg.n_heads, cfg.hidden * 2, cfg.dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.hidden = cfg.hidden

    def forward(self, tokens, asr_confidence):
        B, L = tokens.shape
        conf = self.conf_proj(asr_confidence.unsqueeze(-1)).unsqueeze(1)
        h = torch.cat([conf, self.tok(tokens)], dim=1)
        h = h + self.pos(torch.arange(L + 1, device=tokens.device))[None]
        pad = torch.cat([tokens.new_zeros(B, 1, dtype=torch.bool), tokens.eq(0)], dim=1)
        return self.tf(h, src_key_padding_mask=pad), pad


class SlotAttentionPool(nn.Module):
    """`n_slots` learned queries, each cross-attending over the token states.

    Replaces single-vector pooling, which forced every clearance field through one
    bottleneck and destroyed digit order.
    """

    def __init__(self, hidden: int, n_slots: int, n_heads: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_slots, hidden) * 0.02)
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, states, pad, return_attn=False):
        q = self.queries.expand(states.shape[0], -1, -1)
        out, attn = self.attn(q, states, states, key_padding_mask=pad,
                              need_weights=return_attn, average_attn_weights=True)
        out = self.norm(out)
        return (out, attn) if return_attn else out


class FiLM(nn.Module):
    """Feature-wise linear modulation: h <- (1 + gamma(c)) * h + beta(c)."""

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, 2 * feat_dim)

    def forward(self, h, cond):
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        return (1.0 + gamma).unsqueeze(1) * h + beta.unsqueeze(1)


class AircraftEmbed(nn.Module):
    """Wake category, engine class and engine count to a conditioning vector.

    Only the PERFORMANCE class is used, never the typecode: that is a high-cardinality
    identity label and would be memorised the way a callsign would.
    """

    def __init__(self, dim):
        super().__init__()
        self.wake = nn.Embedding(len(WAKE_VOCAB), 16)
        self.eng = nn.Embedding(len(ENGINE_VOCAB), 16)
        self.mlp = nn.Sequential(nn.Linear(33, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, wake_id, eng_id, n_eng):
        return self.mlp(torch.cat([self.wake(wake_id), self.eng(eng_id),
                                   n_eng.unsqueeze(-1) / 4.0], -1))


class StateFiLM(nn.Module):
    """Ground a clearance field head in the aircraft's current state.

    The output layer is zero-initialised, so the module starts at the identity and the
    grounded model is bit-for-bit equal to the ungrounded one at initialisation. That makes
    the grounding ablation clean.
    """

    def __init__(self, state_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2 * hidden))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, state):
        gamma, beta = self.net(state).chunk(2, dim=-1)
        return h * (1 + gamma) + beta


class ContinuousTimeEmbed(nn.Module):
    """Sinusoidal embedding of a real-valued time in SECONDS, then an MLP.

    Used for the raw-observation past, where each token is a real ADS-B sample carrying its
    own time_to_onset, so spacing and decimation are encoded from the actual timestamps
    rather than assumed uniform. `max_period` spans the past window, so the lowest
    frequency covers it.
    """

    def __init__(self, dim, max_period=240.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, seconds):
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period)
                          * torch.arange(half, device=seconds.device) / max(half - 1, 1))
        a = seconds[..., None] * freqs
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))


def head_circular_stats(logits):
    """Categorical heading logits [B, 72] to (mean_deg, circular_std_deg)."""
    ang = torch.deg2rad(torch.arange(N_HEAD_BINS, device=logits.device) * HEAD_BIN_DEG)
    p = logits.softmax(-1)
    c, s = (p * ang.cos()).sum(-1), (p * ang.sin()).sum(-1)
    mu = torch.rad2deg(torch.atan2(s, c)) % 360
    R = torch.sqrt(c * c + s * s).clamp(1e-6, 1 - 1e-9)
    return mu, torch.rad2deg(torch.sqrt(-2 * torch.log(R)))
