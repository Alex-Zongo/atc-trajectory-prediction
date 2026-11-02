"""Encoders shared by the v2 models: history, instruction tokens, transcript.

HistoryEncoder   30 s at 1 Hz -> 6 patch tokens of 5 s, a NOW token with the exact
                 present, an aircraft token; 2 transformer layers. 8 memory tokens.
InstructionTokens  Z (status + 3 atoms) and the present state -> 4 tokens. Atom identity
                 (an embedding) plus features relative to the present (a small MLP), so
                 "turn 20 deg right" looks alike on any heading. No text in between: the
                 SBM's output atoms are looked up directly.
TextEncoder      own callsign (exact, from ADS-B) and transcript tokens with per-word
                 confidence; 4 transformer layers.

All modules take d (model width) and use pre-norm transformer layers.
"""

import torch
from torch import nn

from ..data import atoms as A
from ..units import ENGINE_VOCAB, WAKE_VOCAB

N_HIST, N_HIST_FEAT, N_CTX, PATCH = 30, 9, 5, 5


def encoder_layers(d, n_layers, heads=4, dropout=0.1):
    layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True,
                                       norm_first=True, activation="gelu")
    return nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)


class AircraftEmbedding(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.wake = nn.Embedding(len(WAKE_VOCAB), d)
        self.engine = nn.Embedding(len(ENGINE_VOCAB), d)
        self.n_eng = nn.Linear(1, d)

    def forward(self, wake, engine, n_engines):
        return self.wake(wake) + self.engine(engine) + self.n_eng(n_engines[:, None] / 4.0)


class HistoryEncoder(nn.Module):
    """Memory tokens [B, 8, d]: NOW, aircraft, then six 5 s patches oldest first."""

    def __init__(self, d=128, n_layers=2, dropout=0.1):
        super().__init__()
        n_patch = N_HIST // PATCH
        self.patch = nn.Linear(PATCH * N_HIST_FEAT, d)
        self.patch_pos = nn.Parameter(torch.randn(n_patch, d) * 0.02)
        self.now = nn.Sequential(nn.Linear(N_HIST_FEAT + N_CTX, d), nn.GELU(), nn.Linear(d, d))
        self.aircraft = AircraftEmbedding(d)
        self.type_emb = nn.Parameter(torch.randn(3, d) * 0.02)      # now, aircraft, patch
        self.layers = encoder_layers(d, n_layers, dropout=dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, hist, ctx, wake, engine, n_engines):
        B = hist.shape[0]
        p = self.patch(hist.reshape(B, -1, PATCH * N_HIST_FEAT)) + self.patch_pos
        now = self.now(torch.cat([hist[:, -1], ctx], -1))
        ac = self.aircraft(wake, engine, n_engines)
        x = torch.cat([(now + self.type_emb[0])[:, None], (ac + self.type_emb[1])[:, None],
                       p + self.type_emb[2]], 1)
        return self.norm(self.layers(x))


class InstructionTokens(nn.Module):
    """Z -> 4 tokens [B, 4, d] (status, lateral, vertical, speed).

    Relative features are computed in torch from per-atom lookup tables, so this runs on
    sampled atoms inside the model with no round trip through numpy or text.
    """

    SLOTS = ("lateral", "vertical", "speed")

    def __init__(self, d=128):
        super().__init__()
        self.status = nn.Embedding(A.SIZES["status"], d)
        self.atom = nn.ModuleDict({s: nn.Embedding(A.SIZES[s], d) for s in self.SLOTS})
        self.feat = nn.ModuleDict({s: nn.Sequential(nn.Linear(A.FEATURE_DIM[s], d), nn.GELU(),
                                                    nn.Linear(d, d)) for s in self.SLOTS})
        self.slot_emb = nn.Parameter(torch.randn(4, d) * 0.02)
        for s in self.SLOTS:
            t = A.SLOTS[s]
            self.register_buffer(f"{s}_kind", torch.tensor(t["kind"]), persistent=False)
            self.register_buffer(f"{s}_sign", torch.tensor(t["sign"]), persistent=False)
            self.register_buffer(f"{s}_value", torch.tensor(t["value"]), persistent=False)

    def features(self, slot, a, trk_deg, alt_ft, gs_kt):
        """Torch twin of atoms.relative_features (tested equal)."""
        kind = getattr(self, f"{slot}_kind")[a]
        sign = getattr(self, f"{slot}_sign")[a]
        val = getattr(self, f"{slot}_value")[a]
        on, keep = (kind == 2).float(), (kind == 1).float()
        if slot == "lateral":
            d = torch.where(sign > 0, torch.remainder(val - trk_deg, 360.0),
                            -torch.remainder(trk_deg - val, 360.0)) * on
            h = torch.deg2rad(val * on + trk_deg * keep)
            dr = torch.deg2rad(d)
            f = torch.stack([torch.sin(h), torch.cos(h), d / 180, torch.sin(dr), torch.cos(dr),
                             on + keep], -1)
        else:
            cur = alt_ft if slot == "vertical" else gs_kt
            scale = 10_000.0 if slot == "vertical" else 100.0
            target = val * on + cur * (1 - on)
            cons = torch.where(on > 0, (torch.sign(target - cur) == sign).float(),
                               torch.ones_like(on))
            f = torch.stack([target / scale, (target - cur) / scale, cons], -1)
        return f * (kind > 0).float()[..., None]

    def forward(self, z, state):
        """z [..., 4] long (status, lateral, vertical, speed); state [..., 3] float
        (track deg, altitude ft, ground speed kt). Leading dims are free."""
        toks = [self.status(z[..., 0]) + self.slot_emb[0]]
        for i, s in enumerate(self.SLOTS, start=1):
            a = z[..., i]
            f = self.features(s, a, state[..., 0], state[..., 1], state[..., 2])
            toks.append(self.atom[s](a) + self.feat[s](f) + self.slot_emb[i])
        return torch.stack(toks, -2)


class TextEncoder(nn.Module):
    """[CS] own callsign [SEP] transcript -> token states [B, L, d] and a padding mask.

    Each word's confidence enters as features (kappa, 1 - kappa, exact flag) added to the
    embedding; the own callsign comes from ADS-B and is marked exact.
    """

    def __init__(self, vocab_size, d=128, n_layers=4, max_len=64, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size + 2, d, padding_idx=0)   # + [CS], [SEP]
        self.cls_id, self.sep_id = vocab_size, vocab_size + 1
        self.pos = nn.Parameter(torch.randn(max_len, d) * 0.02)
        self.role = nn.Embedding(2, d)
        self.conf = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.layers = encoder_layers(d, n_layers, dropout=dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, cs_tokens, tokens, kappa):
        B, Lc = cs_tokens.shape
        dev = tokens.device
        cls = torch.full((B, 1), self.cls_id, device=dev, dtype=torch.long)
        sep = torch.full((B, 1), self.sep_id, device=dev, dtype=torch.long)
        ids = torch.cat([cls, cs_tokens, sep, tokens], 1)
        role = torch.cat([torch.zeros(B, Lc + 2, dtype=torch.long, device=dev),
                          torch.ones_like(tokens)], 1)
        k = torch.cat([torch.ones(B, Lc + 2, device=dev), kappa], 1)
        exact = (role == 0).float()
        pad = ids == 0
        x = (self.tok(ids) + self.pos[:ids.shape[1]] + self.role(role)
             + self.conf(torch.stack([k, 1 - k, exact], -1)))
        return self.norm(self.layers(x, src_key_padding_mask=pad)), pad
