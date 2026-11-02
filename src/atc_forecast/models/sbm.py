"""Semantic belief model (SBM): q(Z | O, X) over four slots of atoms.

Inputs: the transcript with word confidence and the own callsign (TextEncoder), the
history (HistoryEncoder, own weights). Four slot queries (status, lateral, vertical,
speed) read both by cross-attention, exchange information by self-attention, and each ends
in a softmax over its atoms.

Three factorizations share everything but the heads:

  independent  q(Z) = prod_s q(z_s)                          no reveal, one pass
  mixture      q(Z) = sum_k pi_k prod_s q_k(z_s)             K components, one pass
  anyorder     q(Z) = prod_i q(z_pi(i) | z_pi(<i))           exact for any dependence

Any-order training (random reveal): per example draw how many slots to reveal, r in
{0..3}, and which; revealed slots get their true atom's embedding, hidden ones a [mask]
embedding; the loss is the cross-entropy of the hidden slots scaled by 4 / (4 - r), whose
expectation is the negative log-likelihood averaged over orders (Hoogeboom et al., 2022).
Likelihood, sampling and search use the canonical order status -> lateral -> vertical ->
speed; order_spread() measures how far other orders disagree.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..data import atoms as A
from .encoders import HistoryEncoder, TextEncoder

SLOT_NAMES = ("status", "lateral", "vertical", "speed")
SIZES = [A.SIZES[s] for s in SLOT_NAMES]


@dataclass
class SBMConfig:
    vocab_size: int
    d: int = 128
    text_layers: int = 4
    hist_layers: int = 2
    slot_blocks: int = 2
    heads: int = 4
    dropout: float = 0.1
    factorization: str = "anyorder"      # anyorder | independent | mixture
    n_mix: int = 4
    use_history: bool = True


class SlotBlock(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.n1, self.n2, self.n3, self.n4 = (nn.LayerNorm(d) for _ in range(4))
        self.text = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.hist = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.self_ = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * d, d))

    def forward(self, q, text, text_pad, hist, need_weights=False):
        h = self.n1(q)
        a, wt = self.text(h, text, text, key_padding_mask=text_pad, need_weights=need_weights)
        q = q + a
        wh = None
        if hist is not None:
            h = self.n2(q)
            a, wh = self.hist(h, hist, hist, need_weights=need_weights)
            q = q + a
        h = self.n3(q)
        q = q + self.self_(h, h, h, need_weights=False)[0]
        return q + self.mlp(self.n4(q)), (wt, wh)


class SBM(nn.Module):
    def __init__(self, cfg: SBMConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.text = TextEncoder(cfg.vocab_size, d, cfg.text_layers, dropout=cfg.dropout)
        self.hist = HistoryEncoder(d, cfg.hist_layers, cfg.dropout) if cfg.use_history else None
        self.queries = nn.Parameter(torch.randn(4, d) * 0.02)
        self.reveal = nn.ModuleList([nn.Embedding(n + 1, d) for n in SIZES])  # last = [mask]
        self.blocks = nn.ModuleList([SlotBlock(d, cfg.heads, cfg.dropout)
                                     for _ in range(cfg.slot_blocks)])
        self.norm = nn.LayerNorm(d)
        k = cfg.n_mix if cfg.factorization == "mixture" else 1
        self.heads = nn.ModuleList([nn.Linear(d, k * n) for n in SIZES])
        if cfg.factorization == "mixture":
            self.mix = nn.Linear(4 * d, cfg.n_mix)
        self.register_buffer("temperature", torch.ones(4))

    # ------------------------------------------------------------------ encoding
    def encode(self, b):
        text, pad = self.text(b["cs_tokens"], b["tokens"], b["kappa"])
        hist = (self.hist(b["hist"], b["ctx"], b["wake"], b["engine"], b["n_engines"])
                if self.hist is not None else None)
        return text, pad, hist

    def slots(self, enc, revealed, z, need_weights=False):
        """Slot states [B, 4, d]. revealed [B, 4] bool; z [B, 4] long (read where revealed)."""
        text, pad, hist = enc
        B = text.shape[0]
        q = self.queries[None].expand(B, -1, -1)
        emb = []
        for s, n in enumerate(SIZES):
            idx = torch.where(revealed[:, s], z[:, s], torch.full_like(z[:, s], n))
            emb.append(self.reveal[s](idx))
        q = q + torch.stack(emb, 1)
        attn = []
        for blk in self.blocks:
            q, w = blk(q, text, pad, hist, need_weights)
            attn.append(w)
        q = self.norm(q)
        return (q, attn) if need_weights else q

    def logits(self, q):
        """Per-slot logits [B, (K,) n_s], temperature applied."""
        K = self.cfg.n_mix if self.cfg.factorization == "mixture" else 1
        out = []
        for s, n in enumerate(SIZES):
            lg = self.heads[s](q[:, s]) / self.temperature[s]
            out.append(lg.view(-1, K, n) if K > 1 else lg)
        return out

    # ------------------------------------------------------------------ likelihood
    def log_prob(self, b, z, enc=None, order=(0, 1, 2, 3)):
        """log q(z) [B] for complete interpretations z [B, 4]."""
        enc = self.encode(b) if enc is None else enc
        B = z.shape[0]
        f = self.cfg.factorization
        if f == "independent":
            lg = self.logits(self.slots(enc, torch.zeros(B, 4, dtype=torch.bool,
                                                         device=z.device), z))
            return sum(F.log_softmax(lg[s], -1).gather(1, z[:, s:s + 1])[:, 0] for s in range(4))
        if f == "mixture":
            q = self.slots(enc, torch.zeros(B, 4, dtype=torch.bool, device=z.device), z)
            lg = self.logits(q)
            lp = F.log_softmax(self.mix(q.flatten(1)), -1)                  # [B, K]
            for s in range(4):
                ls = F.log_softmax(lg[s], -1)                                # [B, K, n]
                lp = lp + ls.gather(2, z[:, None, s:s + 1].expand(-1, ls.shape[1], 1))[..., 0]
            return torch.logsumexp(lp, -1)
        revealed = torch.zeros(B, 4, dtype=torch.bool, device=z.device)
        total = torch.zeros(B, device=z.device)
        for s in order:
            lg = self.logits(self.slots(enc, revealed, z))[s]
            total = total + F.log_softmax(lg, -1).gather(1, z[:, s:s + 1])[:, 0]
            revealed = revealed.clone()
            revealed[:, s] = True
        return total

    def loss(self, b):
        """Training loss (mean over the batch) and per-slot cross-entropies for logging."""
        z = b["z"]
        B = z.shape[0]
        enc = self.encode(b)
        f = self.cfg.factorization
        if f == "mixture":
            nll = -self.log_prob(b, z, enc)
            return nll.mean(), {}
        if f == "independent":
            revealed = torch.zeros(B, 4, dtype=torch.bool, device=z.device)
        else:
            r = torch.randint(0, 4, (B,), device=z.device)                  # 0..3 revealed
            rank = torch.rand(B, 4, device=z.device).argsort(1).argsort(1)
            revealed = rank < r[:, None]
        lg = self.logits(self.slots(enc, revealed, z))
        ce = torch.stack([F.cross_entropy(lg[s], z[:, s], reduction="none") for s in range(4)],
                         1)                                                  # [B, 4]
        hidden = (~revealed).float()
        w = 4.0 / hidden.sum(1) if f == "anyorder" else torch.ones(B, device=z.device)
        loss = (w * (ce * hidden).sum(1)).mean()
        per_slot = {SLOT_NAMES[s]: (ce[:, s] * hidden[:, s]).sum() / hidden[:, s].sum().clamp(min=1)
                    for s in range(4)}
        return loss, per_slot

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def sample(self, b, n, generators=None, enc=None, p_status=None):
        """n interpretations per window by ancestral sampling in canonical order, [B, n, 4].
        generators: one torch.Generator per window (CPU) for reproducible draws.
        p_status [B, 3]: replaces the model's status distribution, and the motion slots are
        drawn given the status drawn from it (addressee resolution sets who the message is
        for from the whole frequency, evaluation/addressee_v2.py)."""
        enc = self.encode(b) if enc is None else enc
        text, pad, hist = enc
        B = text.shape[0]
        rep = lambda t: None if t is None else t[:, None].expand(-1, n, *t.shape[1:]).reshape(
            B * n, *t.shape[1:])
        enc_r = (rep(text), rep(pad), rep(hist))
        z = torch.zeros(B * n, 4, dtype=torch.long, device=text.device)
        revealed = torch.zeros(B * n, 4, dtype=torch.bool, device=text.device)
        u = (torch.stack([torch.rand(n, 4, generator=g) for g in generators]).reshape(B * n, 4)
             if generators is not None else torch.rand(B * n, 4)).to(text.device)
        f = self.cfg.factorization
        if p_status is not None and f == "mixture":
            raise NotImplementedError("p_status needs a factorization with a status-first order")
        if f == "mixture":
            q = self.slots(enc_r, revealed, z)
            comp = torch.multinomial(F.softmax(self.mix(q.flatten(1)), -1), 1)[:, 0]
            lg = self.logits(q)
            for s in range(4):
                p = F.softmax(lg[s][torch.arange(B * n), comp], -1)
                z[:, s] = _inverse_cdf(p, u[:, s])
            return z.view(B, n, 4)
        lg_all = None
        for s in range(4):
            # independent heads ignore reveals: one pass serves every slot
            if lg_all is None or f != "independent":
                lg_all = self.logits(self.slots(enc_r, revealed, z))
            p = (F.softmax(lg_all[s], -1) if s or p_status is None
                 else p_status.to(text.device).repeat_interleave(n, 0))
            z[:, s] = _inverse_cdf(p, u[:, s])
            revealed = revealed.clone()
            revealed[:, s] = True
        return z.view(B, n, 4)

    @torch.no_grad()
    def status_probs(self, b, enc=None):
        """q(status | O, X) [B, 3], nothing revealed: own motion, own non-motion, other."""
        enc = self.encode(b) if enc is None else enc
        B = enc[0].shape[0]
        none = torch.zeros(B, 4, dtype=torch.bool, device=enc[0].device)
        q = self.slots(enc, none, torch.zeros(B, 4, dtype=torch.long, device=enc[0].device))
        lg = self.logits(q)[0]
        if self.cfg.factorization == "mixture":
            w = F.softmax(self.mix(q.flatten(1)), -1)
            return (w[..., None] * F.softmax(lg, -1)).sum(1)
        return F.softmax(lg, -1)

    @torch.no_grad()
    def beam(self, b, width=16, enc=None):
        """Beam search in canonical order. Returns (z [B, width, 4], logp [B, width],
        certified [B]). certified is True when no pruned partial interpretation scored
        above the best complete one: every factor is <= 1, so a pruned partial can only
        lose probability, and the returned mode is then the exact joint mode."""
        enc = self.encode(b) if enc is None else enc
        text, pad, hist = enc
        B, dev = text.shape[0], text.device
        z = torch.zeros(B, 1, 4, dtype=torch.long, device=dev)
        lp = torch.zeros(B, 1, device=dev)
        best_pruned = torch.full((B,), -float("inf"), device=dev)
        for s in range(4):
            W = z.shape[1]
            rep = lambda t: None if t is None else t[:, None].expand(-1, W, *t.shape[1:]).reshape(
                B * W, *t.shape[1:])
            revealed = torch.zeros(B * W, 4, dtype=torch.bool, device=dev)
            revealed[:, :s] = True
            if self.cfg.factorization == "independent":
                revealed[:] = False
            lg = self.logits(self.slots((rep(text), rep(pad), rep(hist)), revealed,
                                        z.reshape(B * W, 4)))[s]
            if lg.dim() == 3:                          # mixture: marginalize components
                raise NotImplementedError("beam search uses anyorder or independent heads")
            cand = lp[..., None] + F.log_softmax(lg, -1).view(B, W, -1)     # [B, W, n_s]
            flat = cand.reshape(B, -1)
            k = min(width, flat.shape[1])
            top, idx = flat.topk(k, -1)
            if flat.shape[1] > k:
                pruned = flat.scatter(1, idx, -float("inf")).max(1).values
                best_pruned = torch.maximum(best_pruned, pruned)
            parent, atom = idx // cand.shape[2], idx % cand.shape[2]
            z = z.gather(1, parent[..., None].expand(-1, -1, 4)).clone()
            z[:, :, s] = atom
            lp = top
        return z, lp, lp[:, 0] >= best_pruned

    @torch.no_grad()
    def order_spread(self, b, z, orders=None):
        """Std of log q(z) across slot orders: 0 for a perfectly consistent model."""
        import itertools
        orders = orders or list(itertools.permutations(range(4)))
        enc = self.encode(b)
        vals = torch.stack([self.log_prob(b, z, enc, order=o) for o in orders], 1)
        return vals.std(1)


def _inverse_cdf(p, u):
    c = p.cumsum(-1)
    return torch.searchsorted(c, (u * c[:, -1])[:, None]).clamp(max=p.shape[-1] - 1)[:, 0]


def save_sbm(model: SBM, path, extra=None):
    torch.save({"cfg": asdict(model.cfg), "state": model.state_dict(), **(extra or {})}, path)


def load_sbm(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = SBM(SBMConfig(**ck["cfg"])).to(device)
    m.load_state_dict(ck["state"])
    return m.eval(), ck


def window_generators(window_ids, run_seed=0, salt=1):
    return [torch.Generator().manual_seed(hash((int(run_seed), int(w), salt)) & 0x7FFFFFFF)
            for w in np.asarray(window_ids).tolist()]
