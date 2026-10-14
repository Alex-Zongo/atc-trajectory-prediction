"""Stage 1: an autoregressive posterior over the clearance, q(delta | u, kappa, s).

Given a noisy transcript u, a recognizer confidence kappa and the aircraft's current state
s, this models the FULL JOINT posterior over the ten-field clearance autoregressively:

    q(delta | u) = p(clr) * prod_axis p(present | .) p(direction | .) p(magnitude | .)

    field order:  is_clearance
                  -> pres_lat  -> dir_lat  -> heading
                  -> pres_vert -> dir_vert -> altitude
                  -> pres_spd  -> dir_spd  -> speed

Why autoregressive rather than per-field independent. A factorized model draws every field
independently, so a sampled clearance can be incoherent (mishearing the altitude but not
its presence) and the posterior can never be jointly multimodal, which is exactly what
"turn left 270 or turn right 090" requires. Here each conditional reads a GRU state that
has consumed the encoder context and every previously decoded field, so the fields co-vary;
correlation and multimodality come for free, and the likelihood is EXACT, which is what
makes the posterior directly calibratable. There is no latent variable, so no posterior
collapse.

Heading is a CATEGORICAL over 72 five-degree bins, not a circular unimodal distribution: a
digit-by-digit corruption of a three-digit heading has a multimodal posterior (200 becoming
290, 100 or 300) that a unimodal distribution cannot represent, which is why its
concentration parameter never calibrated. Altitude and speed default to small Gaussian
mixtures so their marginals can be multimodal and heavy-tailed; the `cat` alternative puts
them on a physical grid whose bin widths match the phraseology rounding exactly, so every
issued value lands on a bin centre and the head is lossless with respect to the label.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..units import (ALT_BIN_FT, ALT_SCALE_FT, HEAD_BIN_DEG, N_ALT_BINS, N_HEAD_BINS,
                     N_SPD_BINS, SPD_BIN_KT, SPD_SCALE_KT)
from .backbone import EncoderConfig, SlotAttentionPool, StateFiLM, TextBackbone

K_MIX = 4
LOGS_CLAMP = (-8.0, 4.0)
N_SLOTS_DEFAULT = 6

# (name, head type, gate key). The ORDER IS the autoregressive factorization.
FIELDS = [
    ("is_clearance", "bern", None),
    ("pres_lat", "bern", None),
    ("dir_lat", "bern", "has_turn"),
    ("hdg", "cat", "has_turn"),
    ("pres_vert", "bern", None),
    ("dir_vert", "bern", "has_vert"),
    ("alt", "mix", "has_vert"),
    ("pres_spd", "bern", None),
    ("dir_spd", "bern", "has_speed"),
    ("spd", "mix", "has_speed"),
]
TARGET_KEY = {"is_clearance": "is_clearance", "pres_lat": "has_turn", "pres_vert": "has_vert",
              "pres_spd": "has_speed", "dir_lat": "turn_dir", "dir_vert": "vert_dir",
              "dir_spd": "spd_dir", "hdg": "head_bin", "alt": "alt_norm", "spd": "spd_norm"}
# gate key to the decoded field carrying it, so a clause's log-probability counts only
# where the clause is present; an absent clause's magnitude is unconstrained
GATE_FIELD = {"has_turn": "pres_lat", "has_vert": "pres_vert", "has_speed": "pres_spd"}

N_CLASS = {"hdg": N_HEAD_BINS, "alt": N_ALT_BINS, "spd": N_SPD_BINS}
BIN_UNIT = {"alt": ALT_BIN_FT, "spd": SPD_BIN_KT}
VAL_SCALE = {"alt": (ALT_SCALE_FT, ALT_BIN_FT, N_ALT_BINS),
             "spd": (SPD_SCALE_KT, SPD_BIN_KT, N_SPD_BINS)}


def mixture_nll(pi_logits, mu, logs, y):
    """Negative log-likelihood of y under a 1-D Gaussian mixture."""
    logpi = F.log_softmax(pi_logits, dim=-1)
    logn = -0.5 * (((y.unsqueeze(-1) - mu) * torch.exp(-logs)) ** 2 + 2 * logs
                   + math.log(2 * math.pi))
    return -torch.logsumexp(logpi + logn, dim=-1)


def mixture_sample(pi_logits, mu, logs):
    comp = torch.multinomial(F.softmax(pi_logits, -1), 1)
    m = mu.gather(-1, comp).squeeze(-1)
    s = logs.gather(-1, comp).squeeze(-1)
    return m + torch.exp(s) * torch.randn_like(m)


class ClearancePosterior(nn.Module):
    """The Stage-1 model. `sample` draws coherent joint clearances; `greedy` gives the MAP."""

    def __init__(self, cfg: EncoderConfig, k_mix: int = K_MIX, state_dim: int = 0,
                 ground_fields=("hdg", "alt", "spd"), cont_head: str = "mix",
                 n_slots: int | None = None):
        super().__init__()
        assert cont_head in ("mix", "cat")
        H = cfg.hidden
        self.k_mix = k_mix
        self.state_dim = state_dim
        self.cont_head = cont_head
        self._fields = [(n, "cat" if (cont_head == "cat" and n in ("alt", "spd")) else t, g)
                        for n, t, g in FIELDS]
        self._types = {n: t for n, t, _ in self._fields}

        # Optionally ground the NUMERIC heads in the current state (onset altitude, vertical
        # rate, ground speed, heading sin/cos) so physical plausibility can re-rank misheard
        # or dropped digits. Presence and direction heads are left alone.
        self.ground_fields = tuple(ground_fields) if state_dim > 0 else ()
        self.state_films = (nn.ModuleDict({f: StateFiLM(state_dim, H) for f in self.ground_fields})
                            if state_dim > 0 else None)

        self.backbone = TextBackbone(cfg)
        self.n_slots = N_SLOTS_DEFAULT if n_slots is None else int(n_slots)
        self.slots = SlotAttentionPool(H, self.n_slots, cfg.n_heads)
        self.ctx_proj = nn.Sequential(nn.Linear(self.n_slots * H, H), nn.GELU(), nn.LayerNorm(H))
        self.h0 = nn.Linear(H, H)
        self.gru = nn.GRUCell(H, H)
        self.field_emb = nn.Parameter(torch.randn(len(FIELDS), H) * 0.02)
        self.embed = nn.ModuleDict()
        self.head = nn.ModuleDict()
        for name, typ, _ in self._fields:
            nb = N_CLASS.get(name, N_HEAD_BINS)
            self.embed[name] = nn.Embedding(nb, H) if typ == "cat" else nn.Linear(1, H)
            self.head[name] = nn.Linear(H, {"bern": 1, "cat": nb, "mix": 3 * k_mix}[typ])
        # post-hoc recalibration knob, fitted by evaluation.stage1.fit_temperature
        self.temperature = 1.0

    def encode(self, tokens, confidence):
        states, pad = self.backbone(tokens, confidence)
        return self.ctx_proj(self.slots(states, pad).flatten(1))

    @torch.no_grad()
    def slot_attention(self, tokens, confidence):
        """Which tokens each slot reads, [B, n_slots, 1 + L]. For interpretability."""
        states, pad = self.backbone(tokens, confidence)
        _, attn = self.slots(states, pad, return_attn=True)
        return attn

    def _embed_value(self, name, v):
        return (self.embed[name](v.long()) if self._types[name] == "cat"
                else self.embed[name](v.float().unsqueeze(-1)))

    def _mix_params(self, raw):
        pi, mu, logs = raw.split(self.k_mix, dim=-1)
        return pi, mu, logs.clamp(*LOGS_CLAMP)

    def _cat_target(self, name, batch):
        """Bin a normalised magnitude onto the categorical grid. Lossless, because the bin
        widths match the phraseology rounding."""
        scale, binw, nb = VAL_SCALE[name]
        src = batch["alt_norm" if name == "alt" else "spd_norm"]
        return (src * scale / binw).round().clamp(0, nb - 1)

    def _state(self, state):
        """Slice the grounding state to this model's state_dim. The full state
        [alt, vrate, gs, sin, cos] is prefix-compatible, so a narrower model takes a prefix."""
        return None if (self.state_films is None or state is None) else state[..., :self.state_dim]

    def _apply_film(self, name, h, state):
        return (self.state_films[name](h, state)
                if (state is not None and self.state_films is not None
                    and name in self.state_films) else h)

    def loss(self, batch, ss_p: float = 0.0) -> dict:
        """The joint negative log-likelihood of the clearance.

            -log q(delta | u, kappa, s) = -sum_i [gate_i] log q(delta_i | delta_<i, .)

        summed over fields and averaged over the BATCH, so each field contributes in
        proportion to how often it actually occurs. Averaging each field over only the rows
        where it is present and then summing would silently upweight rare clauses by
        1 / P(present), measured at 2.0x for lateral, 1.4x for vertical and 4.1x for speed.
        That is a departure from the likelihood, and the likelihood is what makes this model
        calibrated. Note that the fields being on different scales is NOT a reason to
        reweight: summing per-field NLLs is exactly the chain rule, and the unit constants
        in the continuous heads contribute an additive constant with zero gradient.

        `ss_p` above 0 turns on SCHEDULED SAMPLING: with that probability a step advances
        the recurrence on the model's own draw instead of the true value. Training is
        otherwise fully teacher-forced while inference is free-running, so the chain would
        never have seen its own mistakes, and it is exactly that chain which supports the
        claim that the ten fields co-vary coherently. The fed-back draw is DETACHED, so this
        changes what the state sees and not the gradient path, and supervision always uses
        the true gate and the true target.
        """
        h = self.h0(self.encode(batch["tokens"], batch["confidence"]))
        state = self._state(batch.get("state"))
        ones = torch.ones_like(batch["is_clearance"])
        B = h.shape[0]
        parts, total = {}, h.new_zeros(())
        for i, (name, typ, gate) in enumerate(self._fields):
            raw = self.head[name](self._apply_film(name, h, state))
            y = (self._cat_target(name, batch) if (typ == "cat" and name in VAL_SCALE)
                 else batch[TARGET_KEY[name]])
            m = (batch[gate] if gate is not None else ones) > 0.5
            if typ == "bern":
                per = F.binary_cross_entropy_with_logits(raw.squeeze(-1), y.float(),
                                                         reduction="none")
                draw = torch.bernoulli(torch.sigmoid(raw.squeeze(-1).detach()))
            elif typ == "cat":
                per = F.cross_entropy(raw, y.long(), reduction="none")
                draw = torch.multinomial(F.softmax(raw.detach(), -1), 1).squeeze(-1).float()
            else:
                pi, mu, logs = self._mix_params(raw)
                per = mixture_nll(pi, mu, logs, y.float())
                draw = mixture_sample(pi.detach(), mu.detach(), logs.detach())
            total = total + (per * m).sum() / B
            parts[name] = (per[m].mean() if m.any() else per.new_zeros(())).detach()
            fed = y
            if ss_p > 0.0:
                use = torch.rand(B, device=h.device) < ss_p
                fed = torch.where(use, draw.to(y.dtype), y)
            h = self.gru(self._embed_value(name, fed) + self.field_emb[i], h)
        return {"loss": total, "clearance": parts["is_clearance"], "heading": parts["hdg"],
                "altitude": parts["alt"],
                "presence": (parts["pres_lat"] + parts["pres_vert"] + parts["pres_spd"]) / 3}

    def _assemble(self, g: dict) -> dict:
        """Field dict to physical units. The magnitude multiplier depends on the head: a
        categorical draw is a bin index, a mixture draw is a normalised value."""
        alt_mult = BIN_UNIT["alt"] if self._types["alt"] == "cat" else ALT_SCALE_FT
        spd_mult = BIN_UNIT["spd"] if self._types["spd"] == "cat" else SPD_SCALE_KT
        return {"is_clearance": g["is_clearance"],
                "has_turn": g["pres_lat"], "turn_right": g["dir_lat"],
                "heading_deg": (g["hdg"] * HEAD_BIN_DEG) % 360,
                "has_vert": g["pres_vert"], "climb": g["dir_vert"],
                "altitude_ft": g["alt"] * alt_mult,
                "has_speed": g["pres_spd"], "accel": g["dir_spd"],
                "speed_kt": g["spd"] * spd_mult}

    @torch.no_grad()
    def _decode(self, tokens, confidence, k: int, sample: bool, temperature: float = 1.0,
                state=None, return_logp: bool = False):
        """Run the chain k times.

        `sample=True` gives ancestral, co-varying draws; `sample=False` gives the greedy MAP
        (argmax for discrete heads, mixture mean for continuous). `temperature` below 1
        sharpens the SAMPLING spread and leaves the greedy MAP unchanged.
        """
        ctx = self.encode(tokens, confidence)
        B = ctx.shape[0]
        h = self.h0(ctx).repeat_interleave(k, 0)
        state = self._state(state)
        state_k = state.repeat_interleave(k, 0) if state is not None else None
        T = max(temperature, 1e-3)
        out = {}
        logp = h.new_zeros(h.shape[0]) if return_logp else None
        for i, (name, typ, gate) in enumerate(self._fields):
            raw = self.head[name](self._apply_film(name, h, state_k))
            if typ == "bern":
                p = torch.sigmoid(raw.squeeze(-1) / T)
                v = (torch.bernoulli(p) if sample
                     else (torch.sigmoid(raw.squeeze(-1)) > 0.5).float())
                lp = torch.log(torch.where(v > 0.5, p, 1 - p).clamp_min(1e-12)) if return_logp else None
            elif typ == "cat":
                v = (torch.multinomial(F.softmax(raw / T, -1), 1).squeeze(-1).float() if sample
                     else raw.argmax(-1).float())
                lp = (F.log_softmax(raw / T, -1).gather(-1, v.long().unsqueeze(-1)).squeeze(-1)
                      if return_logp else None)
            else:
                pi, mu, logs = self._mix_params(raw)
                logs_t = (logs + math.log(T)).clamp(*LOGS_CLAMP)
                v = (mixture_sample(pi, mu, logs_t) if sample
                     else (F.softmax(pi, -1) * mu).sum(-1))
                lp = -mixture_nll(pi, mu, logs_t, v) if return_logp else None
            if return_logp:
                m = (out[GATE_FIELD[gate]].reshape(-1) if gate is not None
                     else torch.ones_like(lp))
                logp = logp + lp * (m > 0.5).float()
            out[name] = v.view(B, k)
            h = self.gru(self._embed_value(name, v) + self.field_emb[i], h)
        return (out, logp.view(B, k)) if return_logp else out

    @torch.no_grad()
    def sample(self, tokens, confidence, k: int = 8, temperature=None, state=None,
               return_logp: bool = False):
        """Draw k coherent joint clearances, as a dict of [B, k] tensors whose fields co-vary.

        `return_logp` also returns the joint log-probability of each drawn clearance, for
        ranking draws rather than taking them as-is. Those values MIX probabilities
        (presence, direction, heading bin) with log DENSITIES (mixture magnitudes), so they
        are valid for ranking candidates but are not a normalised probability over
        clearances.
        """
        T = self.temperature if temperature is None else temperature
        r = self._decode(tokens, confidence, k, True, T, state, return_logp)
        if return_logp:
            g, lp = r
            return self._assemble(g), lp
        return self._assemble(r)

    @torch.no_grad()
    def greedy(self, tokens, confidence, state=None) -> dict:
        """The single MAP clearance, as a dict of [B] tensors. This is the POINT estimate."""
        return {k: v.squeeze(1) for k, v in
                self._assemble(self._decode(tokens, confidence, 1, False, state=state)).items()}


def save_stage1(model: ClearancePosterior, vocab, path, **extra):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "vocab": vocab.itos,
                "state_dim": model.state_dim, "ground_fields": list(model.ground_fields),
                "cont_head": model.cont_head, "n_slots": model.n_slots,
                "temperature": float(model.temperature), **extra}, path)


def load_stage1(path, device="cpu"):
    """Rebuild a Stage-1 model and its vocabulary from a checkpoint.

    Architecture is read from the checkpoint where recorded and otherwise inferred from the
    weight shapes, so older checkpoints still load.
    """
    from ..data.text import Vocab

    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd, itos = ck["model"], ck["vocab"]
    gf = ck.get("ground_fields") or sorted({k.split(".")[1] for k in sd
                                            if k.startswith("state_films.")})
    sdim = ck.get("state_dim")
    if sdim is None:
        sdim = sd[f"state_films.{gf[0]}.net.0.weight"].shape[1] if gf else 0
    cont = ck.get("cont_head") or ("mix" if sd["head.alt.weight"].shape[0] == 3 * K_MIX else "cat")
    cfg = EncoderConfig(vocab_size=len(itos), max_len=sd["backbone.pos.weight"].shape[0] - 1)
    model = ClearancePosterior(cfg, state_dim=sdim, cont_head=cont,
                               ground_fields=tuple(gf) or ("hdg", "alt", "spd"),
                               n_slots=ck.get("n_slots"))
    model.load_state_dict(sd)
    # a plain attribute, so it must be re-applied on load or the model runs uncalibrated
    model.temperature = float(ck.get("temperature", 1.0))
    model.eval().to(device)
    return model, Vocab.from_itos(itos), cfg.max_len
