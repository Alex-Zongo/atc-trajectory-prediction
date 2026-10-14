"""Stage 2: a conditional flow-matching DiT over future control sequences, p(tau | delta, c).

Predicts the distribution of future CONTROLS [H x 4] = [gs, hdg_sin, hdg_cos, vrate] given
the past, the clearance and the aircraft. Positions are dead-reckoned from the controls for
the distance metrics. Predicting controls rather than positions matches what a clearance
actually commands.

Conditioning:

  past        self-attention encoder with FiLM(aircraft), encoded ONCE into context tokens
              the future cross-attends to
  clearance   three cross-attention tokens, one per control axis a controller commands,
              each carrying that axis's complete command (presence, direction, target). The
              future retrieves whichever axis it needs. Splitting direction from target
              into five tokens is arbitrary; three axis tokens is the grounded set.
  aircraft    FiLM on the past encoder, and folded into the adaLN conditioning, so it can
              rescale the predicted control RATES
  flow-time   adaLN

Positions are never noised, only the future control values. Rectified flow: x0 ~ N(0, I),
x1 = true controls, x_t = (1 - t) x0 + t x1, target velocity v = x1 - x0; the network
regresses v and sampling integrates the ODE from 0 to 1.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..units import (ALT_SCALE_FT, ENGINE_VOCAB, GS_SCALE_KT, KT_TO_MS, POS_SCALE_M,
                     VR_SCALE_FPM, WAKE_VOCAB)
from .backbone import AircraftEmbed, ContinuousTimeEmbed, FiLM

H_STEPS, P_STEPS = 60, 30
N_CTRL, N_PAST_F = 4, 7


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def last_valid_index(mask):
    """Index of the LAST valid step per row, and whether the row has any.

    `mask.sum(1) - 1` is only the last valid index when the valid steps form an unbroken run
    from step 0. Windows with an internal coverage hole do occur (around 1.8 percent of
    ours), and on those the count-based expression is short by a mean of 18 steps and
    sometimes points at a step that is itself invalid, which scores the forecast against a
    fabricated interpolated position. Every endpoint consumer must use this.

    Rows with no valid step get last = 0 and has_any = False; callers must handle those
    rather than indexing blindly.
    """
    m = mask.bool()
    idx = torch.arange(m.shape[1], device=m.device)
    last = torch.where(m, idx[None, :], torch.full_like(idx[None, :], -1)).amax(1)
    return last.clamp(min=0), last >= 0


def horizon_weights(H, device, mode="integrated"):
    """Per-step loss weights, normalised to mean 1.

    The flow predicts CONTROLS but every metric scores POSITION, which is their running
    sum. An error at step i therefore displaces all H - i later positions while an error at
    the last step displaces one, so weighting steps equally under-penalises early error by
    up to a factor of H. `integrated` weights step i by how many positions it still affects.
    """
    if mode == "uniform":
        return torch.ones(H, device=device)
    w = H - torch.arange(H, device=device, dtype=torch.float32)
    return w / w.mean()


def deadreckon(ctrl, onset_alt_ft):
    """Controls to normalised (dx, dy, alt), matching the fut_pos layout."""
    gs = ctrl[..., 0] * GS_SCALE_KT * KT_TO_MS
    vx, vy = gs * ctrl[..., 1], gs * ctrl[..., 2]
    vr = ctrl[..., 3] * VR_SCALE_FPM / 60.0

    def integ(v):
        c = torch.cumsum(v, 1)
        out = torch.zeros_like(c)
        out[:, 1:] = c[:, :-1]              # position BEFORE this step
        return out

    return torch.stack([integ(vx) / POS_SCALE_M, integ(vy) / POS_SCALE_M,
                        (onset_alt_ft[:, None] + integ(vr)) / ALT_SCALE_FT], -1)


class PastEncoder(nn.Module):
    def __init__(self, dim, heads, layers, p_steps=P_STEPS, n_past_f=N_PAST_F, dropout=0.1,
                 time_encoding=False):
        super().__init__()
        self.p_steps = p_steps
        self.time_encoding = time_encoding
        self.proj = nn.Linear(n_past_f, dim)
        # the raw-observation past embeds each token's real time_to_onset; the grid past
        # uses a learned per-step embedding
        if time_encoding:
            self.time_emb = ContinuousTimeEmbed(dim)
        else:
            self.pos = nn.Embedding(p_steps, dim)
        self.film = FiLM(dim, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def _pos_term(self, past, past_time):
        if self.time_encoding:
            return self.time_emb(past_time)
        return self.pos(torch.arange(self.p_steps, device=past.device))[None]

    def forward(self, past, past_mask, aircraft, past_time=None):
        h = self.proj(past) + self._pos_term(past, past_time)
        h = self.film(h, aircraft)
        return self.tf(h, src_key_padding_mask=~past_mask.bool())

    @torch.no_grad()
    def self_attention(self, past, past_mask, aircraft, past_time=None):
        """Past-to-past attention per layer, head-averaged, [B, layers, P, P]. Replays the
        pre-LN layer forward manually so the weights can be requested."""
        h = self.film(self.proj(past) + self._pos_term(past, past_time), aircraft)
        pad = ~past_mask.bool()
        attns = []
        for layer in self.tf.layers:
            x = layer.norm1(h)
            a, w = layer.self_attn(x, x, x, key_padding_mask=pad, need_weights=True,
                                   average_attn_weights=True)
            h = h + a
            h = h + layer.linear2(layer.activation(layer.linear1(layer.norm2(h))))
            attns.append(w)
        return torch.stack(attns, dim=1)


class ClearanceTokens(nn.Module):
    """One cross-attention token per control axis, each carrying its complete command."""

    def __init__(self, dim):
        super().__init__()
        self.lateral = nn.Linear(4, dim)       # [has_turn, turn_right, hdg_sin, hdg_cos]
        self.vertical = nn.Linear(3, dim)      # [has_vert, climb, alt_norm]
        self.speed = nn.Linear(3, dim)         # [has_speed, accel, spd_norm]
        self.type_emb = nn.Parameter(torch.randn(3, dim) * 0.02)
        # The classifier-free-guidance "unconditional" tokens: a DROPPED clearance, which is
        # distinct from a null one. "Maintain present heading" is valid information; dropped
        # means no information at all.
        self.uncond = nn.Parameter(torch.randn(3, dim) * 0.02)

    def forward(self, d, drop=None):
        toks = torch.stack([self.lateral(d[:, 0:4]), self.vertical(d[:, 4:7]),
                            self.speed(d[:, 7:10])], dim=1) + self.type_emb[None]
        if drop is not None and drop.any():
            u = (self.uncond + self.type_emb)[None].expand(d.shape[0], -1, -1)
            toks = torch.where(drop[:, None, None], u, toks)
        return toks


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.n3 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.adaln = nn.Linear(dim, 9 * dim)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)                 # adaLN-zero

    def forward(self, h, cond, memory, mem_pad, fut_pad=None, return_attn=False):
        s1, c1, g1, s2, c2, g2, s3, c3, g3 = self.adaln(cond).chunk(9, -1)
        sa_in = modulate(self.n1(h), s1, c1)
        sa_out, self_w = self.attn(sa_in, sa_in, sa_in, key_padding_mask=fut_pad,
                                   need_weights=return_attn, average_attn_weights=False)
        h = h + g1.unsqueeze(1) * sa_out
        q = modulate(self.n2(h), s2, c2)
        cross_out, cross_w = self.cross(q, memory, memory, key_padding_mask=mem_pad,
                                        need_weights=return_attn, average_attn_weights=False)
        h = h + g2.unsqueeze(1) * cross_out
        h = h + g3.unsqueeze(1) * self.mlp(modulate(self.n3(h), s3, c3))
        return (h, self_w, cross_w) if return_attn else h


class FlowPredictor(nn.Module):
    def __init__(self, dim=128, heads=4, layers=4, past_layers=2, cfg_drop_p=0.1,
                 h_steps=H_STEPS, p_steps=P_STEPS, n_past_f=N_PAST_F, time_encoding=False,
                 source="gauss", src_sigma=1.0, step_weight="integrated", pos_loss_w=0.0,
                 cfg_drop_end=None):
        super().__init__()
        assert source in ("gauss", "dr")
        # Flow SOURCE distribution. "gauss" is the standard N(0, I). "dr" centres the source
        # on the onset control held constant over the horizon, so the flow learns the
        # DEVIATION from ballistic continuation rather than the trajectory from scratch.
        self.source = source
        self.src_sigma = float(src_sigma)
        self.step_weight = step_weight
        self.pos_loss_w = float(pos_loss_w)
        # cfg_drop_p can be ANNEALED by setting cfg_drop_end. This is not an ordinary
        # optimisation knob: the unconditional branch IS the no-instruction baseline that
        # every reported clearance gain is measured against. Too low and the gain is
        # flattered; too high and it is understated. Ramping DOWN trains a solid baseline
        # first and then specialises; ramping up would undertrain exactly the branch the
        # null comparison needs. Change it as a deliberate ablation, never alongside
        # anything else.
        self.cfg_drop_p = cfg_drop_p
        self.cfg_drop_start = cfg_drop_p
        self.cfg_drop_end = cfg_drop_p if cfg_drop_end is None else float(cfg_drop_end)
        self.h_steps, self.p_steps = h_steps, p_steps
        self.time_encoding = time_encoding
        self.dim = dim

        self.aircraft = AircraftEmbed(dim)
        self.past_enc = PastEncoder(dim, heads, past_layers, p_steps, n_past_f,
                                    time_encoding=time_encoding)
        self.dir_tok = ClearanceTokens(dim)
        self.fut_proj = nn.Linear(N_CTRL, dim)
        self.fut_pos = nn.Embedding(h_steps, dim)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(layers)])
        self.final_n = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_ada = nn.Linear(dim, 2 * dim)
        self.head = nn.Linear(dim, N_CTRL)
        for m in (self.final_ada, self.head):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def _time_emb(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None] * freqs[None]
        return self.time_mlp(torch.cat([a.sin(), a.cos()], -1))

    def encode_cond(self, batch, drop=None):
        ac = self.aircraft(batch["wake"], batch["engine"], batch["n_engines"])
        past_tok = self.past_enc(batch["past"], batch["past_mask"], ac, batch.get("past_time"))
        dir_tok = self.dir_tok(batch["directive"], drop=drop)
        memory = torch.cat([past_tok, dir_tok], 1)
        B, K = memory.shape[0], dir_tok.shape[1]
        mem_pad = torch.cat([~batch["past_mask"].bool(),
                             torch.zeros(B, K, dtype=torch.bool, device=memory.device)], 1)
        return memory, mem_pad, ac

    def velocity(self, xt, t, memory, mem_pad, ac, fut_mask=None, return_attn=False):
        cond = self._time_emb(t) + ac
        h = self.fut_proj(xt) + self.fut_pos(torch.arange(self.h_steps, device=xt.device))[None]
        fut_pad = (~fut_mask.bool()) if fut_mask is not None else None
        self_attns, cross_attns = [], []
        for blk in self.blocks:
            if return_attn:
                h, sw, cw = blk(h, cond, memory, mem_pad, fut_pad=fut_pad, return_attn=True)
                self_attns.append(sw)
                cross_attns.append(cw)
            else:
                h = blk(h, cond, memory, mem_pad, fut_pad=fut_pad)
        shift, scale = self.final_ada(cond).chunk(2, -1)
        v = self.head(modulate(self.final_n(h), shift, scale))
        return (v, self_attns, cross_attns) if return_attn else v

    def _source(self, batch, B, device, k=1):
        """Draw x0. MUST be identical in training and sampling.

        Do NOT try to learn src_sigma through the flow loss. The regression target is
        x1 - x0 = x1 - mu - sigma*eps, and eps is not recoverable from x_t at large t, so
        sigma contributes irreducible loss that gradient descent removes by driving it to
        zero. A zero-sigma source is deterministic, which turns the flow into a
        deterministic map and destroys the sample diversity the whole distributional claim
        rests on. Fit it post hoc against a proper scoring rule, as lambda is fitted.
        """
        eps = torch.randn(B, self.h_steps, N_CTRL, device=device)
        if self.source == "gauss":
            return eps
        # the control channels are exactly past features [3:7], so the onset control is
        # past[:, -1, 3:7] broadcast over the horizon
        mu = batch["past"][:, -1, 3:3 + N_CTRL]
        if k > 1:
            mu = mu.repeat_interleave(k, 0)
        return mu[:, None, :].expand(-1, self.h_steps, -1) + self.src_sigma * eps

    def flow_loss(self, batch):
        """Masked rectified-flow regression.

        No future attention mask. Passing `fut_mask` to `velocity` at SAMPLING time is a
        leak: the mask is built from the next clearance's onset and post-hoc coverage, so it
        tells the forecaster how long the maneuver lasted, which is not available at
        prediction time. Its only legitimate job was in training, keeping valid queries from
        attending to fabricated interpolated values at invalid steps. That need is removed
        instead: invalid steps are OVERWRITTEN with the source sample, so they carry no
        target information and attending to them leaks nothing, and training and sampling
        then run the same unmasked attention.

        Per-window normalisation before averaging across windows. Normalising globally
        weights each window by its valid length, so fully observed windows (about a third of
        the data) contributed around half the loss.
        """
        x1 = batch["fut_ctrl"]
        x0 = self._source(batch, x1.shape[0], x1.device)
        m = batch["fut_mask"][:, :, None]
        x1 = torch.where(m > 0.5, x1, x0)
        t = torch.rand(x1.shape[0], device=x1.device)
        xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
        drop = torch.rand(x1.shape[0], device=x1.device) < self.cfg_drop_p
        memory, mem_pad, ac = self.encode_cond(batch, drop=drop)
        pred = self.velocity(xt, t, memory, mem_pad, ac, fut_mask=None)
        target = x1 - x0
        w = horizon_weights(x1.shape[1], x1.device, self.step_weight)[None, :, None] * m
        per_win = ((pred - target) ** 2 * w).sum((1, 2)) / (w.sum((1, 2)) * N_CTRL + 1e-8)
        loss = per_win.mean()
        if self.pos_loss_w > 0.0:
            # Auxiliary TASK-SPACE term: reconstruct x1 from the predicted field, dead-reckon
            # it, and score the positions actually reported.
            #
            # Dead reckoning is a cumulative sum, so a position at step i depends on every
            # step before it. At an invalid step x1 was replaced by the source sample, so
            # integrating through it corrupts every later position, and masking the loss
            # afterwards does not undo that: a VALID step sitting after an internal gap still
            # has a corrupted position. So this term applies only on the contiguous valid
            # PREFIX, which is where the integral is clean.
            pm = torch.cumprod(m.squeeze(-1), dim=1)[:, :, None]
            x1_hat = xt + (1 - t)[:, None, None] * pred
            oa = batch["past"][:, -1, 2] * ALT_SCALE_FT
            dp = (deadreckon(x1_hat, oa) - batch["fut_pos"]) ** 2
            loss = loss + self.pos_loss_w * ((dp * pm).sum((1, 2))
                                             / (pm.sum((1, 2)) * 3 + 1e-8)).mean()
        return loss

    @torch.no_grad()
    def sample(self, batch, n_steps=20, k=1, guidance=1.0):
        """Integrate the flow ODE.

        `guidance` above 1 is classifier-free guidance, amplifying the clearance as
        v = v_uncond + guidance * (v_cond - v_uncond). `guidance = 0` drops the clearance
        entirely, which is the no-instruction baseline.

        `batch["fut_mask"]` is deliberately NOT read here: it is retrospective target
        bookkeeping and is unavailable at prediction time. The forecaster generates the full
        requested horizon and validity is applied outside the model, at scoring time.
        """
        mem_c, mem_pad, ac = self.encode_cond(batch, drop=None)
        mem_u = None
        if guidance != 1.0:
            drop = torch.ones(mem_c.shape[0], dtype=torch.bool, device=mem_c.device)
            mem_u, _, _ = self.encode_cond(batch, drop=drop)
        if k > 1:
            mem_c = mem_c.repeat_interleave(k, 0)
            mem_pad = mem_pad.repeat_interleave(k, 0)
            ac = ac.repeat_interleave(k, 0)
            if mem_u is not None:
                mem_u = mem_u.repeat_interleave(k, 0)
        B = mem_c.shape[0]
        x = self._source(batch, B, mem_c.device, k=k)
        for i in range(n_steps):
            t = torch.full((B,), i / n_steps, device=x.device)
            if guidance == 0.0:
                # pure unconditional: the conditional field would be multiplied by zero, so
                # computing it would double the cost for nothing
                v = self.velocity(x, t, mem_u, mem_pad, ac, fut_mask=None)
            else:
                v = self.velocity(x, t, mem_c, mem_pad, ac, fut_mask=None)
                if mem_u is not None:
                    vu = self.velocity(x, t, mem_u, mem_pad, ac, fut_mask=None)
                    v = vu + guidance * (v - vu)
            x = x + v / n_steps
        return x

    @torch.no_grad()
    def attention(self, batch, t=0.5, n_steps=10):
        """Both DiT attentions at flow-time t, per head. Integrates the ODE to t first for a
        representative future state, then returns future self-attention
        [B, layers, heads, H, H] and future-to-memory cross-attention
        [B, layers, heads, H, M], where M is the past tokens plus the three axis tokens."""
        memory, mem_pad, ac = self.encode_cond(batch, drop=None)
        B = memory.shape[0]
        x = self._source(batch, B, memory.device)
        for i in range(max(1, int(t * n_steps))):
            tt = torch.full((B,), i / n_steps, device=x.device)
            x = x + self.velocity(x, tt, memory, mem_pad, ac, fut_mask=None) / n_steps
        _, self_a, cross_a = self.velocity(x, torch.full((B,), t, device=x.device), memory,
                                           mem_pad, ac, fut_mask=None, return_attn=True)
        return torch.stack(self_a, 1), torch.stack(cross_a, 1), mem_pad

    @torch.no_grad()
    def past_self_attention(self, batch):
        ac = self.aircraft(batch["wake"], batch["engine"], batch["n_engines"])
        return self.past_enc.self_attention(batch["past"], batch["past_mask"], ac,
                                            batch.get("past_time"))

    @torch.no_grad()
    def aircraft_modulation(self, batch, t=0.5):
        """How strongly the aircraft reshapes each layer, through the scale/shift path
        rather than attention. Returns the L2 magnitude of the change each modulation
        undergoes when the airframe is known versus forced to "unknown"."""
        ac = self.aircraft(batch["wake"], batch["engine"], batch["n_engines"])
        ac_u = self.aircraft(torch.full_like(batch["wake"], WAKE_VOCAB.index("unknown")),
                             torch.full_like(batch["engine"], ENGINE_VOCAB.index("unknown")),
                             batch["n_engines"])
        tt = torch.full((ac.shape[0],), t, device=ac.device)
        cond, cond_u = self._time_emb(tt) + ac, self._time_emb(tt) + ac_u
        names = ["self:shift", "self:scale", "self:gate", "cross:shift", "cross:scale",
                 "cross:gate", "mlp:shift", "mlp:scale", "mlp:gate"]
        rows = [[float(c.norm()) for c in (blk.adaln(cond) - blk.adaln(cond_u))[0].chunk(9, -1)]
                for blk in self.blocks]
        film = (self.past_enc.film.to_gamma_beta(ac)
                - self.past_enc.film.to_gamma_beta(ac_u))
        g, b = film[0].chunk(2, -1)
        return np.array(rows), names, {"gamma": float(g.norm()), "beta": float(b.norm())}


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, npz, meta, idx):
        self.d = {k: npz[k] for k in npz.files}
        self.idx = idx
        wmap = {w: i for i, w in enumerate(WAKE_VOCAB)}
        emap = {e: i for i, e in enumerate(ENGINE_VOCAB)}
        self.wake = meta["wake"].map(wmap).fillna(len(WAKE_VOCAB) - 1).astype(int).values
        self.eng = meta["engine_class"].map(emap).fillna(len(ENGINE_VOCAB) - 1).astype(int).values

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        i = self.idx[j]
        t = lambda a: torch.from_numpy(a[i])
        item = {
            "past": t(self.d["past"]), "past_mask": t(self.d["past_mask"]),
            "fut_ctrl": t(self.d["fut_ctrl"]), "fut_mask": t(self.d["fut_mask"]),
            "fut_pos": t(self.d["fut_pos"]), "directive": t(self.d["directive"]),
            "wake": torch.tensor(self.wake[i]), "engine": torch.tensor(self.eng[i]),
            "n_engines": torch.tensor(float(self.d["n_engines"][i])),
        }
        if "past_time" in self.d:
            item["past_time"] = t(self.d["past_time"])
        return item


def collate(items):
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


def drop_aircraft(batch):
    """A copy of the batch with the airframe forced to the trained "unknown" token: the
    honest degraded case, for when the aircraft type is not in the database."""
    b = dict(batch)
    b["wake"] = torch.full_like(batch["wake"], WAKE_VOCAB.index("unknown"))
    b["engine"] = torch.full_like(batch["engine"], ENGINE_VOCAB.index("unknown"))
    return b


def decimate_past(batch, keep_prob, generator=None):
    """Cadence-robustness augmentation for the raw-observation past.

    Randomly masks past tokens so the model sees the same leg at many ADS-B cadences.
    Timing is preserved because each surviving token keeps its own past_time. Rows that
    would fall below five surviving tokens are left intact.
    """
    if "past_time" not in batch or keep_prob >= 1.0:
        return batch
    m = batch["past_mask"]
    keep = torch.rand(m.shape, device=m.device, generator=generator) < keep_prob
    newm = m * keep.float()
    too_few = newm.sum(1) < 5
    newm[too_few] = m[too_few]
    out = dict(batch)
    out["past_mask"] = newm
    return out


def save_stage2(model: FlowPredictor, path, **extra):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "source": model.source,
                "src_sigma": model.src_sigma, "step_weight": model.step_weight,
                "pos_loss_w": model.pos_loss_w, "h_steps": model.h_steps,
                "time_encoding": model.time_encoding, **extra}, path)


def load_stage2(path, device="cpu"):
    """Rebuild a Stage-2 model from a checkpoint, inferring shapes from the weights."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    time_enc = "past_enc.pos.weight" not in sd        # raw past has no step embedding
    model = FlowPredictor(
        h_steps=sd["fut_pos.weight"].shape[0],
        p_steps=1 if time_enc else sd["past_enc.pos.weight"].shape[0],
        n_past_f=sd["past_enc.proj.weight"].shape[1], time_encoding=time_enc,
        source=ck.get("source", "gauss"), src_sigma=ck.get("src_sigma", 1.0),
        step_weight=ck.get("step_weight", "integrated"), pos_loss_w=ck.get("pos_loss_w", 0.0))
    model.load_state_dict(sd)
    model.eval().to(device)
    return model
