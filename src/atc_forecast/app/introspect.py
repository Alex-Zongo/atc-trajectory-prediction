"""Step-by-step traces of both models, for the architecture explorer.

Each trace replays a model's forward pass one operation at a time, using the model's own
modules in the model's own order, and records what every stage produced: token vectors,
per-head attention, slot reads, each autoregressive field's distribution and draw, the
flow's state at every ODE step, every DiT block's adaLN gates and attention.

A replay that drifted from the real forward would make the explorer a picture of a
different model, so every trace CHECKS ITSELF before returning: the encoder states against
the module's own forward, the sampled clearances against `ClearancePosterior.sample` under
the same seed, the integrated trajectories against `FlowPredictor.sample`. A mismatch
raises rather than rendering something plausible and wrong.
"""

import base64
import math

import numpy as np
import torch
import torch.nn.functional as F

from ..data.text import UNK, tokenize
from ..models.stage1 import (BIN_UNIT, GATE_FIELD, LOGS_CLAMP, mixture_sample)
from ..models.stage2 import deadreckon, modulate
from ..units import (ALT_SCALE_FT, ENGINE_VOCAB, GS_SCALE_KT, HEAD_BIN_DEG, M_TO_NM,
                     POS_SCALE_M, SPD_SCALE_KT, VR_SCALE_FPM, WAKE_VOCAB)

FIELD_INFO = {
    "is_clearance": ("Is a clearance", "transmission carries an instruction"),
    "pres_lat": ("Lateral present", "a turn clause is present"),
    "dir_lat": ("Turn direction", "right (1) or left (0)"),
    "hdg": ("Heading", "assigned heading, 72 bins of 5 deg"),
    "pres_vert": ("Vertical present", "an altitude clause is present"),
    "dir_vert": ("Vertical direction", "climb (1) or descend (0)"),
    "alt": ("Altitude", "assigned altitude, ft"),
    "pres_spd": ("Speed present", "a speed clause is present"),
    "dir_spd": ("Speed direction", "increase (1) or reduce (0)"),
    "spd": ("Speed", "assigned speed, kt"),
}
UNIT_SCALE = {"alt": ALT_SCALE_FT, "spd": SPD_SCALE_KT}


class TraceMismatch(RuntimeError):
    """A replay disagreed with the model it replays."""


def _check(name, a, b, atol=1e-4):
    if not torch.allclose(a.float().cpu(), b.float().cpu(), atol=atol, rtol=1e-4):
        err = float((a.float().cpu() - b.float().cpu()).abs().max())
        raise TraceMismatch(f"{name}: replay differs from the model by up to {err:.3g}")


def q8(x, lo=None, hi=None) -> dict:
    """A float array as base64 uint8 over [lo, hi], for compact transport to the browser.
    Attention is shipped on [0, 1]; activations on their own symmetric range."""
    a = np.asarray(x, dtype=np.float32)
    if lo is None:
        m = float(np.abs(a).max()) or 1.0
        lo, hi = -m, m
    u = np.clip(np.round((a - lo) / (hi - lo) * 255.0), 0, 255).astype(np.uint8)
    return {"shape": list(a.shape), "lo": lo, "hi": hi,
            "b64": base64.b64encode(u.tobytes()).decode("ascii")}


def _r(x, nd=4):
    return np.round(np.asarray(x, dtype=np.float64), nd).tolist()


def _replay_encoder_layer(layer, h, pad):
    """One pre-LN `nn.TransformerEncoderLayer`, with per-head attention weights."""
    x = layer.norm1(h)
    a, w = layer.self_attn(x, x, x, key_padding_mask=pad, need_weights=True,
                           average_attn_weights=False)
    h_mid = h + a
    f = layer.linear2(layer.activation(layer.linear1(layer.norm2(h_mid))))
    return h_mid + f, w, a, f


# --------------------------------------------------------------------------------------
# Stage 1
# --------------------------------------------------------------------------------------

def _field_dist(model, name, typ, raw, T):
    """The distribution a head defines, as the sampler sees it at temperature T."""
    if typ == "bern":
        return {"p": _r(torch.sigmoid(raw.squeeze(-1) / T).cpu())}
    if typ == "cat":
        return {"probs": _r(F.softmax(raw / T, -1).cpu(), 5)}
    pi, mu, logs = model._mix_params(raw)
    logs_t = (logs + math.log(T)).clamp(*LOGS_CLAMP)
    s = UNIT_SCALE[name]
    return {"pi": _r(F.softmax(pi, -1).cpu(), 5), "mu": _r((mu * s).cpu(), 1),
            "sigma": _r((torch.exp(logs_t) * s).cpu(), 2)}


def _decode_trace(model, h, state_k, sample: bool, T: float):
    """`ClearancePosterior._decode`, recording each step. Same ops, same RNG order."""
    steps, values = [], {}
    gru_states = []
    for i, (name, typ, gate) in enumerate(model._fields):
        gru_states.append(h[0].detach().cpu().numpy())
        h_in = model._apply_film(name, h, state_k)
        raw = model.head[name](h_in)
        if typ == "bern":
            p = torch.sigmoid(raw.squeeze(-1) / T)
            v = torch.bernoulli(p) if sample else (torch.sigmoid(raw.squeeze(-1)) > 0.5).float()
        elif typ == "cat":
            v = (torch.multinomial(F.softmax(raw / T, -1), 1).squeeze(-1).float() if sample
                 else raw.argmax(-1).float())
        else:
            pi, mu, logs = model._mix_params(raw)
            logs_t = (logs + math.log(T)).clamp(*LOGS_CLAMP)
            v = (mixture_sample(pi, mu, logs_t) if sample
                 else (F.softmax(pi, -1) * mu).sum(-1))
        rec = {"name": name, "type": typ, "gate": gate,
               "dist": _field_dist(model, name, typ, raw, T),
               "h_norm": _r(h.norm(dim=-1).cpu(), 3)}
        if name in model.ground_fields and state_k is not None:
            # the counterfactual head on the same recurrent state with grounding removed,
            # which is exactly what the state FiLM changed
            rec["ungrounded"] = _field_dist(model, name, typ, model.head[name](h), T)
            film = model.state_films[name].net(state_k)
            gm, bt = film.chunk(2, -1)
            rec["film"] = {"gamma": _r(gm.norm(dim=-1).cpu(), 3),
                           "beta": _r(bt.norm(dim=-1).cpu(), 3)}
        if typ == "mix":
            rec["value"] = _r((v * UNIT_SCALE[name]).cpu(), 1)
        elif typ == "cat":
            rec["value"] = _r((v * HEAD_BIN_DEG if name == "hdg"
                               else v * BIN_UNIT[name]).cpu(), 1)
        else:
            rec["value"] = _r(v.cpu(), 0)
        if gate is not None:
            rec["active"] = _r(values[GATE_FIELD[gate]].cpu(), 0)
        values[name] = v
        steps.append(rec)
        h = model.gru(model._embed_value(name, v) + model.field_emb[i], h)
    return steps, values, np.stack(gru_states)


@torch.no_grad()
def trace_stage1(model, vocab, max_len, text: str, confidence: float, state=None,
                 k: int = 12, seed: int = 0) -> dict:
    """Everything Stage 1 computes for one transcript."""
    dev = next(model.parameters()).device
    words = tokenize(text)[:max_len]
    ids = vocab.encode(text, max_len)
    L = len(words)
    tokens = torch.tensor([ids], device=dev)
    conf = torch.tensor([float(confidence)], device=dev)
    bb = model.backbone

    conf_e = bb.conf_proj(conf.unsqueeze(-1)).unsqueeze(1)
    tok_e = bb.tok(tokens)
    pos_e = bb.pos(torch.arange(max_len + 1, device=dev))[None]
    h = torch.cat([conf_e, tok_e], 1) + pos_e
    pad = torch.cat([tokens.new_zeros(1, 1, dtype=torch.bool), tokens.eq(0)], 1)
    n = L + 1                                              # confidence slot + real tokens
    embed = h[0, :n]
    layers = []
    for layer in bb.tf.layers:
        h, w, a, f = _replay_encoder_layer(layer, h, pad)
        layers.append({"attn": q8(w[0, :, :n, :n].cpu().numpy(), 0.0, 1.0),
                       "attn_norm": _r(a[0, :n].norm(dim=-1).cpu(), 3),
                       "ffn_norm": _r(f[0, :n].norm(dim=-1).cpu(), 3),
                       "out": q8(h[0, :n].cpu().numpy())})
    ref, _ = bb(tokens, conf)
    _check("text encoder", h[0, :n], ref[0, :n])

    q = model.slots.queries[None]
    so, sw = model.slots.attn(q, h, h, key_padding_mask=pad, need_weights=True,
                              average_attn_weights=False)
    slots = model.slots.norm(so)
    ctx = model.ctx_proj(slots.flatten(1))
    _check("context", ctx, model.encode(tokens, conf))
    h0 = model.h0(ctx)

    st = model._state(state)
    T = float(model.temperature)
    greedy_steps, _, gru = _decode_trace(model, h0, st, sample=False, T=1.0)
    torch.manual_seed(seed)
    hk = h0.repeat_interleave(k, 0)
    stk = st.repeat_interleave(k, 0) if st is not None else None
    sample_steps, vals, _ = _decode_trace(model, hk, stk, sample=True, T=T)
    torch.manual_seed(seed)
    ref = model._decode(tokens, conf, k, True, T, st)
    for name, v in vals.items():
        _check(f"sampled {name}", v.view(1, k), ref[name])

    return {
        "text": text, "confidence": float(confidence), "temperature": T,
        "tokens": ["[conf]"] + words,
        "ids": [-1] + ids[:L], "unk": [False] + [i == UNK for i in ids[:L]],
        "max_len": max_len, "truncated": len(tokenize(text)) > max_len,
        "dims": {"hidden": bb.hidden, "heads": bb.tf.layers[0].self_attn.num_heads,
                 "layers": len(bb.tf.layers), "ffn": bb.tf.layers[0].linear1.out_features,
                 "vocab": len(vocab), "slots": model.n_slots, "k_mix": model.k_mix,
                 "state_dim": model.state_dim},
        "state": _r(state[0].cpu(), 4) if state is not None else None,
        "ground_fields": list(model.ground_fields),
        "embed": {"tok": q8(tok_e[0, :L].cpu().numpy()), "sum": q8(embed.cpu().numpy()),
                  "conf": q8(conf_e[0, 0].cpu().numpy())},
        "layers": layers,
        "slots": {"attn": q8(sw[0, :, :, :n].cpu().numpy(), 0.0, 1.0),
                  "out": q8(slots[0].cpu().numpy())},
        "ctx": q8(ctx[0].cpu().numpy()), "h0": q8(h0[0].cpu().numpy()),
        "gru": q8(gru), "k": k,
        "fields": [{"name": n_, "type": t, "gate": g, "label": FIELD_INFO[n_][0],
                    "desc": FIELD_INFO[n_][1]} for n_, t, g in model._fields],
        "greedy": greedy_steps, "samples": sample_steps,
    }


# --------------------------------------------------------------------------------------
# Stage 2
# --------------------------------------------------------------------------------------

def _dit_trace(blk, h, cond, memory, mem_pad):
    """`DiTBlock.forward`, recording the adaLN modulation, per-head attention, and the
    size of each residual update."""
    s1, c1, g1, s2, c2, g2, s3, c3, g3 = blk.adaln(cond).chunk(9, -1)
    sa_in = modulate(blk.n1(h), s1, c1)
    sa_out, sw = blk.attn(sa_in, sa_in, sa_in, need_weights=True, average_attn_weights=False)
    d1 = g1.unsqueeze(1) * sa_out
    h = h + d1
    q = modulate(blk.n2(h), s2, c2)
    cr_out, cw = blk.cross(q, memory, memory, key_padding_mask=mem_pad, need_weights=True,
                           average_attn_weights=False)
    d2 = g2.unsqueeze(1) * cr_out
    h = h + d2
    d3 = g3.unsqueeze(1) * blk.mlp(modulate(blk.n3(h), s3, c3))
    h = h + d3
    mag = lambda v: float(v[0].abs().mean())
    rec = {"adaln": [mag(v) for v in (s1, c1, g1, s2, c2, g2, s3, c3, g3)],
           "update": [float(d[0].norm(dim=-1).mean()) for d in (d1, d2, d3)],
           "h_norm": float(h[0].norm(dim=-1).mean())}
    return h, sw, cw, rec


def _ctrl_phys(x):
    x = x.cpu().numpy()
    return np.stack([x[..., 0] * GS_SCALE_KT,
                     np.degrees(np.arctan2(x[..., 1], x[..., 2])) % 360,
                     x[..., 3] * VR_SCALE_FPM], -1)


def _pos_phys(p):
    p = p.cpu().numpy()
    return np.stack([p[..., 0] * POS_SCALE_M * M_TO_NM, p[..., 1] * POS_SCALE_M * M_TO_NM,
                     p[..., 2] * ALT_SCALE_FT], -1)


@torch.no_grad()
def trace_stage2(flow, batch, n_steps: int = 10, k: int = 8, guidance: float = 1.0,
                 seed: int = 0) -> dict:
    """Everything Stage 2 computes for one window (batch of one) under its directive."""
    dev = batch["past"].device
    pe = flow.past_enc

    wake, eng, n_eng = batch["wake"], batch["engine"], batch["n_engines"]
    ac = flow.aircraft(wake, eng, n_eng)
    past, pmask, ptime = batch["past"], batch["past_mask"], batch.get("past_time")
    proj = pe.proj(past)
    tpos = pe._pos_term(past, ptime)
    gamma, beta = pe.film.to_gamma_beta(ac).chunk(2, -1)
    h = pe.film(proj + tpos, ac)
    pad = ~pmask.bool()
    past_layers = []
    for layer in pe.tf.layers:
        h, w, a, f = _replay_encoder_layer(layer, h, pad)
        past_layers.append({"attn": q8(w[0].cpu().numpy(), 0.0, 1.0),
                            "attn_norm": _r(a[0].norm(dim=-1).cpu(), 3),
                            "ffn_norm": _r(f[0].norm(dim=-1).cpu(), 3)})
    valid = pmask[0].bool()
    _check("past encoder", h[0, valid], pe(past, pmask, ac, ptime)[0, valid])

    dtok = flow.dir_tok(batch["directive"])
    memory, mem_pad, _ = flow.encode_cond(batch)
    _check("memory", memory[0, ~mem_pad[0]], torch.cat([h, dtok], 1)[0, ~mem_pad[0]])
    mem_u = None
    if guidance != 1.0:
        mem_u, _, _ = flow.encode_cond(batch, drop=torch.ones(1, dtype=torch.bool, device=dev))

    rep = lambda x: x.repeat_interleave(k, 0) if x is not None else None
    mem_c, mem_pk, ac_k, mem_uk = rep(memory), rep(mem_pad), rep(ac), rep(mem_u)
    torch.manual_seed(seed)
    x = flow._source(batch, k, dev, k=k)
    xs, vs, blocks, self_a, cross_a = [x.clone()], [], [], [], []
    for i in range(n_steps):
        t = torch.full((k,), i / n_steps, device=dev)
        cond = flow._time_emb(t) + ac_k
        traced_mem = mem_uk if guidance == 0.0 else mem_c
        hh = flow.fut_proj(x) + flow.fut_pos(torch.arange(flow.h_steps, device=dev))[None]
        recs, sws, cws = [], [], []
        for blk in flow.blocks:
            hh, sw, cw, rec = _dit_trace(blk, hh, cond, traced_mem, mem_pk)
            recs.append(rec)
            sws.append(sw[0].cpu().numpy())
            cws.append(cw[0].cpu().numpy())
        shift, scale = flow.final_ada(cond).chunk(2, -1)
        v = flow.head(modulate(flow.final_n(hh), shift, scale))
        v_traced = v
        if guidance not in (0.0, 1.0):
            vu = flow.velocity(x, t, mem_uk, mem_pk, ac_k, fut_mask=None)
            v = vu + guidance * (v - vu)
        x = x + v / n_steps
        xs.append(x.clone())
        vs.append(v_traced[0].norm(dim=-1).cpu().numpy())
        blocks.append(recs)
        self_a.append(np.stack(sws))
        cross_a.append(np.stack(cws))
    torch.manual_seed(seed)
    _check("flow sample", x, flow.sample(batch, n_steps=n_steps, k=k, guidance=guidance),
           atol=1e-3)

    onset_alt = batch["past"][:, -1, 2] * ALT_SCALE_FT
    xs_t = torch.stack(xs)                                  # [S+1, k, H, 4]
    pos = deadreckon(xs_t.flatten(0, 1), onset_alt.repeat(xs_t.shape[0] * k))
    pos = pos.view(xs_t.shape[0], k, flow.h_steps, 3)
    fmask = batch["fut_mask"][0].cpu().numpy() > 0.5
    pt = ptime[0].cpu().numpy() if ptime is not None else np.arange(past.shape[1])[::-1]
    pv = valid.cpu().numpy()
    mem_labels = ([f"-{s:.0f}s" if v_ else "pad" for s, v_ in zip(pt, pv)]
                  + ["LATERAL", "VERTICAL", "SPEED"])
    d = batch["directive"][0].cpu().numpy()

    return {
        "dims": {"dim": flow.dim, "heads": flow.blocks[0].attn.num_heads,
                 "blocks": len(flow.blocks), "past_layers": len(pe.tf.layers),
                 "H": flow.h_steps, "P": past.shape[1], "mem": memory.shape[1],
                 "source": flow.source, "time_encoding": flow.time_encoding},
        "n_steps": n_steps, "k": k, "guidance": guidance,
        "aircraft": {"wake": WAKE_VOCAB[int(wake[0])], "engine": ENGINE_VOCAB[int(eng[0])],
                     "n_engines": float(n_eng[0]), "emb": q8(ac[0].cpu().numpy()),
                     "film_gamma": float(gamma.norm()), "film_beta": float(beta.norm())},
        "past": {"t_s": _r(-pt[pv], 2),
                 "pos": _r(_pos_phys(past[0, valid, :3]), 3),
                 "ctrl": _r(_ctrl_phys(past[0, valid, 3:7]), 2),
                 "valid": pv.tolist(), "proj_norm": _r(proj[0].norm(dim=-1).cpu(), 3),
                 "time_norm": _r(tpos[0].norm(dim=-1).cpu() if tpos.dim() == 3
                                 else tpos[0].norm(dim=-1).cpu(), 3),
                 "layers": past_layers, "out": q8(h[0].cpu().numpy())},
        "directive": {"vec": _r(d, 4),
                      "lateral": {"present": bool(d[0] > 0.5), "right": bool(d[1] > 0.5),
                                  "hdg": float(np.degrees(np.arctan2(d[2], d[3])) % 360)},
                      "vertical": {"present": bool(d[4] > 0.5), "climb": bool(d[5] > 0.5),
                                   "alt_ft": float(d[6] * ALT_SCALE_FT)},
                      "speed": {"present": bool(d[7] > 0.5), "accel": bool(d[8] > 0.5),
                                "kt": float(d[9] * GS_SCALE_KT)},
                      "tokens": q8(dtok[0].cpu().numpy()),
                      "token_norm": _r(dtok[0].norm(dim=-1).cpu(), 3)},
        "memory": {"labels": mem_labels, "pad": mem_pad[0].cpu().numpy().tolist()},
        "flow": {"ctrl": _r(_ctrl_phys(xs_t), 2), "pos": _r(_pos_phys(pos), 3),
                 "v_norm": _r(np.stack(vs), 4), "blocks": blocks,
                 "self_attn": q8(np.stack(self_a), 0.0, 1.0),
                 "cross_attn": q8(np.stack(cross_a), 0.0, 1.0)},
        "truth": {"mask": fmask.tolist(),
                  "pos": _r(_pos_phys(batch["fut_pos"][0]), 3),
                  "ctrl": _r(_ctrl_phys(batch["fut_ctrl"][0]), 2)},
    }
