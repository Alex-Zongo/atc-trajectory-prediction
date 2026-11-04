"""Precomputed radio messages for the semantic model, with word-level confidence.

    python -m atc_forecast.data.messages --data data/v2 --profile jacktol

Every trajectory window gets several messages, each with the 4-slot label Z it should be
read as. Precomputing them makes an SBM epoch a pure tensor pass, and makes every run see
the same noisy transcripts (the draw is a function of the seed and the window id).

    variant     window     status        motion slots     text
    own         maneuver   own_motion    the instruction  its own callsign, the instruction
                                                          through the channel; sometimes a
                                                          "maintain" clause on a steady axis
    maintain    steady     own_motion    one MAINTAIN     "maintain FL240", "fly present
                                                          heading", ...
    own_other   any        own_other     all NONE         own callsign, no motion: "contact
                                                          approach ...", "squawk ..."
    other       any        other         all NONE         a real instruction to another
                                                          aircraft flying at the same time
                                                          (within 2 min), under its own or
                                                          a confusable callsign

The label is what was SAID, not what was heard: if the channel drops the altitude clause,
the vertical slot is still the issued altitude, and the model has to learn how much a
missing clause leaves unknown.

Callsigns go through the channel too (measured dropout and error rates in the profile's
`fields.callsign`). The model also receives the aircraft's own callsign from ADS-B as a
separate exact input, so deciding "is this for me" is a comparison the model can learn.

Word confidence: each word gets a probability drawn from the recognizer's fitted clean or
error distribution (channel/word_conf.py). A word counts as an error when it is a content
word (digits, directions, verbs, callsign words, garbled or filler words) that does not
align with what was said. Phrasing variants are not errors.
"""

import argparse
import difflib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..channel.model import (
    DIGIT_CONFUSIONS,
    FILLERS,
    GARBLE,
    NoiseConfig,
    corrupt_segment,
    load_channel,
)
from ..channel.word_conf import WordConf
from . import atoms as A
from .phraseology import (
    DIGITS,
    PHONETIC,
    TELEPHONY,
    segment_directive,
    spoken_altitude,
    spoken_digits,
    telephony,
)
from .text import AUG_CORPUS, Vocab, _context_pool, tokenize

VARIANTS = ["own", "maintain", "own_other", "other", "swap"]

NON_MOTION = [
    "contact approach one three five point six five", "contact center one three three point niner",
    "contact tower one two zero point niner", "contact departure one two five point six five",
    "squawk four seven two one", "squawk one two zero zero", "radar contact", "ident",
    "verify altitude", "say airspeed", "roger", "good day", "wilco",
    "traffic twelve o'clock five miles opposite direction", "report the airport in sight",
    "altimeter three zero one two", "expect the ils runway two eight left",
    "say again", "monitor tower one one niner point one",
]
MAINTAIN_LATERAL = ["fly present heading", "continue present heading",
                    "maintain present heading"]
MAINTAIN_SPEED = ["maintain present speed", "maintain {spd} knots", "keep the speed {spd}"]
# recorded on NorCal frequencies; used with MessageConfig.real_phrasing
FACILITY_OPENERS = ["norcal departure radar contact", "norcal departure", "norcal approach",
                    "oakland center", "radar contact", "good morning", "good afternoon"]
GREETINGS = ["good day", "have a good day", "thank you", "good afternoon", "bye bye"]
NON_MOTION_REAL = [
    "contact oakland center one one niner point seven five",
    "contact oakland center one two seven point eight good day",
    "contact norcal departure one two seven point zero",
    "contact norcal approach one three five point six five",
    "contact norcal one two zero point niner have a good day",
    "monitor tower one two zero point niner", "frequency change approved",
    "norcal departure radar contact", "radar contact climb via the sid",
    "traffic ten o'clock four miles northbound",
]

# Generated non-motion transmissions (MessageConfig.handoff): handoffs with a random VHF
# frequency, squawk codes, altimeter settings, traffic. Their digits must not read as a
# target value, which ten fixed phrases (NON_MOTION_REAL) do not teach.
HANDOFF_FACILITIES = ["oakland center", "norcal departure", "norcal approach", "norcal",
                      "center", "departure", "approach", "tower", "ground", "bay approach",
                      "socal approach", "oakland tower", "san jose tower"]


def _digits(s: str) -> str:
    return " ".join(DIGITS[c] for c in s)


def frequency(rng) -> str:
    """A VHF frequency as a controller says it, or as a recognizer gives it: 118.000 to
    136.975 in 25 kHz steps, trailing zeros dropped, "point" sometimes said and sometimes
    lost."""
    mhz = int(rng.integers(118, 137))
    khz = f"{int(rng.integers(0, 40)) * 25:03d}".rstrip("0") or "0"
    if rng.random() < 0.3:
        khz = khz[:2]
    sep = " point " if rng.random() < 0.7 else " "
    return _digits(str(mhz)) + sep + _digits(khz)


def handoff_phrase(rng, freq_only=False) -> str:
    fac = str(rng.choice(HANDOFF_FACILITIES))
    f = frequency(rng)
    forms = [f"contact {fac} {f}", f"contact {fac} on {f}", f"{fac} {f}",
             f"monitor {fac} {f}", f"contact {fac} {f} good day",
             f"contact {fac} {f} see ya later"]
    if freq_only or rng.random() < 0.6:
        return str(rng.choice(forms))
    other = [f"squawk {_digits(''.join(str(d) for d in rng.integers(0, 8, 4)))}",
             f"altimeter {_digits('2' + str(rng.integers(98, 100)) + str(rng.integers(0, 10)))}"
             if rng.random() < 0.3 else
             f"altimeter {_digits('30' + str(rng.integers(0, 10)) + str(rng.integers(0, 10)))}",
             f"traffic {_digits(str(rng.integers(1, 13)))} o'clock "
             f"{_digits(str(rng.integers(2, 15)))} miles",
             f"{fac} radar contact", "frequency change approved"]
    return str(rng.choice(other))


_WORD_DIGIT = {v: k for k, v in DIGITS.items()}
CONTENT = (set(DIGITS.values()) | {"left", "right", "climb", "descend", "increase", "reduce"}
           | {w for vs in GARBLE.values() for v in vs for w in v.split()} | set(FILLERS)
           | set(PHONETIC.values()) | {w.lower() for n in TELEPHONY.values() for w in n.split()})


@dataclass(frozen=True)
class MessageConfig:
    profile: str = "jacktol"
    draws_own: int = 4
    draws_aux: int = 1
    p_maintain_clause: float = 0.2
    phrase_variant_p: float = 0.3
    keyword_dropout_p: float = 0.1
    context_scale: float = 1.0
    max_len: int = 48
    cs_len: int = 12
    seed: int = 0
    # "other" messages: an instruction actually given to another aircraft flying within
    # distractor_dt seconds of this window (the traffic sharing the frequency), or, with
    # "any", to any maneuvering aircraft of the same day (the first build)
    distractors: str = "concurrent"
    distractor_dt: float = 120.0
    tag: str = ""                   # output folder messages_{profile}{tag}
    vocab_from: str = ""            # reuse this vocab.json, so trained models can read it
    # Callsign-only negatives: the window's OWN instruction under another aircraft's
    # callsign (variant "swap", status other). The history fits the instruction, so only
    # the callsign tells the message is not for this aircraft. Without them every "other"
    # message also carries another aircraft's instruction, the history alone separates
    # the two, and the model never has to compare callsigns.
    swap: bool = False
    # Real-radio phrasing seen on recorded frequencies and absent from the base grammar:
    # "heavy"/"super" after the callsign, facility openers, "maintain <alt>" for a climb
    # or descent, greetings, handoffs with a frequency. Each touch has this probability.
    real_phrasing: float = 0.0
    # Generated non-motion transmissions (handoff_phrase): this share of own_other bodies,
    # half of it for "other" bodies (another aircraft's handoff), and a quarter of it
    # appended to own instructions with the label unchanged ("climb ..., contact ...")
    handoff: float = 0.0
    vocab_extend: str = ""          # start from this vocab.json, append new words (ids kept)


def _cs_noise(profile):
    f = load_channel(profile).fields.get("callsign", {})
    return float(f.get("dropout", 0.05)), float(f.get("error", 0.05))


def confusable(cs_words: list[str], rng) -> list[str]:
    """A callsign one slip away: one digit swapped for a confusable one, or a different
    airline name. This is what makes "is it for me" hard on a busy frequency."""
    w = list(cs_words)
    digits = [i for i, x in enumerate(w) if x in _WORD_DIGIT]
    if digits and rng.random() < 0.8:
        i = int(rng.choice(digits))
        d = _WORD_DIGIT[w[i]]
        w[i] = DIGITS[str(rng.choice(DIGIT_CONFUSIONS.get(d, ["1"])))]
        return w
    names = [n.lower() for n in TELEPHONY.values()]
    return [str(rng.choice(names))] + w[1:] if w else w


def _heard_callsign(cs: str, p_drop, p_err, rng) -> str:
    if rng.random() < p_drop:
        return ""
    words = cs.lower().split()
    return " ".join(confusable(words, rng)) if rng.random() < p_err else " ".join(words)


def _wrap(cs, body, ctx_pool, scale, rng):
    parts = [cs] if cs else []
    if ctx_pool and rng.random() < min(0.25 * scale, 0.95):
        parts.append(str(rng.choice(ctx_pool)))
    parts.append(body.rstrip("."))
    if ctx_pool and rng.random() < min(0.45 * scale, 0.95):
        parts.append(str(rng.choice(ctx_pool)))
    return " ".join(p for p in parts if p).strip()


def word_errors(said: str, heard: str) -> np.ndarray:
    """Per heard token: True when it is a content word that does not align with what was
    said. Alignment is on tokens, so a re-phrased function word is not an error."""
    a, b = tokenize(said), tokenize(heard)
    err = np.zeros(len(b), bool)
    for op, _, _, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op in ("replace", "insert"):
            for j in range(j1, j2):
                err[j] = b[j] in CONTENT
    return err


def _real(cs, cs_heard, body_said, body_heard, wake, p, rng):
    """Real-radio phrasing applied identically to what was said and what was heard.
    Returns (cs_said, cs_heard, body_said, body_heard)."""
    if p <= 0:
        return cs, cs_heard, body_said, body_heard
    if wake in ("H", "S") and rng.random() < min(1.0, 4 * p):
        suf = " heavy" if wake == "H" else " super"
        cs = cs + suf
        cs_heard = cs_heard + suf if cs_heard else cs_heard
    if rng.random() < 2 * p:
        for verb in ("climb and maintain", "descend and maintain"):
            body_said = body_said.replace(verb, "maintain")
            body_heard = body_heard.replace(verb, "maintain")
    if rng.random() < p:
        o = str(rng.choice(FACILITY_OPENERS))
        body_said, body_heard = f"{o} {body_said}", f"{o} {body_heard}"
    if rng.random() < p:
        g = str(rng.choice(GREETINGS))
        body_said, body_heard = f"{body_said} {g}", f"{body_heard} {g}"
    return cs, cs_heard, body_said, body_heard


def _seg(row) -> dict:
    return {"lateral": row["lateral"], "vertical": row["vertical"], "speed": row["speed"],
            "hdg_exit": row["hdg_exit"], "alt_exit_ft": row["alt_exit_ft"],
            "spd_exit_kt": row["spd_exit_kt"], "callsign": row["callsign"]}


def _maintain_clause(axis, row, rng):
    if axis == "lateral":
        return str(rng.choice(MAINTAIN_LATERAL))
    if axis == "vertical":
        return f"maintain {spoken_altitude(row['alt_ft'])}"
    spd = int(round(row["gs_kt"] / 10.0) * 10)
    return str(rng.choice(MAINTAIN_SPEED)).format(spd=spoken_digits(spd))


def _steady_axes(row):
    """Axes on which a MAINTAIN is truthful for this window: not maneuvering on that axis,
    and for altitude, actually level now."""
    man = row["kind"] == "maneuver"
    ax = []
    if not man or row["lateral"] not in ("L", "R"):
        ax.append("lateral")
    if (not man or row["vertical"] not in ("climb", "descend")) and abs(row["vr_fpm"]) < 300:
        ax.append("vertical")
    if not man or row["speed"] not in ("accel", "decel"):
        ax.append("speed")
    return ax


def generate(meta: pd.DataFrame, pool: pd.DataFrame, cfg: MessageConfig,
             chunk_seed: int) -> pd.DataFrame:
    """Messages for a block of windows; "other" instructions are drawn from `pool`, the
    maneuver windows of the same day and split: by default those of other aircraft within
    cfg.distractor_dt seconds (concurrent traffic, so the instruction is one that could
    be heard on the same frequency and fits a similar flight). Deterministic in
    (cfg.seed, chunk_seed)."""
    rng = np.random.default_rng([cfg.seed, chunk_seed])
    wc = WordConf(cfg.profile)
    p_drop, p_err = _cs_noise(cfg.profile)
    ncfg = NoiseConfig(profile=cfg.profile, phrase_variant_p=cfg.phrase_variant_p,
                       keyword_dropout_p=cfg.keyword_dropout_p)
    rows = meta.to_dict("records")
    man_rows = pool.to_dict("records")
    concurrent = cfg.distractors == "concurrent" and {"t0", "icao24"} <= set(pool.columns)
    if concurrent:
        order = np.argsort(pool["t0"].to_numpy())
        pool_t = pool["t0"].to_numpy()[order]
        pool_ic = pool["icao24"].to_numpy()[order]

    def distractor(r):
        if concurrent:
            lo = np.searchsorted(pool_t, r["t0"] - cfg.distractor_dt)
            hi = np.searchsorted(pool_t, r["t0"] + cfg.distractor_dt)
            c = order[lo:hi][pool_ic[lo:hi] != r["icao24"]]
            if len(c):
                return man_rows[int(rng.choice(c))]
        return man_rows[int(rng.integers(len(man_rows)))] if man_rows else r
    out = []

    def emit(r, variant, draw, said, heard, status, z):
        err = word_errors(said, heard)
        out.append({"window_id": r["window_id"], "variant": variant, "draw": draw,
                    "text": heard, "said": said, "status": status,
                    "atom_lateral": z[0], "atom_vertical": z[1], "atom_speed": z[2],
                    "n_word_errors": int(err.sum()),
                    "kappa": wc.sample(err, rng)})

    def emit_swap(r, own_cs, pool, steady):
        """The window's own instruction (maneuver) or maintain clause (steady) under another
        aircraft's callsign, heard (never dropped: the callsign is the only cue)."""
        for d in range(cfg.draws_aux):
            o = distractor(r)
            other_cs = (" ".join(confusable(own_cs.split(), rng)) if rng.random() < 0.5
                        else telephony(o["callsign"]).lower())
            if other_cs == own_cs:
                other_cs = " ".join(confusable(own_cs.split(), rng))
            cs_h = _heard_callsign(other_cs, 0.0, p_err, rng)
            if cs_h == own_cs or other_cs == own_cs:
                continue
            if r["kind"] == "maneuver":
                n = corrupt_segment(_seg(r), ncfg, rng)
                b_s = segment_directive(_seg(r), spoken=True).rstrip(".")
                b_h = n["noisy_directive"].rstrip(".")
            elif steady:
                b_s = b_h = _maintain_clause(str(rng.choice(steady)), r, rng)
            else:
                continue
            cs_s, cs_h, b_s, b_h = _real(other_cs, cs_h, b_s, b_h, o.get("wake"),
                                         cfg.real_phrasing, rng)
            ctx_seed = int(rng.integers(1 << 30))
            said = _wrap(cs_s, b_s, pool, cfg.context_scale, np.random.default_rng(ctx_seed))
            heard = _wrap(cs_h, b_h, pool, cfg.context_scale, np.random.default_rng(ctx_seed))
            emit(r, "swap", d, said, heard, 2, [A.NONE] * 3)

    for r in rows:
        own_cs = telephony(r["callsign"]).lower()
        pool = _context_pool(r.get("leg_type"), r.get("phase"))
        steady = _steady_axes(r)
        if r["kind"] == "maneuver":
            z = (r["atom_lateral"], r["atom_vertical"], r["atom_speed"])
            seg = _seg(r)
            for d in range(cfg.draws_own):
                n = corrupt_segment(seg, ncfg, rng)
                said_body = segment_directive(seg, spoken=True).rstrip(".")
                heard_body = n["noisy_directive"].rstrip(".")
                zz = list(z)
                if steady and rng.random() < cfg.p_maintain_clause:
                    ax = str(rng.choice(steady))
                    clause = _maintain_clause(ax, r, rng)
                    said_body = f"{said_body}, {clause}" if said_body else clause
                    heard_body = f"{heard_body}, {clause}" if heard_body else clause
                    zz[["lateral", "vertical", "speed"].index(ax)] = A.MAINTAIN
                if cfg.handoff > 0 and rng.random() < cfg.handoff / 4:
                    h = handoff_phrase(rng, freq_only=True)
                    said_body, heard_body = f"{said_body}, {h}", f"{heard_body}, {h}"
                cs_h = _heard_callsign(own_cs, p_drop, p_err, rng)
                cs_s, cs_h, said_body, heard_body = _real(own_cs, cs_h, said_body, heard_body,
                                                          r.get("wake"), cfg.real_phrasing, rng)
                ctx_seed = int(rng.integers(1 << 30))
                said = _wrap(cs_s, said_body, pool, cfg.context_scale,
                             np.random.default_rng(ctx_seed))
                heard = _wrap(cs_h, heard_body, pool, cfg.context_scale,
                              np.random.default_rng(ctx_seed))
                emit(r, "own", d, said, heard, 0, zz)
        elif steady:
            for d in range(cfg.draws_aux):
                ax = str(rng.choice(steady))
                clause = _maintain_clause(ax, r, rng)
                zz = [A.NONE] * 3
                zz[["lateral", "vertical", "speed"].index(ax)] = A.MAINTAIN
                cs_h = _heard_callsign(own_cs, p_drop, p_err, rng)
                cs_s, cs_h, cl_s, cl_h = _real(own_cs, cs_h, clause, clause, r.get("wake"),
                                               cfg.real_phrasing, rng)
                emit(r, "maintain", d, f"{cs_s} {cl_s}", f"{cs_h} {cl_h}".strip(), 0, zz)
        for d in range(cfg.draws_aux):
            body = str(rng.choice(NON_MOTION + NON_MOTION_REAL if cfg.real_phrasing > 0
                                  else NON_MOTION))
            if cfg.handoff > 0 and rng.random() < cfg.handoff:
                body = handoff_phrase(rng)
            cs_h = _heard_callsign(own_cs, p_drop, p_err, rng)
            cs_s, cs_h, b_s, b_h = _real(own_cs, cs_h, body, body, r.get("wake"),
                                         cfg.real_phrasing, rng)
            emit(r, "own_other", d, f"{cs_s} {b_s}", f"{cs_h} {b_h}".strip(), 1,
                 [A.NONE] * 3)
        for d in range(cfg.draws_aux):
            o = distractor(r)
            other_cs = (" ".join(confusable(own_cs.split(), rng)) if rng.random() < 0.5
                        else telephony(o["callsign"]).lower())
            if other_cs == own_cs:
                other_cs = " ".join(confusable(own_cs.split(), rng))
            seg = _seg(o)
            n = corrupt_segment(seg, ncfg, rng)
            cs_h = _heard_callsign(other_cs, p_drop, p_err, rng)
            b_s = segment_directive(seg, spoken=True).rstrip(".")
            b_h = n["noisy_directive"].rstrip(".")
            if cfg.handoff > 0 and rng.random() < cfg.handoff / 2:
                b_s = b_h = handoff_phrase(rng)
            cs_s, cs_h, b_s, b_h = _real(other_cs, cs_h, b_s, b_h, o.get("wake"),
                                         cfg.real_phrasing, rng)
            said = f"{cs_s} {b_s}"
            heard = f"{cs_h} {b_h}".strip()
            emit(r, "other", d, said, heard, 2, [A.NONE] * 3)
        if cfg.swap:
            emit_swap(r, own_cs, pool, steady)
    return pd.DataFrame(out)


def _job(args):
    return generate(*args)


def build(data_dir, cfg: MessageConfig = MessageConfig(), workers=8):
    data = Path(data_dir)
    meta = pd.read_parquet(data / "windows.parquet")
    out = data / f"messages_{cfg.profile}{cfg.tag}"
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    # "other" messages are drawn from the same day and split, so no instruction text
    # crosses a split
    jobs = []
    for bi, (_, g) in enumerate(meta.groupby(["day", "split"], sort=True)):
        pool = g[g["kind"] == "maneuver"]
        for ci, i in enumerate(range(0, len(g), 10000)):
            jobs.append((g.iloc[i:i + 10000], pool, cfg, bi * 1000 + ci))
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_job, jobs))
    msgs = pd.concat(parts, ignore_index=True)
    print(f"messages - {len(msgs)} generated in {time.time() - t:.0f}s", flush=True)

    own = meta[["window_id", "callsign"]].copy()
    own["cs_text"] = [telephony(c).lower() for c in own["callsign"]]
    if cfg.vocab_extend:
        # keep every existing id; new words go at the end, so a model trained on the old
        # vocabulary reads old messages unchanged and only learns the new rows
        old = json.loads(Path(cfg.vocab_extend).read_text())
        full = Vocab(list(msgs["text"].unique()) + list(own["cs_text"].unique()),
                     extra=AUG_CORPUS + NON_MOTION + NON_MOTION_REAL + MAINTAIN_LATERAL
                     + FACILITY_OPENERS + GREETINGS + ["heavy", "super"])
        seen = set(old)
        vocab = Vocab.from_itos(old + [w for w in full.itos[2:] if w not in seen])
    elif cfg.vocab_from:
        vocab = Vocab.from_itos(json.loads(Path(cfg.vocab_from).read_text()))
    else:
        vocab = Vocab(list(msgs["text"].unique()) + list(own["cs_text"].unique()),
                      extra=AUG_CORPUS + NON_MOTION + MAINTAIN_LATERAL)
    toks = np.zeros((len(msgs), cfg.max_len), np.int16)
    kap = np.zeros((len(msgs), cfg.max_len), np.uint8)
    for i, (txt, k) in enumerate(zip(msgs["text"], msgs["kappa"])):
        ids = vocab.encode(txt, cfg.max_len)
        toks[i] = ids
        n = min(len(k), cfg.max_len)
        kap[i, :n] = np.round(np.asarray(k[:n]) * 255)
    cs_tok = np.array([vocab.encode(c, cfg.cs_len) for c in own["cs_text"]], np.int16)
    np.save(out / "tokens.npy", toks)
    np.save(out / "kappa.npy", kap)
    np.save(out / "own_callsign.npy", cs_tok)          # row = window_id
    msgs.drop(columns=["kappa"]).to_parquet(out / "messages.parquet", index=False)
    (out / "vocab.json").write_text(json.dumps(vocab.itos))
    summary = {
        "config": asdict(cfg), "n_messages": len(msgs), "vocab": len(vocab),
        "unknown_token_frac": float((toks == 1).sum() / max((toks > 0).sum(), 1)),
        "word_conf_fitted": WordConf(cfg.profile).fitted,
        "by_variant": msgs["variant"].value_counts().to_dict(),
        "truncated_frac": float((msgs["text"].map(lambda s: len(tokenize(s)))
                                 > cfg.max_len).mean()),
        "word_error_rate": float(msgs["n_word_errors"].sum()
                                 / msgs["text"].map(lambda s: len(tokenize(s))).sum()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/v2")
    ap.add_argument("--profile", default="jacktol")
    ap.add_argument("--draws-own", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--distractors", default="concurrent", choices=("concurrent", "any"))
    ap.add_argument("--distractor-dt", type=float, default=120.0)
    ap.add_argument("--tag", default="", help="output folder suffix: messages_{profile}{tag}")
    ap.add_argument("--vocab-from", default="", help="reuse an existing vocab.json")
    ap.add_argument("--vocab-extend", default="",
                    help="extend an existing vocab.json (ids kept, new words appended)")
    ap.add_argument("--swap", action="store_true",
                    help="add callsign-only negatives (own instruction, other callsign)")
    ap.add_argument("--real-phrasing", type=float, default=0.0,
                    help="probability of each real-radio phrasing touch (0 = base grammar)")
    ap.add_argument("--handoff", type=float, default=0.0,
                    help="share of no-motion messages that are generated handoffs, squawks, "
                         "altimeter settings or traffic (0 = the fixed phrases only)")
    a = ap.parse_args()
    build(a.data, MessageConfig(profile=a.profile, draws_own=a.draws_own, seed=a.seed,
                                distractors=a.distractors, distractor_dt=a.distractor_dt,
                                tag=a.tag, vocab_from=a.vocab_from, vocab_extend=a.vocab_extend,
                                swap=a.swap, real_phrasing=a.real_phrasing,
                                handoff=a.handoff), workers=a.workers)


if __name__ == "__main__":
    main()
