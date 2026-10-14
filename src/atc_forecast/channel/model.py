"""The ASR observation channel: clearance to noisy transcript.

Why a channel at all. The clearances are a deterministic function of the trajectory, so
the dataset has ZERO transcript uncertainty by construction. The uncertainty this work is
about is the uncertainty of ATC communication, so it has to be injected. That is a
strength rather than a workaround: the channel becomes a controlled independent variable.

A channel is a per-FIELD rate model fitted to a real recognizer (see `fit.py`), applied to
a spoken rendering of the clearance. Each field is corrupted by its own independent draw:
dropped, digit-confused, direction-flipped, or spoken and then corrected. Errors are
measured against what the controller ISSUED, meaning the value as it would actually be
spoken, never the raw trajectory value, otherwise phraseology rounding would masquerade as
recognition error.

Two things the channel deliberately does. The noise process is MANY-TO-ONE: distinct true
clearances map to the same heard string, since "climb and maintain one thousand" is
consistent with a correct 1,000, a slipped 2,000, or a flipped descent. That genuine
posterior over true clearances is the communication uncertainty Stage 1 must represent.
And the surface layer plus the confidence value are the OBSERVABLE evidence: pure semantic
noise is undetectable, because a misheard heading is still a valid clearance, so with no
visible cue the model could only ever learn a constant uncertainty.

Known limitation, measured. The independence assumption across fields holds on real data
(the odds of a second field being wrong given a first is wrong come out at or below the
independent baseline). The confidence model does not: a real recognizer's confidence
predicts its own error rate strongly, while a fitted channel emits one value plus noise, so
the confidence input carries no per-utterance information. Supplying an empirical
confidence distribution is the open improvement.
"""

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.phraseology import (round_heading, round_speed, spoken_altitude, spoken_digits,
                                spoken_heading)

PROFILES_DIR = Path(__file__).parent / "profiles"

# Phonetically confusable digits; "five" against "niner" is the classic pair.
DIGIT_CONFUSIONS = {
    "0": ["9"], "1": ["9", "2"], "2": ["3", "1"], "3": ["2", "8"], "4": ["5"],
    "5": ["9", "4"], "6": ["7"], "7": ["6"], "8": ["3"], "9": ["5", "1"],
}

# How a recognizer mangles the words themselves. These variants must also be in the
# tokenizer vocabulary, or every garbled keyword collapses to <unk> and the surface
# realism is lost before the model ever sees it.
GARBLE = {
    "descend": ["descent", "dissend", "the send"],
    "climb": ["climbed", "clime", "climp"],
    "maintain": ["remain", "maintan", "main tain"],
    "heading": ["heding", "head in", "headin"],
    "turn": ["turnn", "term"],
    "left": ["lift", "let"], "right": ["write", "rite"],
    "knots": ["nots", "knotts"], "thousand": ["thousan", "thouzand"],
    "speed": ["spead", "speeds"],
    "increase": ["increas", "increaze"], "reduce": ["reduse", "redoos"],
}
FILLERS = ["uh", "ah", "um", "er"]
FUNCTION_WORDS = {"and", "to", "the"}


@dataclass(frozen=True)
class NoiseConfig:
    """Per-call channel settings.

    `profile` names a JSON profile in profiles/. `mean_degradation` and `concentration`
    draw a latent audio quality d per transmission, which only scales the rates for a
    profile declaring `scaling: by_degradation`; a profile fitted to a real recognizer at
    its real operating quality declares `fixed` and ignores d.

    The two phrasing knobs are phraseology rather than audio quality, so they are NOT
    scaled by d:

      phrase_variant_p  render a clause in a non-standard but real form ("fly heading",
                        "descend to", bare "N knots", heading-word omission).
      keyword_dropout_p drop the scale or slot keyword, leaving bare digits, so "flight
                        level three five zero" becomes "three five zero". The field and the
                        magnitude must then be inferred from the verb, context and current
                        state rather than read off the keyword, which is a real recognizer
                        failure that digit confusion alone never produces.

    Both default to 0 for reproducible evaluation; training sets them above 0.
    """
    profile: str = "jacktol"
    mean_degradation: float = 0.15
    concentration: float = 3.0
    phrase_variant_p: float = 0.0
    keyword_dropout_p: float = 0.0


@dataclass(frozen=True)
class Channel:
    """A named channel loaded from profiles/<name>.json.

    Per-field rates are omission-dominant, as measured. Fields with few real records are
    listed in `provisional`: those are priors, not measurements, and should be reported as
    such.
    """
    name: str
    spoken: bool = True
    scaling: str = "fixed"
    fields: dict = field(default_factory=dict)
    correction_rate: float = 0.0
    verb_flip: float = 0.0
    full_dropout: float = 0.0
    confidence: dict = field(default_factory=lambda: {"kind": "fixed", "value": 0.8,
                                                      "noise": 0.05})
    surface: dict = field(default_factory=dict)
    usable_for_sampling: tuple = ()
    provisional: tuple = ()
    source: str = ""

    def rate(self, field_name: str, kind: str, default: float = 0.0) -> float:
        return float(self.fields.get(field_name, {}).get(kind, default))


@lru_cache(maxsize=None)
def load_channel(name_or_path: str) -> Channel:
    """Load a channel by name (profiles/<name>.json) or explicit path."""
    p = Path(name_or_path)
    if not p.exists():
        p = PROFILES_DIR / f"{name_or_path}.json"
    d = json.loads(p.read_text())
    known = {f: d[f] for f in Channel.__dataclass_fields__ if f in d}
    known["usable_for_sampling"] = tuple(d.get("usable_for_sampling", ()))
    known["provisional"] = tuple(d.get("provisional", ()))
    return Channel(**known)


def list_channels() -> list[str]:
    return sorted(p.stem for p in PROFILES_DIR.glob("*.json")) if PROFILES_DIR.exists() else []


def _sample_degradation(rng, cfg: NoiseConfig) -> float:
    """Latent audio quality per transmission, right-skewed since most calls are clean."""
    m = float(np.clip(cfg.mean_degradation, 1e-4, 1 - 1e-4))
    c = max(cfg.concentration, 0.1)
    return float(rng.beta(m * c, (1 - m) * c))


def _corrupt_digits(value: int, rng, rate: float) -> int:
    s = str(int(value))
    out = [rng.choice(DIGIT_CONFUSIONS[ch])
           if ch in DIGIT_CONFUSIONS and rng.random() < rate else ch for ch in s]
    return int("".join(out))


def _surface_corrupt(text: str, d: float, rates: dict, rng):
    """The visible degradation: the evidence the encoder gets to see."""
    if not text:
        return text, []
    tags, words = [], text.rstrip(".").split()
    out = []
    for w in words:
        key = w.strip(",.").lower()
        if key in FUNCTION_WORDS and rng.random() < rates.get("word_drop", 0.0) * d:
            tags.append("word_drop")
            continue
        if key in GARBLE and rng.random() < rates.get("garble", 0.0) * d:
            out.append(str(rng.choice(GARBLE[key])))
            tags.append("garble")
            continue
        out.append(w)
    if out and rng.random() < rates.get("filler", 0.0) * d:
        out.insert(int(rng.integers(0, len(out) + 1)), str(rng.choice(FILLERS)))
        tags.append("filler")
    # stepped-on transmission: cut off mid-clearance, often losing the number
    if len(out) > 3 and rng.random() < rates.get("truncation", 0.0) * d:
        tags.append("truncation")
        return " ".join(out[:int(rng.integers(2, len(out)))]), tags
    return " ".join(out) + ".", tags


def _confidence(ch: Channel, d: float, rng) -> float:
    c = ch.confidence
    kind = c.get("kind", "fixed")
    if kind == "fixed":
        base = float(c.get("value", 0.8))
    elif kind == "empirical_bins" and c.get("bins"):
        bins = c["bins"]
        w = np.asarray([max(b.get("count", 1), 0) for b in bins], float)
        b = bins[int(rng.choice(len(bins), p=w / w.sum()))] if w.sum() > 0 else {}
        base = float(b.get("mean_confidence", 0.8))
    else:
        base = 1.0 - d
    return float(np.clip(base + rng.normal(0, c.get("noise", 0.0)), 0.0, 1.0))


def _phrase_heading(side, h, rng, p, kw_p=0.0):
    if rng.random() < kw_p:
        return str(rng.choice([f"turn {side} {h}", f"{side} {h}", h]))
    if rng.random() >= p:
        return f"turn {side} heading {h}"
    return str(rng.choice([f"turn {side} {h}", f"{side} turn heading {h}", f"{side} turn {h}",
                           f"fly heading {h}", f"heading {h}"]))


def _phrase_altitude(verb, a, rng, p, bare=None, kw_p=0.0):
    if bare is not None and rng.random() < kw_p:
        return str(rng.choice([f"{verb} {bare}", f"{verb} to {bare}", bare]))
    if rng.random() >= p:
        return f"{verb} and maintain {a}"
    return str(rng.choice([f"{verb} to {a}", f"{verb} {a}", f"{verb} and maintain {a}"]))


def _phrase_speed(verb, s, rng, p, kw_p=0.0):
    if rng.random() < kw_p:
        return str(rng.choice([f"{verb} {s}", f"{s}"]))
    if rng.random() >= p:
        return f"{verb} speed {s} knots"
    return str(rng.choice([f"{verb} speed to {s} knots", f"{s} knots", f"{verb} {s} knots"]))


def _dropped(issued, d, conf):
    h, a, s = issued
    return {"noisy_directive": "", "was_dropped": True, "corrupted": "full_dropout",
            "degradation": d, "asr_confidence": conf, "has_correction": False,
            "issued_hdg": h, "issued_alt_ft": a, "issued_spd_kt": s,
            "heard_hdg": np.nan, "heard_alt_ft": np.nan, "heard_spd_kt": np.nan,
            "hdg_err_deg": np.nan, "alt_err_ft": np.nan, "spd_err_kt": np.nan}


def corrupt_segment(seg, cfg: NoiseConfig, rng) -> dict:
    """Pass one clearance through the channel.

    `seg` needs lateral, vertical, speed, hdg_exit, alt_exit_ft, spd_exit_kt. Returns the
    noisy transcript, the recognizer's confidence, the heard values, and the per-field
    error against the issued values.
    """
    ch = load_channel(cfg.profile)
    d = _sample_degradation(rng, cfg)
    scale = d if ch.scaling == "by_degradation" else 1.0
    lateral, vertical, speed = seg["lateral"], seg["vertical"], seg["speed"]

    issued_hdg = float(round_heading(float(seg["hdg_exit"])))
    issued_alt = float(max(round(float(seg["alt_exit_ft"]) / 1000.0), 1) * 1000)
    issued_spd = float(round_speed(float(seg["spd_exit_kt"])))
    heard_hdg, heard_alt, heard_spd = issued_hdg, issued_alt, issued_spd
    conf = _confidence(ch, d, rng)
    tags, parts, has_correction = [], [], False

    if rng.random() < ch.full_dropout * scale:
        return _dropped((issued_hdg, issued_alt, issued_spd), d, conf)

    hdg_str = spoken_heading if ch.spoken else (lambda h: f"{round_heading(h):03d}")
    alt_str = spoken_altitude if ch.spoken else (lambda a: f"{int(a):,}")
    spd_str = (lambda s: spoken_digits(int(s))) if ch.spoken else (lambda s: str(int(s)))

    if lateral in ("L", "R"):
        dir_ = lateral
        if rng.random() < ch.rate("direction", "flip") * scale:
            dir_ = "R" if dir_ == "L" else "L"
            tags.append("direction_flip")
        side = "left" if dir_ == "L" else "right"
        if rng.random() < max(ch.rate("direction", "dropout"),
                              ch.rate("heading", "dropout")) * scale:
            tags.append("lateral_dropout")
            heard_hdg = np.nan
        elif rng.random() < ch.correction_rate * scale:
            # misspoke then corrected; the FINAL value is the truth
            wrong = _corrupt_digits(int(issued_hdg), rng, 0.7)
            parts.append(f"turn {side} heading {hdg_str(wrong)} correction {hdg_str(issued_hdg)}")
            tags.append("correction")
            has_correction = True
        elif rng.random() < ch.rate("heading", "number_dropout") * scale:
            parts.append(f"turn {side} heading")
            tags.append("number_dropout")
            heard_hdg = np.nan
        else:
            h = _corrupt_digits(int(issued_hdg), rng,
                                ch.rate("heading", "digit_error") * scale) % 360
            heard_hdg = float(h) if h != 0 else 360.0
            if abs(heard_hdg - issued_hdg) > 0.5:
                tags.append("heading_digit")
            parts.append(_phrase_heading(side, hdg_str(heard_hdg), rng, cfg.phrase_variant_p,
                                         kw_p=cfg.keyword_dropout_p))

    if vertical in ("climb", "descend"):
        v = vertical
        if rng.random() < ch.verb_flip * scale:
            v = "descend" if v == "climb" else "climb"
            tags.append("verb_flip")
        if rng.random() < ch.rate("altitude", "dropout") * scale:
            tags.append("altitude_dropout")
            heard_alt = np.nan
        elif rng.random() < ch.correction_rate * scale:
            wrong = _corrupt_digits(int(issued_alt / 1000), rng, 0.7) * 1000
            parts.append(f"{v} and maintain {alt_str(wrong)} correction {alt_str(issued_alt)}")
            tags.append("correction")
            has_correction = True
        elif rng.random() < ch.rate("altitude", "number_dropout") * scale:
            parts.append(f"{v} and maintain")
            tags.append("number_dropout")
            heard_alt = np.nan
        else:
            heard_alt = float(_corrupt_digits(int(issued_alt / 1000), rng,
                                              ch.rate("altitude", "digit_error") * scale) * 1000)
            if abs(heard_alt - issued_alt) > 1:
                tags.append("altitude_digit")
            # bare digits with the scale keyword removed: flight-level digits above FL180,
            # else the thousands digit
            a = int(round(heard_alt))
            bare = spd_str(a // 100 if a >= 18000 else a // 1000)
            parts.append(_phrase_altitude(v, alt_str(heard_alt), rng, cfg.phrase_variant_p,
                                          bare=bare, kw_p=cfg.keyword_dropout_p))

    if speed in ("accel", "decel"):
        verb = "increase" if speed == "accel" else "reduce"
        if rng.random() < ch.rate("speed", "dropout") * scale:
            tags.append("speed_dropout")
            heard_spd = np.nan
        elif rng.random() < ch.rate("speed", "number_dropout") * scale:
            parts.append(f"{verb} speed")
            tags.append("number_dropout")
            heard_spd = np.nan
        else:
            heard_spd = float(_corrupt_digits(int(issued_spd), rng,
                                              ch.rate("speed", "digit_error") * scale))
            if abs(heard_spd - issued_spd) > 0.5:
                tags.append("speed_digit")
            parts.append(_phrase_speed(verb, spd_str(heard_spd), rng, cfg.phrase_variant_p,
                                       kw_p=cfg.keyword_dropout_p))

    text = ", ".join(parts) + "." if parts else ""
    if any(ch.surface.get(k, 0.0) > 0 for k in ("garble", "filler", "word_drop", "truncation")):
        text, stags = _surface_corrupt(text, scale, ch.surface, rng)
        tags += stags

    has_lat = lateral in ("L", "R")
    has_vert = vertical in ("climb", "descend")
    has_spd = speed in ("accel", "decel")
    hdg_err = (abs((heard_hdg - issued_hdg + 180) % 360 - 180)
               if has_lat and not np.isnan(heard_hdg) else np.nan)
    return {
        "noisy_directive": text, "was_dropped": not bool(text),
        "corrupted": "|".join(tags), "degradation": d, "asr_confidence": conf,
        "has_correction": has_correction,
        "issued_hdg": issued_hdg, "issued_alt_ft": issued_alt, "issued_spd_kt": issued_spd,
        "heard_hdg": heard_hdg if has_lat else np.nan,
        "heard_alt_ft": heard_alt if has_vert else np.nan,
        "heard_spd_kt": heard_spd if has_spd else np.nan,
        "hdg_err_deg": hdg_err,
        "alt_err_ft": (abs(heard_alt - issued_alt)
                       if has_vert and not np.isnan(heard_alt) else np.nan),
        "spd_err_kt": (abs(heard_spd - issued_spd)
                       if has_spd and not np.isnan(heard_spd) else np.nan),
    }


def add_noise(ds: pd.DataFrame, cfg: NoiseConfig = NoiseConfig(), seed: int = 0) -> pd.DataFrame:
    """Add the heard transcript, its confidence and the semantic error to a segment frame."""
    rng = np.random.default_rng(seed)
    man = ds[ds["is_maneuver"]] if "is_maneuver" in ds.columns else ds
    rows = [corrupt_segment(s, cfg, rng) for _, s in man.iterrows()]
    return pd.concat([man.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
