"""What Stage 1 is given: the transcript u and the confidence kappa, under three modes.

  clean     the issued clearance exactly as spoken, no recognition error. The reference
            point: whatever spread Stage 1 still reports here is ambiguity in the phrasing
            itself, not in the recognizer.
  channel   the issued clearance passed through a fitted ASR observation channel, the
            setting every reported result uses.
  custom    any text typed by the user, for probing the model directly. Numerals can be
            spoken out first ("270" -> "two seven zero"), since training transcripts
            always spell numbers the way a controller says them.

Both non-custom modes can be wrapped in radio context (a callsign and phase-coherent
phraseology the model must read past), exactly as training wrapped them.
"""

import re
from dataclasses import dataclass

import numpy as np

from ..channel.model import NoiseConfig, corrupt_segment
from ..data.phraseology import (round_heading, round_speed, segment_directive, spoken_altitude,
                                spoken_digits)
from ..data.text import UNK, augment_transcript, tokenize

MODES = {"clean": "Clean (issued clearance)", "channel": "ASR channel",
         "custom": "Custom text"}
# The shipped channel emits 0.82 +- 0.05, so that is the confidence Stage 1 was trained
# around. A clean transcript gets the same value by default, which keeps the comparison
# between modes about the TEXT; the slider exists to probe the confidence input itself.
TRAINED_CONFIDENCE = 0.82


@dataclass(frozen=True)
class Observation:
    mode: str = "channel"
    profile: str = "jacktol"
    phrase_variant_p: float = 0.35
    keyword_dropout_p: float = 0.25
    context: bool = True
    confidence: float = TRAINED_CONFIDENCE
    custom_text: str = ""
    speak_numbers: bool = True
    seed: int = 0


def speak_numbers(text: str) -> str:
    """Numerals rewritten in the spoken form training used: "FL350" as "flight level three
    five zero", whole thousands as "five thousand", anything else digit by digit with
    "niner". Commas inside numbers are allowed ("5,000")."""
    def fl(m):
        return "flight level " + spoken_digits(m.group(1))

    def num(m):
        n = int(m.group(0).replace(",", ""))
        if n >= 1000 and n % 1000 == 0 and n < 18000:
            return spoken_altitude(n)
        return spoken_digits(m.group(0).replace(",", ""))

    text = re.sub(r"\bfl\s*(\d{2,3})\b", fl, text, flags=re.IGNORECASE)
    text = re.sub(r"(\d)\.(\d)", r"\1 point \2", text)          # frequencies
    return re.sub(r"\d{1,3}(?:,\d{3})+|\d+", num, text)


def unknown_words(text: str, vocab) -> list[str]:
    """Tokens Stage 1 will see only as <unk>."""
    return [w for w in tokenize(text) if vocab.stoi.get(w, UNK) == UNK]


def issued_clauses(seg) -> list[str]:
    """The issued clearance as readable clauses, quantised as a controller would say it."""
    out = []
    if seg["lateral"] in ("L", "R"):
        out.append(f"turn {'left' if seg['lateral'] == 'L' else 'right'} heading "
                   f"{round_heading(float(seg['hdg_exit'])):03d}")
    if seg["vertical"] in ("climb", "descend"):
        a = max(round(float(seg["alt_exit_ft"]) / 1000.0), 1) * 1000
        out.append(f"{seg['vertical']} and maintain "
                   + (f"FL{a // 100}" if a >= 18000 else f"{a:,}"))
    if seg["speed"] in ("accel", "decel"):
        out.append(f"{'increase' if seg['speed'] == 'accel' else 'reduce'} speed to "
                   f"{round_speed(float(seg['spd_exit_kt']))} kt")
    return out


def has_clause(seg) -> bool:
    return (seg["lateral"] in ("L", "R") or seg["vertical"] in ("climb", "descend")
            or seg["speed"] in ("accel", "decel"))


def observe(seg, obs: Observation, leg_type=None, phase=None, rng=None) -> dict:
    """The (text, confidence) pair for one clearance, plus what produced it.

    `seg` needs lateral, vertical, speed, hdg_exit, alt_exit_ft and spd_exit_kt. The RNG is
    consumed in the same order as `evaluation.compose`: channel first, then context.
    """
    rng = rng if rng is not None else np.random.default_rng(obs.seed)
    issued = issued_clauses(seg)
    if obs.mode == "custom":
        typed = obs.custom_text.strip()
        text = speak_numbers(typed) if obs.speak_numbers else typed
        return {"text": text, "confidence": float(obs.confidence), "core": typed,
                "corrupted": "", "issued": issued}
    if obs.mode == "clean":
        core = segment_directive(seg, spoken=True)
        conf, corrupted = float(obs.confidence), ""
    else:
        n = corrupt_segment(seg, NoiseConfig(profile=obs.profile,
                                             phrase_variant_p=obs.phrase_variant_p,
                                             keyword_dropout_p=obs.keyword_dropout_p), rng)
        core, conf, corrupted = n["noisy_directive"], float(n["asr_confidence"]), n["corrupted"]
    text = (augment_transcript(core, rng, leg_type=leg_type, phase=phase) if obs.context
            else (core or ""))
    return {"text": text, "confidence": conf, "core": core, "corrupted": corrupted,
            "issued": issued}
