"""Tokenizer, realistic transcript augmentation, and the Stage-1 dataset.

A clean clearance is not what a recognizer's output looks like, so each training example is
wrapped to resemble a real transmission:

  * a CALLSIGN prefix the model must learn to ignore;
  * surrounding PHRASEOLOGY, gated to be operationally coherent with the segment's flight
    phase, which the model must learn to attend away from;
  * with probability `p_irrelevant` the whole example becomes a NON-clearance transmission
    (readback, position report, frequency change), labelled is_clearance = 0 with every
    clause absent, so the model learns that most radio traffic carries no instruction.

The surrounding context is phase-coherent on purpose. This dataset is entirely airborne, so
there are no taxi or ground-movement clearances; only a descent gets an approach or landing
clearance, and a climb-out gets en-route handoffs. Incoherent context would teach the model
to expect "cleared for takeoff" mid-descent and would put runway numbers where a heading
belongs.
"""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..channel.model import FILLERS, GARBLE, NoiseConfig, corrupt_segment
from ..units import ALT_SCALE_FT, N_HEAD_BINS, SPD_SCALE_KT, VR_SCALE_FPM
from .phraseology import spoken_alnum, spoken_digits

PAD, UNK = 0, 1

AIRLINES = ["southwest", "united", "american", "delta", "alaska", "skywest",
            "jetblue", "fedex", "spirit", "frontier"]

IRRELEVANT = [
    "{cs} roger", "{cs} say again", "{cs} contact tower one two zero point niner",
    "{cs} contact approach one three five point six five", "{cs} squawk one two zero zero",
    "{cs} verify altitude", "{cs} radar contact", "{cs} good day", "{cs} wilco",
    "{cs} airport in sight", "wind two seven zero at one five", "{cs} ident",
]

# Numbers in the context pools always sit behind an unambiguous non-directive cue word
# (runway, contact, squawk, point) and never form a second turn, climb or speed clause,
# so the model can learn to route past them rather than mis-reading "runway two eight"
# as a heading.
CTX_ANY = [
    "contact center one three three point niner", "squawk four seven two one", "ident",
    "traffic in sight", "traffic passing left to right", "radar contact",
    "verify your altitude", "say airspeed",
]
CTX_ARRIVAL = [
    "cleared visual approach runway two eight left", "cleared to land runway three zero",
    "expect the ils runway two eight right", "join the localizer",
    "contact approach one three five point six five",
    "contact tower one two zero point niner", "report the airport in sight",
]
CTX_ENROUTE = [
    "when able proceed direct", "resume own navigation",
    "contact departure one two five point six five", "expect higher in ten miles",
]
RICH_CONTEXT = CTX_ANY + CTX_ARRIVAL + CTX_ENROUTE

# The garble variants and fillers MUST be in vocabulary, or every garbled keyword collapses
# to <unk> and the surface realism the channel injected is destroyed before the model sees
# it, so it never learns to read through it.
_GARBLE_STRINGS = [w for variants in GARBLE.values() for w in variants]
AUG_CORPUS = (AIRLINES + [s.format(cs="southwest one two three") for s in IRRELEVANT]
              + RICH_CONTEXT + _GARBLE_STRINGS + FILLERS
              + ["november", "point", "roger", "wilco", "ident", "airport", "sight"])


def _context_pool(leg_type, phase):
    pool = list(CTX_ANY)
    if leg_type == "arrival" or phase == "descent":
        pool += CTX_ARRIVAL
    if leg_type in ("departure", "overflight") or phase in ("climb", "cruise"):
        pool += CTX_ENROUTE
    return pool


def tokenize(text) -> list[str]:
    """Split on whitespace and commas, with numbers broken into individual digits and a
    leading "fl" kept as its own token."""
    toks = []
    for w in re.split(r"[\s,]+", str(text).lower().strip(" .")):
        w = w.strip(".,")
        if not w:
            continue
        m = re.fullmatch(r"(fl)?(\d[\d,]*)", w)
        if m:
            if m.group(1):
                toks.append("fl")
            toks.extend(list(m.group(2).replace(",", "")))
        else:
            toks.append(w)
    return toks


class Vocab:
    def __init__(self, texts, extra=()):
        seen = set()
        for t in list(texts) + list(extra):
            seen.update(tokenize(t))
        self.itos = ["<pad>", "<unk>"] + sorted(seen)
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text, max_len) -> list[int]:
        ids = [self.stoi.get(t, UNK) for t in tokenize(text)][:max_len]
        return ids + [PAD] * (max_len - len(ids))

    @classmethod
    def from_itos(cls, itos):
        """Rebuild from a saved checkpoint's token list."""
        v = cls.__new__(cls)
        v.itos = list(itos)
        v.stoi = {s: i for i, s in enumerate(v.itos)}
        return v


def random_callsign(rng) -> str:
    """A spoken callsign: airline telephony plus flight number, or a spelled GA tail."""
    if rng.random() < 0.65:
        n = "".join(str(rng.integers(0, 10)) for _ in range(int(rng.integers(1, 5))))
        return f"{rng.choice(AIRLINES)} {spoken_digits(n)}"
    tail = ("".join(str(rng.integers(0, 10)) for _ in range(int(rng.integers(2, 4))))
            + "".join(chr(ord("a") + int(rng.integers(0, 26)))
                      for _ in range(int(rng.integers(0, 3)))))
    return "november " + spoken_alnum(tail)


def augment_transcript(text, rng, leg_type=None, phase=None, scale=1.0) -> str:
    """Wrap a noisy clearance in realistic surrounding context the model must ignore.

    `scale` multiplies the context frequency; 0 leaves the callsign only.
    """
    pool = _context_pool(leg_type, phase)
    lead_p, trail_p = min(0.25 * scale, 0.95), min(0.45 * scale, 0.95)
    parts = []
    if rng.random() < 0.85:                     # callsigns are nearly always present on real radio
        parts.append(random_callsign(rng))
    if pool and rng.random() < lead_p:
        parts.append(str(rng.choice(pool)))
    parts.append(str(text).rstrip("."))
    if pool and rng.random() < trail_p:
        parts.append(str(rng.choice(pool)))
    return " ".join(p for p in parts if p).strip() + "."


def irrelevant_transmission(rng) -> str:
    return rng.choice(IRRELEVANT).format(cs=random_callsign(rng))


@dataclass
class DataConfig:
    """Stage-1 data settings.

    `resample=True` draws FRESH noise every epoch, which is what training wants.
    `resample=False` gives deterministic per-item noise, which evaluation and calibration
    need to be reproducible.

    `aug_factor` draws that many independent corruptions per clean sample each epoch. It is
    the tractable version of "show the model all noisings": coverage without enumeration,
    at the cost of a proportionally longer epoch.
    """
    max_len: int = 48
    max_degradation: float = 0.6
    p_irrelevant: float = 0.15
    fixed_degradation: float | None = None
    aug_factor: int = 1
    profile: str = "jacktol"
    resample: bool = False
    phrase_variant_p: float = 0.0
    keyword_dropout_p: float = 0.0
    context_scale: float = 1.0


class ClearanceDataset(Dataset):
    """Noisy transcript to clearance targets.

    Only the transcript text is stochastic. Every target is a deterministic function of the
    true clearance, so they are precomputed once and __getitem__ does only the unavoidable
    string work.
    """

    FIELD_COLS = ["lateral", "vertical", "speed", "hdg_exit", "alt_exit_ft", "spd_exit_kt"]

    def __init__(self, meta: pd.DataFrame, vocab: Vocab, dcfg: DataConfig, seed: int = 0):
        self.meta = meta.reset_index(drop=True)
        self.vocab, self.dcfg, self.seed = vocab, dcfg, seed
        self._f = self.meta[self.FIELD_COLS].to_dict("records")
        self._legtype = self.meta["leg_type"].to_numpy() if "leg_type" in self.meta else None
        self._phase = self.meta["phase"].to_numpy() if "phase" in self.meta else None

        lat, vert, sp = (self.meta[c].to_numpy() for c in ("lateral", "vertical", "speed"))
        num = lambda c: pd.to_numeric(self.meta[c], errors="coerce").to_numpy()
        hdg, alt, kts = num("hdg_exit"), num("alt_exit_ft"), num("spd_exit_kt")
        has_t = np.isin(lat, ["L", "R"])
        has_v = np.isin(vert, ["climb", "descend"])
        has_s = np.isin(sp, ["accel", "decel"])
        # issued values, quantised exactly as the channel renders them
        ih = (np.round(np.nan_to_num(hdg) / 5) * 5).astype(int) % 360
        ih = np.where(ih == 0, 360, ih)
        ialt = np.maximum(np.round(np.nan_to_num(alt) / 1000.0), 1) * 1000
        ikt = np.round(np.nan_to_num(kts) / 10.0) * 10
        T = torch.tensor
        self.targets = {
            "has_turn": T(has_t, dtype=torch.float32),
            "has_vert": T(has_v, dtype=torch.float32),
            "has_speed": T(has_s, dtype=torch.float32),
            "turn_dir": T(lat == "R", dtype=torch.float32),
            "head_bin": T((ih // 5) % N_HEAD_BINS, dtype=torch.long),
            "true_hdg_rad": T(np.radians(ih) * has_t, dtype=torch.float32),
            "vert_dir": T(vert == "climb", dtype=torch.float32),
            "alt_norm": T(ialt * has_v / ALT_SCALE_FT, dtype=torch.float32),
            "true_alt_ft": T(ialt * has_v, dtype=torch.float32),
            "spd_dir": T(sp == "accel", dtype=torch.float32),
            "spd_norm": T(ikt * has_s / SPD_SCALE_KT, dtype=torch.float32),
        }
        # optional state grounding, present only when meta went through attach_onset_state
        self._state = None
        if "alt_onset_ft" in self.meta.columns:
            col = lambda c: np.nan_to_num(pd.to_numeric(self.meta.get(c, 0.0),
                                                        errors="coerce").to_numpy())
            self._state = np.stack([col("alt_onset_ft") / ALT_SCALE_FT,
                                    col("vrate_onset_fpm") / VR_SCALE_FPM,
                                    col("gs_onset_kt") / SPD_SCALE_KT,
                                    col("hdg_onset_sin"), col("hdg_onset_cos")],
                                   1).astype(np.float32)

    @property
    def state_dim(self) -> int:
        return 0 if self._state is None else self._state.shape[1]

    def __len__(self):
        return len(self.meta) * max(1, self.dcfg.aug_factor)

    def _no_clearance(self, tokens):
        z = {"tokens": tokens, "confidence": torch.tensor(1.0),
             "is_clearance": torch.tensor(0.0), "degradation": torch.tensor(0.0)}
        for k, v in self.targets.items():
            z[k] = torch.zeros((), dtype=v.dtype)
        if self._state is not None:
            z["state"] = torch.zeros(self.state_dim, dtype=torch.float32)
        return z

    def __getitem__(self, i):
        rng = (np.random.default_rng() if self.dcfg.resample
               else np.random.default_rng(self.seed + i * 7919
                                          + int(torch.initial_seed() % 100_000)))
        # aug_factor copies share the sample but each index gets its own corruption
        real_i = i % len(self.meta)
        enc = lambda t: torch.tensor(self.vocab.encode(t, self.dcfg.max_len), dtype=torch.long)

        if rng.random() < self.dcfg.p_irrelevant:
            return self._no_clearance(enc(irrelevant_transmission(rng)))

        d = (self.dcfg.fixed_degradation if self.dcfg.fixed_degradation is not None
             else float(rng.uniform(0.0, self.dcfg.max_degradation)))
        n = corrupt_segment(self._f[real_i],
                            NoiseConfig(profile=self.dcfg.profile,
                                        mean_degradation=max(d, 1e-4),
                                        phrase_variant_p=self.dcfg.phrase_variant_p,
                                        keyword_dropout_p=self.dcfg.keyword_dropout_p), rng)
        if n["was_dropped"] and rng.random() < 0.5:
            return self._no_clearance(enc(random_callsign(rng)))

        item = {k: v[real_i] for k, v in self.targets.items()}
        item["tokens"] = enc(augment_transcript(
            n["noisy_directive"], rng,
            leg_type=self._legtype[real_i] if self._legtype is not None else None,
            phase=self._phase[real_i] if self._phase is not None else None,
            scale=self.dcfg.context_scale))
        item["confidence"] = torch.tensor(n["asr_confidence"], dtype=torch.float32)
        item["is_clearance"] = torch.tensor(1.0)
        item["degradation"] = torch.tensor(float(n["degradation"]))
        if self._state is not None:
            item["state"] = torch.tensor(self._state[real_i], dtype=torch.float32)
        return item
