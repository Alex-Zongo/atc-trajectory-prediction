"""Recognizer output to the SBM's inputs: tokens with per-word confidence.

The SBM was trained on spoken-form transcripts ("descend and maintain five thousand",
"heading two seven zero") with one confidence per word, drawn from distributions fitted
to real recognizers' word probabilities (channel/word_conf.py). A real recognizer's
output is brought to the same form one word at a time, so each word's probability
follows every token it becomes:

  "FL350" (0.91)    -> flight level three five zero   (0.91 each)
  "5,000" (0.97)    -> five thousand                  (0.97 each)
  "270" (0.62)      -> two seven zero                 (0.62 each)

Casing and punctuation are dropped. Words outside the training vocabulary become <unk>
and are counted, since a high unknown share means the recognizer's style differs from
the one the channel was fitted to.

Transmissions are read from JSON lines; see evaluation/real_v2.py for the format.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from ..app.observe import speak_numbers
from ..data.text import PAD, UNK, Vocab, tokenize
from ..data.messages import MessageConfig
from ..data.phraseology import telephony

MAX_LEN = MessageConfig.max_len
CS_LEN = MessageConfig.cs_len
PROB_KEYS = ("probability", "prob", "confidence", "conf", "score")
TEXT_KEYS = ("word", "text", "token")
# spoken forms the training transcripts write differently (ICAO "niner", "tree", "fife")
ALIASES = {"nine": "niner", "tree": "three", "fife": "five", "fower": "four"}


def normalize_word(word: str) -> list[str]:
    """One recognizer word to the spoken-form words training used."""
    w = str(word).strip().lower()
    w = re.sub(r"[^\w,.'\-]", "", w).strip(".,'-")
    if not w:
        return []
    if w == "fl":
        return ["flight", "level"]
    if w in ALIASES:
        return [ALIASES[w]]
    w = w.replace("-", " ")
    out = []
    for part in w.split():
        if re.search(r"\d", part):
            part = re.sub(r"^fl(\d{2,3})$", r"FL\1", part)
            out += speak_numbers(part).lower().split()
        else:
            out.append(part)
    return out


def words_to_tokens(words, default_conf: float = 1.0):
    """[(word, probability)] or [{"word": ..., "probability": ...}] or a plain string to
    (tokens, confidences)."""
    if isinstance(words, str):
        words = [(w, None) for w in words.split()]
    toks, conf = [], []
    for w in words:
        if isinstance(w, dict):
            text = next((w[k] for k in TEXT_KEYS if k in w), "")
            p = next((w[k] for k in PROB_KEYS if k in w and w[k] is not None), None)
        else:
            text, p = w
        p = default_conf if p is None else float(np.clip(float(p), 0.0, 1.0))
        for nw in normalize_word(text):
            for t in tokenize(nw):
                toks.append(t)
                conf.append(p)
    return toks, conf


def encode(vocab: Vocab, toks, conf, max_len: int = MAX_LEN):
    """Token ids [max_len] (int64), confidences [max_len] (float32) and the unknown count.
    Confidences are quantized to 1/255 as the training store keeps them."""
    ids = [vocab.stoi.get(t, UNK) for t in toks][:max_len]
    n_unk = sum(i == UNK for i in ids)
    kap = np.zeros(max_len, np.float32)
    kap[:len(ids)] = np.round(np.asarray(conf[:len(ids)], np.float32) * 255) / 255
    return np.array(ids + [PAD] * (max_len - len(ids)), np.int64), kap, n_unk


def encode_callsign(vocab: Vocab, callsign: str | None, cs_len: int = CS_LEN):
    """The aircraft's own callsign from ADS-B (e.g. UAL123), spoken as the controller
    would say it, as the SBM's exact callsign input."""
    return np.array(vocab.encode(telephony(callsign or "").lower(), cs_len), np.int64)


def _time(x) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    return datetime.fromisoformat(str(x).replace("Z", "+00:00")).timestamp()


@dataclass
class Transmission:
    """One controller transmission.

    id        any string
    t_end     end of the transmission (unix seconds or ISO 8601): the prediction time
    words     recognizer words with probabilities; or text, without confidences
    icao24    the addressee's ICAO 24-bit address, when known; else
    callsign  the addressee's callsign in ICAO form (UAL123), matched against ADS-B
    group     optional label to break the scores down by (instruction type, sector, ...)
    """
    id: str
    t_end: float
    words: list = field(default_factory=list)
    text: str | None = None
    icao24: str | None = None
    callsign: str | None = None
    group: str | None = None
    t0: float | None = None       # prediction time; t_end unless shifted
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict, i: int = 0):
        known = {"id", "t_end", "t0", "end", "words", "text", "hypothesis_text", "icao24",
                 "callsign", "group"}
        t = _time(d.get("t_end", d.get("t0", d.get("end"))))
        if t is None:
            raise ValueError(f"transmission {d.get('id', i)}: no t_end")
        return cls(id=str(d.get("id", i)), t_end=t, words=d.get("words") or [],
                   text=d.get("text", d.get("hypothesis_text")),
                   icao24=(str(d["icao24"]).lower() if d.get("icao24") else None),
                   callsign=d.get("callsign"), group=d.get("group"), t0=t,
                   extra={k: v for k, v in d.items() if k not in known})

    def tokens(self, default_conf: float = 1.0):
        return words_to_tokens(self.words if self.words else (self.text or ""), default_conf)


def read_transmissions(path) -> list[Transmission]:
    lines = Path(path).read_text().splitlines()
    return [Transmission.from_dict(json.loads(s), i) for i, s in enumerate(lines) if s.strip()]
