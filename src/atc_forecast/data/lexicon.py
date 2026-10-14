"""The ATC token vocabulary for Stage 1.

A vocabulary built only from readable, template-rendered text ("...maintain 5,000") does
not cover real ATC speech. Real controllers speak spelled-out numbers and the ICAO phonetic
alphabet, and use phraseology templates never generate. Measured against real transcripts,
a template-only vocabulary ran about 61 percent out-of-vocabulary even on the clean
reference text, which means the model literally cannot read real speech.

The curated core below fixes that. `harvest_tokens` adds a frequency-filtered long tail
from a real transcript corpus as a safety net, and `build_vocab` assembles the result.

Stage 1 must both use this vocabulary and train on SPOKEN-rendered text, so the tokens it
learns are the tokens a recognizer emits.
"""

import re
from collections import Counter

NUMBERS = ["zero", "one", "two", "three", "tree", "four", "five", "fife", "six", "seven",
           "eight", "nine", "niner", "ten", "hundred", "thousand"]
DIGIT_CHARS = list("0123456789")

PHONETIC = ["alfa", "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "fox", "golf",
            "hotel", "india", "juliett", "juliet", "kilo", "lima", "mike", "november",
            "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform", "victor",
            "whiskey", "xray", "x-ray", "yankee", "zulu"]

COMMANDS = ["turn", "left", "right", "heading", "fly", "climb", "climbing", "descend",
            "descending", "descent", "maintain", "and", "at", "to", "speed", "reduce",
            "increase", "slow", "expedite", "resume", "normal", "knots", "altitude",
            "level", "flight", "feet", "foot", "degrees", "cross", "proceed", "direct",
            "join", "intercept", "present", "continue", "stop"]

PHRASEOLOGY = ["cleared", "clear", "clearance", "approach", "tower", "ground", "departure",
               "center", "centre", "runway", "via", "hold", "holding", "short", "line",
               "up", "wait", "taxi", "contact", "monitor", "squawk", "ident", "radar",
               "traffic", "wind", "visual", "localizer", "localiser", "ils", "vector",
               "vectors", "final", "established", "vacate", "report", "expect", "approved",
               "unable", "own", "navigation", "point", "decimal", "then", "until",
               "further", "advised", "sequence", "number", "following", "behind",
               "position", "gate", "ramp", "apron", "frequency", "altimeter", "qnh"]

# readbacks and acknowledgements; these teach is_clearance = 0
CHATTER = ["roger", "wilco", "affirm", "affirmative", "negative", "standby", "say", "again",
           "good", "day", "morning", "afternoon", "evening", "night", "thank", "thanks",
           "you", "bye", "with", "information", "have", "copy", "copied", "checking", "in",
           "out", "understood", "confirm", "confirmed", "correct", "for"]

# repair operators, so "heading 320 correction 330" can be represented with 330 as target
CORRECTION = ["correction", "disregard", "belay", "previous", "cancel", "break", "make",
              "it", "that", "is", "was", "should", "be", "instead", "not", "no"]


def lexicon() -> list[str]:
    """The curated ATC token set."""
    out: set[str] = set()
    for group in (NUMBERS, DIGIT_CHARS, PHONETIC, COMMANDS, PHRASEOLOGY, CHATTER, CORRECTION):
        out.update(group)
    return sorted(out)


def harvest_tokens(texts, min_count: int = 3) -> list[str]:
    """Frequency-filtered real words from a transcript corpus.

    Restricted to purely alphabetic tokens of two or more characters, so recognizer and
    annotation junk (radio frequencies, runway identifiers, non-speech tags, partial-word
    markers) never enters the vocabulary. Digits are already covered by the curated core.
    """
    from .text import tokenize

    c = Counter(tok for t in texts for tok in tokenize(t))
    return sorted(w for w, n in c.items() if n >= min_count and re.fullmatch(r"[a-z]{2,}", w))


def spoken_directives(meta) -> list[str]:
    """Render the training clearances in spoken form, which is what Stage 1 must read."""
    from .phraseology import segment_directive

    cols = ["lateral", "vertical", "speed", "hdg_exit", "alt_exit_ft", "spd_exit_kt"]
    return [segment_directive(f, spoken=True) for f in meta[cols].to_dict("records")]


def load_transcripts(reports_glob: str) -> list[str]:
    """Reference and hypothesis text from ASR record JSONL files, for the vocab harvest."""
    import glob
    import json

    texts: list[str] = []
    for p in glob.glob(reports_glob):
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            texts += [r[k] for k in ("reference_text", "hypothesis_text") if r.get(k)]
    return texts


def build_vocab(meta, transcripts=None, harvest_min: int = 3):
    """Spoken training clearances, the augmentation corpus, the curated lexicon, and
    optionally a harvested real-transcript long tail."""
    from .text import AUG_CORPUS, Vocab

    extra = list(AUG_CORPUS) + lexicon()
    if transcripts:
        extra += harvest_tokens(transcripts, harvest_min)
    return Vocab(spoken_directives(meta), extra=extra)
