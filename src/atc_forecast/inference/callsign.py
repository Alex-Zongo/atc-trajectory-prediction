"""Find which aircraft a transcript names, among the callsigns on ADS-B at that time.

Recognizers write flight numbers in many shapes: "United 2437", "united two four three
seven", "United twenty four thirty seven", "United, 24-37", "Skywest 514.90" (the number
run on into the next one). Parsing callsigns freely from that is fragile, so matching
goes the other way: for each candidate callsign broadcast nearby (UAL2437), its airline's
spoken name (telephony, data/phraseology.py) must appear in the text and its flight
number must begin the digit string that follows the name and end on a token boundary
("Skywest 514.90" can be SKW514 or SKW51490, never SKW5; "United 20367" is not UAL2).
Among candidates of the same airline the longest matching flight number wins.

    find_callsigns("climb and maintain 5000 United 24-37 heavy", ["UAL2437", "SWA12"])
    -> [Match(callsign="UAL2437", start=4, end=7, role="pilot")]

Only airlines with a telephony entry are matched; general-aviation tails are left out
(precision first). `role` guesses the speaker from where the callsign sits: a controller
opens with it; a pilot's readback ends with it, and a pilot's call names the facility
first ("NorCal, United 12, ...").
"""

import re
from dataclasses import dataclass

from ..data.phraseology import TELEPHONY

UNITS = {"zero": 0, "oh": 0, "o": 0, "one": 1, "two": 2, "three": 3, "tree": 3, "four": 4,
         "fower": 4, "five": 5, "fife": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "niner": 9}
TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
        "eighty": 80, "ninety": 90}
# a pilot's call opens with the facility it is calling ("NorCal, United 12, ...")
FACILITIES = {"norcal", "departure", "approach", "center", "oakland", "tower", "ground", "socal",
              "bay", "clearance", "radio"}
NAME_TO_ICAO = {}
for code, name in TELEPHONY.items():
    NAME_TO_ICAO.setdefault(name.lower(), []).append(code)


def tokens(text: str) -> list[str]:
    """Lower-case words, with digit runs split off ("24-37" -> 24, 37; "514.90" -> 514, 90)."""
    t = str(text).lower().replace("-", " ")
    return re.findall(r"[a-z']+|\d+", t)


def digits_after(toks: list[str], i: int, max_tokens: int = 8, prefixes: bool = False):
    """The digit string spoken from token i on: numerals, digit words, teens and tens
    ("twenty four" -> 24), until the first word that is not part of a number. Returns
    (digits, next token index); with prefixes=True, also the digit strings that end on a
    token boundary ("24-80" -> 24, 2480), the only places a flight number may end."""
    out, j, cuts = "", i, []
    while j < len(toks) and j - i < max_tokens:
        w = toks[j]
        if w.isdigit():
            out += w
        elif w in UNITS:
            out += str(UNITS[w])
        elif w in TEENS:
            out += str(TEENS[w])
        elif w in TENS:
            if j + 1 < len(toks) and toks[j + 1] in UNITS and UNITS[toks[j + 1]] > 0:
                out += str(TENS[w] + UNITS[toks[j + 1]])
                j += 1
            else:
                out += str(TENS[w])
        elif w == "hundred" and out:
            out += "00"
        else:
            break
        j += 1
        cuts.append(out)
    return (out, j, cuts) if prefixes else (out, j)


@dataclass
class Match:
    callsign: str
    start: int          # token index of the airline name
    end: int            # one past the last number token
    role: str           # "controller" (callsign first), "pilot" (callsign last) or "unknown"


def find_callsigns(text: str, candidates) -> list[Match]:
    """The candidate callsigns (ICAO form, e.g. UAL2437) the text names, in text order."""
    toks = tokens(text)
    by_code = {}
    for cs in candidates:
        m = re.fullmatch(r"([A-Z]{3})(\d+)[A-Z]?", str(cs).strip().upper())
        if m:
            by_code.setdefault(m.group(1), []).append((m.group(2), str(cs).strip().upper()))
    out, i = [], 0
    names = sorted(NAME_TO_ICAO, key=lambda n: -len(n.split()))
    while i < len(toks):
        hit = None
        for name in names:
            parts = name.split()
            if toks[i:i + len(parts)] == parts:
                hit = (name, len(parts))
                break
        if hit is None:
            i += 1
            continue
        name, n = hit
        digits, end, cuts = digits_after(toks, i + n, prefixes=True)
        best = None
        for code in NAME_TO_ICAO[name]:
            for num, cs in by_code.get(code, []):
                if num in cuts and (best is None or len(num) > len(best[0])):
                    best = (num, cs)
        if best:
            opens = i == 0 or (i == 1 and toks[0] not in FACILITIES)
            role = ("controller" if opens else "pilot" if (end >= len(toks) - 2 or i == 1) else "unknown")
            out.append(Match(best[1], i, end, role))
            i = end
        else:
            i += n
    return out


__all__ = ["find_callsigns", "Match", "digits_after", "tokens"]
