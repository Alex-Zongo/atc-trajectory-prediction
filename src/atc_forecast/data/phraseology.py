"""Maneuver segment to FAA-phraseology clearance.

Each maneuver onset maps to one clearance element, and compound segments combine them:

    turn     -> "turn {left|right} heading {hdg}"
    climb    -> "climb and maintain {alt}"
    descend  -> "descend and maintain {alt}"
    accel    -> "increase speed to {kt} knots"
    decel    -> "reduce speed to {kt} knots"

Following FAA Order 7110.65: headings are three spoken digits with 000 spoken as 360,
altitudes round to the nearest 1,000 ft and become flight levels at or above 18,000 ft,
speeds round to the nearest 10 kt. Targets are the segment's EXIT values, which is what
the controller assigned.

The modelling target is `segment_directive`, which omits the callsign. The callsign is a
nuisance variable: the same directive is issued to hundreds of different callsigns, so it
carries no maneuver information, and leaving it in lets an encoder learn carrier-to-route
confounds. Identity that does matter enters as the aircraft performance class instead.
"""

import re

import pandas as pd

DIGITS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
          "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "niner"}

PHONETIC = {
    "A": "alpha", "B": "bravo", "C": "charlie", "D": "delta", "E": "echo",
    "F": "foxtrot", "G": "golf", "H": "hotel", "I": "india", "J": "juliett",
    "K": "kilo", "L": "lima", "M": "mike", "N": "november", "O": "oscar",
    "P": "papa", "Q": "quebec", "R": "romeo", "S": "sierra", "T": "tango",
    "U": "uniform", "V": "victor", "W": "whiskey", "X": "xray", "Y": "yankee",
    "Z": "zulu",
}

TELEPHONY = {
    "SWA": "Southwest", "UAL": "United", "AAL": "American", "DAL": "Delta",
    "ASA": "Alaska", "SKW": "SkyWest", "JBU": "JetBlue", "FFT": "Frontier",
    "NKS": "Spirit", "AAY": "Allegiant", "HAL": "Hawaiian", "QXE": "Horizon",
    "ENY": "Envoy", "RPA": "Brickyard", "EDV": "Endeavor", "JIA": "Blue Streak",
    "AWI": "Air Wisconsin", "ASH": "Mesa", "FDX": "FedEx", "UPS": "UPS",
    "GTI": "Giant", "ATN": "Air Transport", "ABX": "Abex", "SWG": "Swift",
    "ACA": "Air Canada", "AMX": "Aeromexico", "VOI": "Volaris",
    "BAW": "Speedbird", "DLH": "Lufthansa", "AFR": "Air France", "KLM": "KLM",
    "UAE": "Emirates", "QTR": "Qatari", "EVA": "Eva", "ANA": "All Nippon",
    "JAL": "Japan Air", "KAL": "Korean Air", "AAR": "Asiana", "CPA": "Cathay",
    "SIA": "Singapore", "QFA": "Qantas", "CCA": "Air China",
    "CES": "China Eastern", "CSN": "China Southern", "PAL": "Philippine",
    "EIN": "Shamrock", "VIR": "Virgin", "ICE": "Iceair", "FIN": "Finnair",
}


def spoken_digits(s) -> str:
    return " ".join(DIGITS[c] for c in str(s) if c in DIGITS)


def spoken_alnum(s) -> str:
    out = []
    for c in str(s).upper():
        if c in PHONETIC:
            out.append(PHONETIC[c])
        elif c in DIGITS:
            out.append(DIGITS[c])
    return " ".join(out)


def _is_missing(cs: str) -> bool:
    return cs.strip() == "" or cs.strip().lower() in {"nan", "none", "null"}


def ac_class(callsign: str) -> str:
    """Coarse aircraft class, the part of identity that is legitimately predictive."""
    cs = str(callsign).strip().upper()
    if _is_missing(cs):
        return "unknown"
    if re.fullmatch(r"[A-Z]{3}\d[A-Z0-9]*", cs):
        return "airline"
    if re.fullmatch(r"N\d[A-Z0-9]*", cs):
        return "ga"
    if re.fullmatch(r"[0-9A-F]{6}", cs):
        return "unknown"
    return "other"


def telephony(callsign: str, spoken: bool = True) -> str:
    cs = str(callsign).strip().upper()
    if _is_missing(cs):
        return "unknown traffic" if spoken else "UNKNOWN"
    m = re.fullmatch(r"([A-Z]{3})(\d[A-Z0-9]*)", cs)
    if m:
        designator, flight = m.group(1), m.group(2)
        name = TELEPHONY.get(designator)
        if not spoken:
            return f"{name or designator} {flight}"
        return f"{name or spoken_alnum(designator)} {spoken_digits(flight)}".strip()
    if re.fullmatch(r"N\d[A-Z0-9]*", cs):
        return "November " + spoken_alnum(cs[1:]) if spoken else cs
    return spoken_alnum(cs) if spoken else cs


def round_heading(hdg: float, base: int = 5) -> int:
    """Round to the nearest `base` degrees; heading 000 is expressed as 360."""
    h = int(base * round(float(hdg) / base)) % 360
    return h or 360


def round_altitude(ft: float) -> int:
    return int(round(float(ft) / 1000.0) * 1000)


def round_speed(kt: float, base: int = 10) -> int:
    return int(base * round(float(kt) / base))


def spoken_heading(hdg: float) -> str:
    return spoken_digits(f"{round_heading(hdg):03d}")


def readable_heading(hdg: float) -> str:
    return f"{round_heading(hdg):03d}"


def spoken_altitude(ft: float) -> str:
    a = max(round_altitude(ft), 1000)
    if a >= 18000:
        return f"flight level {spoken_digits(a // 100)}"
    return f"{spoken_digits(a // 1000)} thousand"


def readable_altitude(ft: float) -> str:
    a = max(round_altitude(ft), 1000)
    return f"FL{a // 100}" if a >= 18000 else f"{a:,}"


def _directive_parts(seg, spoken: bool) -> list[str]:
    parts: list[str] = []
    if seg["lateral"] in ("L", "R"):
        side = "left" if seg["lateral"] == "L" else "right"
        hdg = spoken_heading(seg["hdg_exit"]) if spoken else readable_heading(seg["hdg_exit"])
        parts.append(f"turn {side} heading {hdg}")
    if seg["vertical"] in ("climb", "descend"):
        alt = (spoken_altitude if spoken else readable_altitude)(seg["alt_exit_ft"])
        parts.append(f"{seg['vertical']} and maintain {alt}")
    if seg["speed"] in ("accel", "decel"):
        verb = "increase" if seg["speed"] == "accel" else "reduce"
        kt = round_speed(seg["spd_exit_kt"])
        parts.append(f"{verb} speed to {spoken_digits(kt) if spoken else kt} knots")
    return parts


def segment_directive(seg, spoken: bool = True) -> str:
    """The clearance WITHOUT the callsign; this is the modelling target."""
    parts = _directive_parts(seg, spoken)
    return ", ".join(parts) + "." if parts else ""


def segment_instruction(seg, spoken: bool = True) -> str:
    """Full clearance WITH the callsign, the realistic transcript form. Empty for steady
    segments, where no clearance is issued."""
    parts = _directive_parts(seg, spoken)
    if not parts:
        return ""
    return f"{telephony(seg['callsign'], spoken=spoken)}, " + ", ".join(parts) + "."


def add_instructions(segs: pd.DataFrame) -> pd.DataFrame:
    """Add the text and identity columns:

      atc_instruction        spoken, with callsign
      atc_readable           readable, with callsign
      atc_directive          readable, no callsign     (modelling target)
      atc_directive_spoken   spoken, no callsign       (ASR-aligned target)
      ac_class, callsign_known
    """
    segs = segs.copy()
    segs["atc_instruction"] = segs.apply(segment_instruction, axis=1)
    segs["atc_readable"] = segs.apply(lambda s: segment_instruction(s, spoken=False), axis=1)
    segs["atc_directive"] = segs.apply(lambda s: segment_directive(s, spoken=False), axis=1)
    segs["atc_directive_spoken"] = segs.apply(segment_directive, axis=1)
    segs["ac_class"] = segs["callsign"].apply(ac_class)
    segs["callsign_known"] = segs["ac_class"] != "unknown"
    return segs
