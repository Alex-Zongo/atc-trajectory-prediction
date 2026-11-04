"""The interpretation Z: four slots, each holding one atom from a finite list.

An atom is one complete value of one slot, a direction and a target together, so a slot's
distribution is a single softmax and the probability of any (direction, value) pair can be
read off exactly.

    status    0 own-motion   the message is for this aircraft and commands motion
              1 own-other    for this aircraft, but no motion (frequency, squawk, report)
              2 other        addressed to another aircraft
    lateral   NONE, MAINTAIN, L005 ... L360, R005 ... R360           2 + 2 x 72  = 146
    vertical  NONE, MAINTAIN, climb 1,000 ... 52,000 ft, descend ...  2 + 2 x 52  = 106
    speed     NONE, MAINTAIN, increase 60 ... 600 kt, reduce ...      2 + 2 x 55  = 112

NONE means "nothing said about this axis", which carries no information. MAINTAIN means
"keep the present value" ("fly present heading", "maintain FL240" while level at it,
"maintain present speed"), which does. If status is not own-motion, every motion slot is
NONE by construction.

Values follow the phraseology grids: 5 deg headings (360 for north), 1,000 ft altitudes,
10 kt speeds. Altitude is pressure altitude, which is what ATC assigns. The speed slot is
ground speed, as the targets are (the caveat is named in the paper).

The CTP turns atoms into instruction tokens with features computed against the present
state (relative turn, altitude change, speed change, and whether the direction agrees);
the tables below make that a lookup, with no text in between.
"""

import numpy as np

STATUS = ["own_motion", "own_other", "other"]
NONE, MAINTAIN = 0, 1

HDG_VALUES = np.arange(5, 361, 5)                    # 72
ALT_VALUES = np.arange(1000, 52001, 1000)            # 52
SPD_VALUES = np.arange(60, 601, 10)                  # 55


def _slot(values, dirs):
    """Per-atom tables: kind (0 none, 1 maintain, 2 directed), sign (-1/+1), value."""
    n = 2 + len(dirs) * len(values)
    kind = np.zeros(n, np.int64)
    sign = np.zeros(n, np.float32)
    value = np.zeros(n, np.float32)
    kind[MAINTAIN] = 1
    for d, s in enumerate(dirs):
        sl = slice(2 + d * len(values), 2 + (d + 1) * len(values))
        kind[sl], sign[sl], value[sl] = 2, s, values
    return {"kind": kind, "sign": sign, "value": value}


LATERAL = _slot(HDG_VALUES, (-1.0, +1.0))            # L, R
VERTICAL = _slot(ALT_VALUES, (+1.0, -1.0))           # climb, descend
SPEED = _slot(SPD_VALUES, (+1.0, -1.0))              # increase, reduce
SLOTS = {"lateral": LATERAL, "vertical": VERTICAL, "speed": SPEED}
SIZES = {"status": 3, "lateral": len(LATERAL["kind"]), "vertical": len(VERTICAL["kind"]),
         "speed": len(SPEED["kind"])}


def _index(values, v, d):
    v = np.asarray(v, float)
    i = np.clip(np.searchsorted(values, v), 0, len(values) - 1)
    # nearest grid value, then offset by the direction block
    i = np.where((i > 0) & (np.abs(values[np.maximum(i - 1, 0)] - v) <= np.abs(values[i] - v)),
                 i - 1, i)
    return 2 + np.asarray(d) * len(values) + i


def lateral_atom(lateral, hdg_exit):
    """'L'/'R' with the exit track, else NONE. Track rounds to 5 deg, 0 becomes 360."""
    lateral = np.asarray(lateral)
    h = np.round(np.nan_to_num(np.asarray(hdg_exit, float)) / 5.0) * 5 % 360
    h = np.where(h == 0, 360, h)
    turn = np.isin(lateral, ["L", "R"])
    return np.where(turn, _index(HDG_VALUES, h, (lateral == "R").astype(int)), NONE)


def vertical_atom(vertical, alt_exit_ft):
    vertical = np.asarray(vertical)
    a = np.clip(np.round(np.nan_to_num(np.asarray(alt_exit_ft, float)) / 1000.0) * 1000,
                ALT_VALUES[0], ALT_VALUES[-1])
    on = np.isin(vertical, ["climb", "descend"])
    return np.where(on, _index(ALT_VALUES, a, (vertical == "descend").astype(int)), NONE)


def speed_atom(speed, spd_exit_kt):
    speed = np.asarray(speed)
    s = np.clip(np.round(np.nan_to_num(np.asarray(spd_exit_kt, float)) / 10.0) * 10,
                SPD_VALUES[0], SPD_VALUES[-1])
    on = np.isin(speed, ["accel", "decel"])
    return np.where(on, _index(SPD_VALUES, s, (speed == "decel").astype(int)), NONE)


def describe(slot: str, atom: int) -> str:
    """Readable atom, for figures and debugging."""
    if slot == "status":
        return STATUS[atom]
    t = SLOTS[slot]
    if atom == NONE:
        return "-"
    if atom == MAINTAIN:
        return {"lateral": "present hdg", "vertical": "maintain alt",
                "speed": "present spd"}[slot]
    v, s = int(t["value"][atom]), t["sign"][atom]
    if slot == "lateral":
        return f"{'L' if s < 0 else 'R'}{v:03d}"
    if slot == "vertical":
        lvl = f"FL{v // 100}" if v >= 18000 else f"{v:,}"
        return f"{'climb' if s > 0 else 'descend'} {lvl}"
    return f"{'increase' if s > 0 else 'reduce'} {v}"


def relative_features(slot: str, atoms, trk_deg, alt_ft, gs_kt):
    """Instruction-token features for atoms [N] against the present state [N].

    lateral   abs (sin h, cos h), signed turn in the commanded direction / 180,
              (sin, cos) of that turn, consistency (always 1: any turn is possible)
    vertical  abs alt / 10,000 ft, (target - current) / 10,000 ft, consistency
    speed     abs / 100 kt, (target - current) / 100 kt, consistency

    MAINTAIN reads as the present value: relative 0, consistent. NONE is all zeros (the
    model adds a learned "nothing said" embedding instead). Consistency is 1 when the
    commanded direction agrees with the sign of the change, e.g. "descend" to a level
    below the aircraft; an inconsistent atom is a strong sign of misrecognition.
    """
    t = SLOTS[slot]
    a = np.asarray(atoms)
    kind, sign, val = t["kind"][a], t["sign"][a], t["value"][a].astype(float)
    on, keep = kind == 2, kind == 1
    if slot == "lateral":
        cur = np.asarray(trk_deg, float)
        d = np.where(sign > 0, (val - cur) % 360, -((cur - val) % 360))
        d = np.where(on, d, 0.0)
        h = np.radians(np.where(on, val, np.where(keep, cur, 0.0)))
        f = np.stack([np.sin(h), np.cos(h), d / 180, np.sin(np.radians(d)),
                      np.cos(np.radians(d)), (on | keep).astype(float)], -1)
        f[kind == 0] = 0
        return f.astype(np.float32)
    cur = np.asarray(alt_ft if slot == "vertical" else gs_kt, float)
    scale = 10_000.0 if slot == "vertical" else 100.0
    target = np.where(on, val, cur)
    rel = (target - cur) / scale
    cons = np.where(on, (np.sign(target - cur) == sign).astype(float), 1.0)
    f = np.stack([target / scale, rel, cons], -1)
    f[kind == 0] = 0
    return f.astype(np.float32)


FEATURE_DIM = {"lateral": 6, "vertical": 3, "speed": 3}
