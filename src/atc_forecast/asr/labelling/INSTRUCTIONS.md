# Labelling ATC transcripts: instructions

**Version 2 (2026-10-22).** Mention this version number in your final report.

You are labelling the **content** of air traffic control radio transcripts. For each speaker
turn, record what was literally said: who is addressed, and every instruction element with
its value. These labels are later compared against speech-recognizer output, so they must
describe **only what the text says**.

## The one rule that matters most

**Label literally. Never repair, complete or infer.**

- If a number is incomplete or implausible, record it as written ("descend flight level one"
  is flight level 1) and set `status` accordingly. Do not guess what was meant.
- If a value is missing ("descend and maintain"), record the clause with `value: null` and
  `status: "value_missing"`.
- Do not use knowledge of real airports, airlines or procedures to fill gaps.
- Do not read or use any file other than the input file you are given and this document.

## Input

One JSON object per line:

```json
{"item_id": "...", "corpus": "atco2" | "atcosim",
 "turns": [{"turn": 0, "role": "controller" | "pilot" | "unknown", "text": "..."}]}
```

A turn is one speaker's continuous transmission. `role` comes from the corpus and is
reliable; do not change it.

## Output

One JSON object per input line, same `item_id`, one entry per input turn, in order:

```json
{"item_id": "...",
 "turns": [{
   "turn": 0,
   "type": "instruction",
   "language": "english",
   "callsign": {"text": "Oscar Kilo Papa Mike Bravo", "normalized": "oscar kilo papa mike bravo"},
   "clauses": [{
     "kind": "altitude",
     "action": "descend",
     "direction": "descend",
     "value": 100,
     "unit": "FL",
     "value_words": "flight level one hundred",
     "status": "complete",
     "relative": false,
     "repeated": false,
     "span": "descend flight level one hundred"
   }],
   "notes": null
 }]}
```

### Turn fields

| field | values |
|---|---|
| `type` | `instruction` (directs the aircraft) · `readback` (pilot repeating an instruction) · `request` · `report` (position, altitude, information) · `acknowledgement` (roger, wilco, …) · `other` |
| `language` | `english` · `mixed` (contains a non-English part) · `non_english` · `no_speech` |
| `callsign` | the aircraft addressed or speaking, or `null`. `text`: verbatim. `normalized`: lowercase, with digits written as numerals (`lufthansa 4 2 1`), letters kept as spoken words. If several callsigns occur, use the first and mention the others in `notes`. |
| `notes` | anything ambiguous, else `null` |

### Clause fields

One clause per instruction element. "Turn left heading 270, descend flight level 80" is two
clauses. Label clauses in readbacks too: a readback is still content a recognizer must get
right.

| field | values |
|---|---|
| `kind` | `heading` · `turn` (direction or relative turn without a heading) · `altitude` · `speed` · `vertical_rate` · `frequency` · `squawk` · `direct_to` · `route` · `approach` · `landing` · `takeoff` · `line_up` · `taxi` · `hold` · `go_around` · `report` (an instruction to report something) · `traffic` · `weather` · `altimeter` · `other` |
| `action` | the instruction word as spoken (`turn`, `climb`, `descend`, `maintain`, `reduce`, `increase`, `contact`, `squawk`, `proceed`, `cleared`, …), or `null` |
| `direction` | `left` · `right` · `climb` · `descend` · `increase` · `reduce`, only if spoken; else `null` |
| `value` | the number as spoken, as a number; for `squawk`, a string of 4 digits; for `direct_to`/`route`/`approach`/`landing`/`takeoff`, the named point or runway as a string; else `null` |
| `unit` | `deg` · `ft` · `FL` · `kt` · `mach` · `MHz` · `code` · `fpm` · `nm` · `runway` · `point`, or `null` |
| `value_words` | the exact words of the value, or `null` |
| `status` | `complete` · `value_missing` (instruction given without its value) · `truncated` (cut off mid-value or mid-word) · `corrected` (speaker corrected it; give the final value and put the earlier one in `notes`) |
| `relative` | `true` for relative values ("turn left by 20 degrees", "climb 2000 feet") |
| `repeated` | `true` if the same element is said twice in the same turn; label it once |
| `span` | the exact words of the clause |

### Value conventions

- **Altitude:** keep the reference the speaker used. "flight level one hundred" →
  `value: 100, unit: "FL"`. "four thousand five hundred" → `value: 4500, unit: "ft"`. "level
  one hundred" → `value: 100, unit: "FL"`.
- **Heading:** degrees as spoken; "zero niner zero" → 90, "three six zero" → 360.
- **Speed:** knots as a number; "mach decimal seven eight" → `value: 0.78, unit: "mach"`.
- **Frequency:** "one three three decimal four" → `value: 133.4, unit: "MHz"`. "point" and
  "decimal" both mean the decimal separator.
- **Digits:** "niner" = 9, "tree" = 3, "fife" = 5. Digit-by-digit numbers form one number.
  "Thousand" and "hundred" are multipliers within one value.

### Conventions for common cases

- **Wind:** two `weather` clauses, one for the direction (`unit: deg`) and one for the speed
  (`unit: kt`), each with its own `value_words`.
- **Runways:** `value` is the runway designator as a string of what was spoken: `"24"`,
  `"09"`, `"28L"`. Use it for `takeoff`, `landing`, `approach`, `line_up` and `taxi`.
- **Taxi and holding points:** `kind: taxi` (or `hold`), `value` the named point as a string
  (`"echo one"`), `unit: "point"`.
- **Circuit and pattern** ("join downwind", "report base", "number two"): `kind: other`,
  unless the words include a landing or approach clearance.
- **Sequence numbers** ("number one"): `kind: other`, `value: 1`.
- **"Frequency change approved", "monitor tower"** without a frequency: `kind: frequency`,
  `value: null`, `status: complete`, since no number is expected.
- **Numbers spoken without their decimal separator** ("one three five six eight zero" for a
  frequency): record literally (`135680`) and say so in `notes`.
- **Controller questions** ("confirm …", "say …"): turn `type: other`; still label any
  instruction content in it.
- **Role and content disagree** (a `controller` turn that reads like a pilot report): keep
  the given `role` and choose `type` from the content.

## Corpus markup

Do not label markup as content.

**ATCO2:** `[hes]` hesitation · `[noise]` noise · `[spk]` speaker noise · `[ukn]` unknown
word · `[XT]` cross-talk · `[NE …] … [/NE]` non-English part (set `language: "mixed"` and do
not label content inside it).

**ATCOSIM:** `~x` a letter spelled out (`~p ~s ~a` = the letters P S A, often a callsign) ·
`@word` an unusual word, often an airline or place name (drop the `@`) · `word=` a cut-off
word (`co=`) · `<OT> … </OT>` off-topic talk, not content · `<FL> … </FL>` foreign
language · `[EMPTY]` no speech (`language: "no_speech"`) · `[HNOISE]` noise ·
`[FRAGMENT]`, `[UNKNOWN]`, `[NONSENSE]` unintelligible.

## Worked examples

**Controller instruction followed by a readback** (two turns):

- controller: "Oscar Kilo Papa Mike Bravo descend flight level one hundred" →
  `type: instruction`, callsign "Oscar Kilo Papa Mike Bravo", one clause: altitude, descend,
  100 FL.
- pilot: "level one hundred Oscar Kilo Papa Mike Bravo" → `type: readback`, same callsign,
  one clause: `kind: altitude, action: null, direction: null, value: 100, unit: FL`.

**Missing value:** "speedbird two three turn right heading" → heading clause, `direction:
right`, `value: null`, `status: value_missing`.

**Correction:** "turn left heading two seven zero correction two eight zero" → one heading
clause, `value: 280`, `status: corrected`, notes "first said 270".

**Not an instruction:** "roger good day" → `type: acknowledgement`, `clauses: []`.

## Output rules

- One output line per input line, valid JSON, nothing else on the line.
- Keep every input turn, in order, even if it has no clauses.
- When unsure, choose the literal reading and explain in `notes`.
