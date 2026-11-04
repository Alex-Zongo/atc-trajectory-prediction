# Running and evaluating v2 on real data

The models were trained on real ADS-B and simulated transcripts. This page covers real transcripts paired with real ADS-B: how to prepare them, how to run the models on them, how to evaluate, and what such an evaluation measures.

- [What you need](#what-you-need)
- [Transmissions file](#transmissions-file)
- [ADS-B](#ads-b)
- [Evaluate](#evaluate)
- [Predict from Python](#predict-from-python)
- [What a matched set measures](#what-a-matched-set-measures)
- [Checks behind the interface](#checks-behind-the-interface)
- [Known differences from the training data](#known-differences-from-the-training-data)

## What you need

1. The checkpoints: `uv run python scripts/artifacts.py fetch --models-only`.
2. Transmissions: one line per controller transmission, with the recognizer's words and confidences, its end time, and who it was for (see below).
3. ADS-B state vectors covering each transmission, from 40 s before its end to 64 s after.
4. The OpenSky aircraft database, for wake and engine class. It is downloaded once to `data/cache/` on first use; `--aircraft-db none` treats every type as unknown, which the models were trained to handle.

## Transmissions file

JSON lines, one object per transmission:

```json
{"id": "kjfk_app_0001",
 "t_end": 1767225612.4,
 "words": [{"word": "united", "probability": 0.98, "start": 0.42, "end": 0.71},
           {"word": "two", "probability": 0.91}, {"word": "descend", "probability": 0.88},
           {"word": "and", "probability": 0.99}, {"word": "maintain", "probability": 0.97},
           {"word": "5,000", "probability": 0.62}],
 "icao24": "a1b2c3",
 "group": "descend"}
```

| field | |
|---|---|
| `id` | any string |
| `t_end` | end of the transmission, unix seconds or ISO 8601 (`"2026-01-01T00:00:12.4Z"`). It is the prediction time t0; `--t0-offset` shifts it |
| `words` | the recognizer's words. Each takes `word` (or `text`) and `probability` (or `prob`, `confidence`); other keys are ignored. faster-whisper's `word.word` / `word.probability`, and this repository's `asr/transcribe.py` output, fit as they are |
| `text` | instead of `words`, when the recognizer gives no confidences; every word then gets `--default-conf` (1.0) |
| `icao24` | the addressee's 24-bit address, when the matching gives it; otherwise |
| `callsign` | its ICAO callsign (`UAL2`), looked up in the ADS-B within `--match-window` seconds (300) |
| `group` | optional label to break the scores down by: instruction type, sector, recognizer, ... |

**Matching transmissions to aircraft.** If your transmissions are not yet matched to aircraft, `inference/callsign.py` finds the aircraft a transcript names among the callsigns on ADS-B at that time. For each candidate, its airline telephony must appear in the text, followed by its flight number. The number may be written as numerals, digit by digit, or in group form ("twenty four thirty seven"), and it must end on a token boundary, so "United 20367" never matches UAL2. The match also guesses the speaker: a controller opens with the callsign; a pilot ends a readback with it, or names the facility first. It covers airlines with a telephony entry and favors precision over recall.

Words are normalized one at a time, so each word's confidence follows every token it becomes: case and punctuation are dropped, and numerals are spelled as training spoke them (`5,000` → *five thousand*, `270` → *two seven zero*, `FL350` → *flight level three five zero*, `nine` → *niner*). The report gives the share of tokens outside the training vocabulary. Above a few percent, the recognizer's style differs from the one the channel was fitted to (an ATC-tuned Whisper, `jacktol/whisper-large-v3-finetuned-for-ATC`), and the semantic model sees those words as unknown.

## ADS-B

`inference/adsb.py` reads both sources into OpenSky state vectors in SI units, which is what the window builder takes:

| source | files | notes |
|---|---|---|
| OpenSky | parquet or CSV, either with the columns `collect/opensky.py` writes or with pyopensky's raw history columns (`lat`, `lon`, `baroaltitude`, `heading`, `vertrate`, `lastposupdate`, ...) | `--source opensky` |
| adsb.lol | readsb / tar1090 trace files, the format of adsb.lol's daily `globe_history` archives (`traces/<xx>/trace_full_<icao>.json`, gzip with or without `.gz`) | `--source readsb`; feet, knots and ft/min are converted; rows flagged stale are dropped; on-ground rows are kept but marked |

`--source auto` (default) picks readsb for `.json` files and OpenSky otherwise. Globs are expanded, so `--adsb "traces/**/*.json"` works. To use another source, produce a table with these columns:

```
icao24, callsign, time_position (unix s), latitude, longitude, baro_altitude (m),
geo_altitude (m), velocity (m/s), true_track (deg), vertical_rate (m/s), on_ground
```

Pressure altitude (`baro_altitude`) is what the models use, because it is what ATC assigns. `--altitude geo` switches to GNSS height, but the models were trained on pressure altitude.

Each window needs the aircraft airborne above 50 kt, with no raw gap over 4 s in the 40 s before t0 or the 64 s after it. Rejected transmissions are counted by reason, never patched: `no_addressee`, `no_track`, `history_short`, `history_gap`, `history_stale`, `future_short`, `future_gap`, `slow`. Receiver coverage is the usual cause.

## Evaluate

```bash
uv run python -m atc_forecast.evaluation.real_v2 \
    --transmissions transmissions.jsonl --adsb "adsb/*.parquet" \
    --sbm runs/v2/sbm_jacktol_ho_s0.pt --ctp runs/v2/ctp_basis_atoms_zfix_s0.pt \
    --hist-only runs/v2/ctp_basis_none_s0.pt --scene-radius-nm 40 \
    --out results/real.json --per-message results/real_messages.jsonl [--wandb]
```

Every condition runs on the same windows with the same seeds:

| condition | |
|---|---|
| `composed` | the main model: the SBM reads the message, 12 interpretations × 6 futures |
| `known` | the same, with "this message is for this aircraft" given, as the matching asserts |
| `mode` | the SBM's single most likely interpretation, 72 futures under it |
| `hist_only` | the history-only predictor (`--hist-only`): `composed − hist_only` is what the radio adds |
| `scene` | with `--scene-radius-nm R`: every airborne aircraft within R nm (up to `--max-aircraft`, 16) also hears the message; they compete for it, and the addressee's forecast takes only its share |

`results/real.json` holds:
- counts and rejection reasons;
- the share of unknown tokens;
- the semantic model's mean split between "for me, motion", "for me, no motion" and "for someone else";
- per condition: energy score, ADE/FDE (horizontal m, vertical ft), min-ADE₆, coverage and outer-decile rank share;
- PIT and rank histograms (10 bins, each 0.1 when calibrated);
- with `--group`: the same per group;
- with a scene: how often the addressee ranks first.

`--per-message` writes one line per transmission: the top three interpretations with probabilities, the interpretations actually drawn, and every condition's scores. That file is where to start reading failures.

For batches of thousands of transmissions, use a GPU. On an M4 Pro, 300 transmissions with all five conditions take about 9 min, most of it in the scene condition (14 aircraft per scene on average).

## Predict from Python

```python
from atc_forecast.inference import Forecaster, load_states

fc = Forecaster.load("runs/v2/sbm_jacktol_ho_s0.pt", "runs/v2/ctp_basis_atoms_zfix_s0.pt",
                     hist_only="runs/v2/ctp_basis_none_s0.pt")
states = load_states("traces/**/*.json", source="readsb")
out = fc.forecast(states, icao24="a1b2c3", t0=1767225612.4, words=words)
```

`forecast` returns:
- `readings`: the beam's top interpretations as `(text, probability, atoms)`;
- `status`: the semantic model's split over who the message is for;
- `interpretations`: the 12 drawn readings;
- `velocities` [72, 60, 3]: track frame, m/s;
- `lat`, `lon`, `alt_ft` [72, 60];
- the token and unknown counts, and the callsign used.

By default it needs only the history (`future=False`), so it also runs live. For batches, the lower-level calls are `windows(states, requests)`, `batch(windows, rows, messages)`, `compose(b)`, `readings(b)`, `status(b)`, `history_only(b)` and `to_geo(...)`. `evaluation/real_v2.py` shows them in use.

The aircraft's own callsign comes from its ADS-B broadcast at t0. The semantic model compares it with the callsign heard in the message, so the broadcast must be the flight's current one: airframes fly several flight numbers a day.

## What a matched set measures

A transmission matched to an aircraft by callsign and time is a valid end-to-end test of `p(Y | O, X)`. The recorded 60 s is the real outcome of whatever was said, so no instruction label is needed: the energy score compares the 72 futures with what the aircraft flew. This differs from v1, whose instruction decoder needed each transcript paired with the instruction it contains.

What it does not separate:

- **Instructions acted on later than 60 s, or not at all** (readbacks, "expect" clauses, pilot discretion) count as forecast error, as they would in operation. A `group` field for clearance type or readback shows how much.
- **Messages that are not instructions** (frequency changes, traffic information) belong in the set too. The model should forecast "no change" for them, and the `hist_only` comparison shows whether it does.
- **Semantic accuracy** (did the SBM read the instruction right) needs a human label per transmission. Without one, `readings` in the per-message file are the place to judge it.
- **Which aircraft the message was for**, in the scene condition, is taken from the matching. A wrong match shows up as a low `p_addressee` with the true aircraft elsewhere.

On the synthetic test set the main model scores 1,675 m on own-maneuver messages, against 3,202 for the history-only model. On real speech the gap shrinks by however much the recognizer's real errors exceed the simulated channel; the unknown-token share and `status_mean` are the first signs of it.

## Checks behind the interface

- **Same windows as training.** 300 test messages were rebuilt from raw OpenSky files through `Forecaster.windows`. History, context, centering, future, aircraft type, tokens and confidences all equal the training store exactly (largest difference 3e-8, the 1/255 confidence quantization).
- **Same scores as the test evaluation.** On the 41 of those messages that the stored test evaluation also scored, with the same seeds, the composed energy score matches `compose_v2` to 0.00 m.
- **Tests.** `tests/test_inference_v2.py` covers:
  - word normalization keeping its confidences;
  - the history-only windows equal the full ones;
  - readsb unit conversion and stale rows;
  - the callsign at t0;
  - `Forecaster.forecast`;
  - the evaluation command line, run end to end on a synthetic track with tiny random models.

## Known differences from the training data

- **Callsigns in the training labels.** About a fifth of training windows carry a callsign from earlier in the airframe's day, not the flight's current one: the window builder labels a leg with its first callsign. The synthetic messages use the same label, so training is self-consistent. On real data, the interface uses the broadcast callsign, which is correct.
- **Recognizer style.** Training transcripts are spoken-form, from a channel fitted on an ATC-tuned Whisper. Other recognizers write numerals, abbreviations or casing differently. Normalization covers the common cases; watch the unknown-token share.
- **Airspace.** Training ADS-B is five days of the San Francisco Bay Area. Inputs use fixed physical scales, not dataset statistics, so other airspaces need no re-normalization. Different procedures and traffic mixes can still shift the error.
