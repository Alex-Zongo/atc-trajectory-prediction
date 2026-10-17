# Aircraft trajectory prediction under ATC instruction uncertainty

Predicting where an aircraft will be in the next minute when the instruction it was given is known only through an imperfect speech transcript.

## Why trajectory prediction

Air traffic control (ATC) depends on knowing where aircraft are going: conflict detection, arrival sequencing and separation assurance all rest on a prediction of each aircraft's near-term trajectory. 
Over a horizon of a minute or two, that trajectory can be driven by the controller's instructions — *turn left heading 270*, *descend and maintain 5,000*, *reduce speed to 210 knots*. An instruction, here, is the strongest single cue to what an aircraft is about to do, often well before the change shows up on surveillance.

## What a predictor has to work with

- **Surveillance.** ADS-B reports give the aircraft's past positions, altitude, speed and track. They show a maneuver only once it has started.
- **Voice instructions.** Controllers and pilots talk over the radio, so instructions are not available as structured data. A ground system learns them through **automatic speech recognition (ASR)**, which is itself a learned model and therefore error-prone: it confuses digits (*five* and *niner*), drops words, swaps *climb* and *descend*, and can output a fluent instruction that was never said.

A predictor that takes the transcript at face value inherits those errors without knowing it. This project keeps track of how uncertain the instruction is and carries that uncertainty into the predicted trajectory.

## Approach: two stages

```
radio voice ─► ASR ─► transcript ─►  Stage 1: instruction decoder  ─► possible instructions
                                                                         │ (with probabilities)
past trajectory + aircraft type ───────────────────────────────────────┐ ▼
                                     Stage 2: trajectory predictor  ─► possible trajectories
```

- **Stage 1** reads a noisy transcript and returns a probability distribution over what the instruction could have been: which heading, altitude and speed changes, in which direction, to which values.
- **Stage 2** takes one instruction, the aircraft's recent trajectory and its type, and returns a distribution of trajectories for the next 60 seconds. Keeping the stages separate is deliberate:

- **Each source of uncertainty can be studied on its own.** Stage 1 isolates what the transcription adds: how sure can we be about what was said? Stage 2 isolates what remains even when the instruction is transcribed perfectly: pilot response time, turn and climb rates, wind and aircraft performance all spread the trajectory. Feeding Stage 2 the true instruction measures that part alone.
- **Each stage can be replaced independently.** A different recognizer, a different instruction decoder or a different trajectory model can be swapped in and evaluated without retraining the rest.

Combined, the question becomes: **is it better to predict from the full set of plausible instructions than from a single best guess, and what does it cost?**

## Data: real trajectories, reconstructed instructions

The ideal dataset would pair every radio transmission with the instruction it contains and
the trajectory the aircraft then flew. No such public corpus exists at scale, and building
one from recordings is harder than it looks:

- **Alignment.** Matching recorded audio to aircraft by callsign and time mostly picks the
  wrong transmission. In one attempt, the matched transmission mentioned the heading the
  aircraft went on to fly in 0.2% of cases and the altitude in 1.3%, no better than
  chance: a pilot hears many transmissions (frequency changes, traffic, readbacks), and a
  maneuver can follow an instruction given minutes earlier or a published procedure.
- **Ground truth.** Measuring how transcription errors propagate requires knowing what was actually said for every transcript. Real recognizer output only gives what was heard; the rest needs hand annotation, and annotated ATC speech amounts to hours (the recognizer simulation here is fitted on one hour of ATCO2).
- **Scale and control.** Five days of Bay Area ADS-B give 174,046 maneuvers in 18,640 flight legs. Reconstructing their instructions uses all of them, and simulating the recognizer makes the transcription error rate a setting that can be varied.

So the data is real where it can be and reconstructed where it must be:

| part | source |
|---|---|
| past and future trajectories | real ADS-B (OpenSky) |
| instruction | inferred from each maneuver, written in FAA phraseology |
| transcript | the instruction passed through a simulation of a real ATC recognizer |

The reconstruction has known limits. It treats every maneuver as instructed, although pilot-initiated maneuvers, VFR flights (squawk is unknown for 67% of maneuvers) and published procedures such as SIDs, STARs and approaches look the same in ADS-B. It takes the value the aircraft settles on as the assigned value, places the instruction at the start of the maneuver with no pilot reaction time, and uses standard phraseology, which is more uniform than real radio speech. Testing on real speech needs a set of real transmissions, each labelled with the instruction it contains.

## The main experiment

Every held-out case is predicted four ways:

| condition (`code name`) | instruction given to Stage 2 |
|---|---|
| true instruction (`oracle`) | the instruction actually issued; the upper bound |
| no instruction (`none`) | none; the past trajectory alone |
| single best guess (`point`) | Stage 1's most likely instruction |
| full distribution (`belief`) | several instructions sampled from Stage 1 |

Average distance error alone cannot settle this comparison: it favours a confident prediction whether or not that confidence is justified. Predictions are therefore also scored on whether their spread is honest:

- **energy score** — a single score for the whole set of predicted trajectories, lower is
  better; it rewards being close *and* appropriately spread.
- **endpoint log-density** — how plausible the true final position is under the prediction, higher is better.
- **80% coverage** — how often the true position falls inside the 80% prediction region; a well-calibrated predictor scores about 0.80.

Current result on 3,149 held-out cases, with transcripts from a simulated recognizer:

| condition | mean error (nm) | 80% coverage | energy score | endpoint log-density |
|---|---|---|---|---|
| true instruction | 0.107 | 0.83 | 0.067 | 3.71 |
| no instruction | 0.200 | 0.82 | 0.124 | 1.86 |
| single best guess | 0.160 | **0.60** | 0.126 | **−3.95** |
| full distribution | 0.174 | 0.77 | **0.110** | 1.94 |

The single best guess has the lower average error but is overconfident: it covers the truth only 60% of the time and scores no better than ignoring the instruction. Keeping the full distribution gives the best energy score of the realistic conditions, at a small cost in average error.

## Install

Managed with [uv](https://docs.astral.sh/uv/); `uv.lock` pins every dependency.

```bash
uv sync --all-extras        # create .venv from the lock
uv run pytest               # 10 tests, no data or network needed
```

Prefix any command below with `uv run`. Plain pip also works:
`python -m venv .venv && .venv/bin/pip install -e ".[all]"`. Python 3.11+ (3.12 pinned);
PyTorch uses CUDA, Apple MPS or CPU automatically. Optional extras: `collect` (OpenSky),
`asr` (transcription), `app` (viewer), `track` (wandb), `dev` (tests).

## Get the trained models and data

Model weights and the prepared dataset are not in git. They are published with the
repository's `data-v1` release:

```bash
gh auth login                                 # once; needs access to this repository
python scripts/artifacts.py fetch             # models + prepared dataset (~420 MB)
```

Downloads are checked against `SHA256SUMS`. Without the GitHub CLI, download the release
assets by hand and run `fetch --from <folder>`. After retraining, the owner updates the
release with `python scripts/artifacts.py publish`.

| path | content |
|---|---|
| `runs/stage1.pt`, `runs/stage2.pt` | trained instruction decoder and trajectory predictor |
| `runs/lambda.json` | Stage 2 spread calibration |
| `data/dataset/meta.parquet` | one row per maneuver and its instruction |
| `data/dataset/meta_grounded.parquet` | the same, plus the aircraft state when the instruction was issued |
| `data/windows_raw/` | prediction cases: past trajectory, instruction, recorded future |
| `data/adsb/adsb_<day>.parquet` | raw ADS-B, needed only to rebuild the dataset; not redistributed, see [step 1](#1-collect-ads-b) |

## Quick start

With the models and data fetched:

```bash
# the main experiment (results/compose.csv)
python -m atc_forecast.evaluation.compose --stage1 runs/stage1.pt --stage2 runs/stage2.pt \
    --data data/windows_raw --meta data/dataset/meta_grounded.parquet \
    --lambda-json runs/lambda.json --out results/compose.csv

# interactive viewer
streamlit run src/atc_forecast/app/viewer.py -- --stage1 runs/stage1.pt \
    --stage2 runs/stage2.pt --data data/windows_raw \
    --meta data/dataset/meta_grounded.parquet --lambda-json runs/lambda.json
```

The viewer shows, for one case at a time, the transcript, the instructions Stage 1 thinks
were given, and the predicted trajectories under each condition.

## Rebuild from scratch

### 1. Collect ADS-B

```bash
python -m atc_forecast.collect.opensky 2026-01-15 2026-01-16 --out-dir data/adsb
```

Needs OpenSky historical-data access (`OPENSKY_USERNAME`, `OPENSKY_PASSWORD`). Days already downloaded are skipped. The default area is the San Francisco Bay Area; change `BAY_AREA_BBOX` in `collect/opensky.py` for another airspace.

### 2. Build the dataset

```bash
python scripts/build_dataset.py --adsb "data/adsb/adsb_*.parquet" --out data/dataset
```

Real ADS-B does not come with the instructions that were issued, so they are
reconstructed. Each track is cut into **maneuvers** wherever its heading, altitude or speed starts to change, and each maneuver gets the instruction that would have produced it, written in FAA phraseology from the values the maneuver ends at (e.g. a left turn ending on 183° becomes *turn left heading one eight five*). Segments of steady flight get no instruction. Each maneuver start then becomes one prediction case: 30 s of past trajectory, the instruction, and the next 60 s.

Three rules keep training and evaluation honest:

1. **Split by flight.** All cases from one flight leg go to either training or validation,
   never both.
2. **Past ends at the instruction.** The past stops before the maneuver begins, so the
   instruction cannot be read off a maneuver already under way.
3. **Future ends at the next instruction.** Each case's future reflects exactly one
   instruction.

### 3. Simulate the speech recognizer

Because the instructions are reconstructed from the tracks, they are exact. To study transcription errors, the dataset passes each instruction through a **simulated recognizer** (the *observation channel*): the instruction is spoken out, then corrupted at rates measured on a real ATC-tuned recognizer (`channel/profiles/jacktol.json`, fitted on the ATCO2 corpus). This turns the noise level into something that can be set and varied.

To fit the simulation to another recognizer, from a JSONL of
`{reference_text, hypothesis_text, confidence_proxy}` per utterance:

```bash
python -m atc_forecast.channel.fit records.jsonl --name myasr --write
```

Matching error rates is guaranteed by fitting, so `fit` also checks the assumptions it
cannot guarantee. On ATCO2:

| assumption | measured |
|---|---|
| errors in different parts of an instruction are independent | holds (0.69–0.92× against 1.0× assumed) |
| errors do not depend on utterance length | holds (correlation −0.035) |
| recognizer confidence predicts errors | **it does** (correlation −0.658), so `fit` samples confidence from its measured distribution rather than using one fixed value |

The shipped `jacktol` profile predates that change and still emits a fixed confidence
(0.82), so the current models have not learned to use confidence.

### 4. Train

Stage 2 first, because Stage 1 reads the aircraft state from the Stage 2 cases:

```bash
python -m atc_forecast.training.stage2 --data data/windows_raw --out runs/stage2.pt --epochs 30
python -m atc_forecast.training.stage1 --meta data/dataset/meta_grounded.parquet \
    --window-dir data/windows_raw --state-grounded --out runs/stage1.pt --epochs 14
```

During training Stage 2 reports how much the instruction improves prediction
(`clearance_gain_nm`) and how much knowing the aircraft type does (`aircraft_gain_nm`).

### 5. Calibrate

Both stages are adjusted after training so their spread matches reality, fitted on half of
the flights and checked on the other half:

```bash
python -m atc_forecast.evaluation.stage2 --model runs/stage2.pt --data data/windows_raw \
    --out runs/lambda.json --n-bins 6
python -m atc_forecast.evaluation.stage1 --model runs/stage1.pt \
    --meta data/dataset/meta_grounded.parquet --fit-temperature
```

Stage 1 gets a **sampling temperature** (how widely it spreads over possible
instructions). Stage 2 gets a **spread factor per look-ahead time**: one factor for the whole minute left predictions overconfident in the first seconds and too wide at the end.

### 6. Evaluate

Run the main experiment as in the Quick start. For the full distribution, each sampled instruction gets its own set of trajectories, calibrated before they are pooled, so the result separates transcription uncertainty from execution uncertainty.

### Optional: real audio

```bash
python -m atc_forecast.asr.transcribe audio/ --out transcripts.jsonl \
    --model jlvdoorn/whisper-large-v3-atco2-asr --references refs.json
```

Not needed for the pipeline; useful for fitting the simulated recognizer to your own ASR.
Use an ATC-tuned model; general-purpose ones are much worse on digits and callsigns.

Recordings matched to flights by callsign and time are not a valid Stage 1 test set; see [Data](#data-real-trajectories-reconstructed-instructions).

## Repository layout

```
src/atc_forecast/
  collect/      ADS-B download (OpenSky), aircraft type database
  data/         maneuver detection, phraseology, flight legs, prediction cases, tokenizer
  channel/      the simulated recognizer and its fitting
  models/       stage1.py (instruction decoder), stage2.py (trajectory predictor)
  training/     training loops for both stages
  evaluation/   scores, calibration, the main experiment (compose.py)
  asr/          transcription of real audio
  app/          viewer
scripts/        build_dataset.py, artifacts.py
tests/
```

In the code an instruction is called a **clearance** (e.g. `ClearancePosterior`,
`clearance_gain_nm`), and the package is named `atc_forecast`.

## Design notes

Choices that matter, with the reason:

- **Stage 1 decodes an instruction one part at a time.** Presence, direction and value of the turn, then of the altitude change, then of the speed change; each part sees the parts before it. Sampled instructions are therefore always coherent, and Stage 1 can hold genuinely different readings at once, such as *left 270* versus *right 090*.
- **Headings are predicted over 72 bins of 5°**, not as one value with an error bar: a misheard *two zero zero* can plausibly be 290, 100 or 300, and only a binned distribution can represent that.
- **Stage 1 sees the aircraft's current state.** Altitude, speed and heading let it reject physically implausible readings, such as a speed far outside what the aircraft flies.
- **Stage 2 predicts speed, heading and vertical rate, not positions.** That is what an instruction commands; positions follow by integrating them. Early errors are weighted more heavily because they shift every later position.
- **Stage 2 is a flow-matching generative model.** It turns random noise into a set of plausible trajectories, so it can represent several different futures, not just an average path.
- **The no-instruction condition comes from the same model.** Stage 2 is trained with the instruction randomly hidden (10% of the time), which makes it its own past-only baseline. Changing that rate changes the baseline every gain is measured against.
- **Units are scaled by fixed aviation constants**, never by dataset statistics, so a trained model stays valid when data is added or another airspace is used.
- **VFR flights are flagged, not removed.** A VFR aircraft is not following ATC instructions, so its reconstructed instruction is fictitious; squawk is often missing, so the flag has an "unknown" level and filtering is left to the user.

## Citation

A paper describing this work is in preparation. Until it is published, please cite the software (GitHub's **Cite this repository** button gives the same entry from
`CITATION.cff`):

```bibtex
@software{zongo2026atc,
  author  = {Zongo, Alex and Petit, Mathurin and Brusset, Louis and Kam, Jordan and
             Torun, Emir and Ghazanfari, Mahyar and Wei, Peng and Bayen, Alexandre},
  title   = {Aircraft Trajectory Prediction under {ATC} Instruction Uncertainty},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Alex-Zongo/atc-trajectory-prediction},
  note    = {Software; paper in preparation}
}
```

If you use the ADS-B data, please also cite the OpenSky Network:

```bibtex
@inproceedings{schafer2014opensky,
  author    = {Sch{\"a}fer, Matthias and Strohmeier, Martin and Lenders, Vincent and
               Martinovic, Ivan and Wilhelm, Matthias},
  title     = {Bringing Up {OpenSky}: A Large-scale {ADS-B} Sensor Network for Research},
  booktitle = {Proceedings of the 13th International Symposium on Information
               Processing in Sensor Networks (IPSN)},
  year      = {2014},
  pages     = {83--94}
}
```

## Acknowledgements

This project was inspired by [Louis Brusset](https://github.com/LouisBrusset) and [Mathurin](https://github.com/matu1003)'s earlier work. Aircraft surveillance data comes from the [OpenSky Network](https://opensky-network.org); the simulated recognizer is fitted on the [ATCO2](https://www.atco2.org) corpus.

## License

MIT; see `LICENSE`.
