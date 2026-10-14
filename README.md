# Trajectory forecasting under ATC clearance uncertainty

Two-stage aircraft trajectory forecasting that keeps the uncertainty about *what the
aircraft was told to do* instead of committing to one guess:

```
stage 1   q(delta | u, kappa, s)    noisy transcript -> distribution over the clearance
stage 2   p(tau  | delta, c)        clearance + context -> distribution over trajectories
```

`u` is the recognized transcript, `kappa` the recognizer's confidence, `s` the aircraft's
state at clearance onset, `delta` a ten-field structured clearance, `c` the past trajectory
plus the airframe, and `tau` the future control sequence.

The question the design exists to answer: **does preserving uncertainty over what an
aircraft was instructed to do produce more reliable forecasts than committing to one
inferred clearance, and at what cost?** The `evaluation.compose` experiment answers it
directly by scoring four conditions on identical held-out windows: the true clearance
(ceiling), no clearance (past-only baseline), Stage 1's single MAP clearance (`point`), and
k clearances sampled from Stage 1's posterior (`belief`).

Point accuracy cannot settle that comparison. A mean-over-samples error penalises
dispersion and an ensemble-mean error penalises multimodality, so both favour a confident
forecast whether or not it is right. The reported metrics are therefore proper scoring
rules (energy score), density (endpoint KDE log-density) and calibration (coverage, PIT),
with point error reported alongside as the cost.

## Install

```bash
pip install -e ".[dev]"            # core + tests
pip install -e ".[all]"            # adds collection, ASR, viewer, tracking
pytest                             # 10 tests, no data or network needed
```

Python 3.11+. Torch runs on CUDA, Apple MPS or CPU, picked automatically.

## Pipeline

### 1. Collect ADS-B

```bash
python -m atc_forecast.collect.opensky 2026-01-15 2026-01-16 --out-dir data/adsb
```

Historical OpenSky access is a separate grant from a normal account. Credentials come from
`OPENSKY_USERNAME` / `OPENSKY_PASSWORD` or `pyopensky`'s settings file. Queries are hourly
and sequential, and whole days already downloaded are skipped, so an interrupted run can be
relaunched. Replace `BAY_AREA_BBOX` with your own airspace.

### 2. Build the dataset

```bash
python scripts/build_dataset.py --adsb "data/adsb/adsb_*.parquet" --out data/dataset
```

Three things happen. Tracks are segmented into **maneuver segments** by change points on the
three channels a controller commands (heading, altitude, speed); each maneuver onset gets a
synthesised FAA-phraseology clearance from its segment's exit values; segments are grouped
into **legs** and sliced into Stage-2 windows.

Three leakage cuts are baked into the windows:

1. **split by leg** — a leg is atomic, so no trajectory point lands on both sides of a split
2. **past strictly before onset** — the past is `[onset - P, onset)`, so the clearance
   cannot be read off the maneuver already in progress
3. **future truncated at the next onset** — each window's future is explained by exactly
   one clearance, since a later clearance would be an unconditioned confound

Outputs: `segments.parquet`, `legs.parquet`, `meta.parquet` (the maneuver subset, which is
Stage 1's training set), `meta_grounded.parquet` (with the onset state joined on) and
`windows_raw/windows.npz`.

### 3. The observation channel

The clearances are a deterministic function of the trajectory, so the dataset has **zero
transcript uncertainty by construction**. The uncertainty this work is about is the
uncertainty of ATC *communication*, so it is injected by a channel — which makes the noise
level a controlled independent variable rather than an accident of the corpus.

A channel is a **per-field rate model fitted to a real ATC-finetuned recognizer**, applied
to a spoken rendering of the clearance. `profiles/jacktol.json` ships fitted to one such
recognizer on ATCO2. Fit your own:

```bash
python -m atc_forecast.channel.fit records.jsonl --name myasr --write
```

from a JSONL of `{reference_text, hypothesis_text, confidence_proxy}` per utterance.

**Fitting marginal rates is not validation.** The rates match the recognizer by
construction, so measuring them again proves nothing. What fitting cannot guarantee is the
structure layered on top, so `fit` also reports it and you should report it with any result:

| assumption | how it is tested | measured on ATCO2 |
|---|---|---|
| errors independent across fields | P(B wrong \| A wrong) over P(B wrong \| A right) | 0.69x–0.92x against the 1.0x assumed, so it holds |
| no dependence on utterance length | corr(WER, length) | −0.035, so it holds |
| confidence is informative | corr(confidence, WER) | **−0.658**, and the bottom confidence quartile is 5.6x as error-prone as the top |

The third one is a real gap. A channel emitting a single fixed confidence hands Stage 1 an
input carrying no per-utterance information, so the model cannot learn to widen its
posterior when the recognizer is unsure. `fit` emits an **empirical confidence
distribution** by default, which closes it; `--fixed-confidence` reproduces the older
behaviour.

### 4. Train

Stage 2 first, since Stage 1's state grounding reads the onset state out of the windows:

```bash
python -m atc_forecast.training.stage2 --data data/windows_raw --out runs/stage2.pt --epochs 30
python -m atc_forecast.training.stage1 --meta data/dataset/meta_grounded.parquet \
    --window-dir data/windows_raw --state-grounded --out runs/stage1.pt --epochs 14
```

Stage 2 logs two ablations at every evaluation, so their value is visible *during* training:
`clearance_gain_nm` (ADE without the clearance minus ADE with it) and `aircraft_gain_nm`,
which doubles as the graceful-degradation cost when the airframe is unknown. The clearance
gain is also reported stratified by whether the clearance carries a clause, because the
aggregate is diluted by steady windows where the past trivially determines the future.

Stage 1 selects its checkpoint on `energy` by default, not `val_nll`. `val_nll` is
teacher-forced but the model is deployed free-running, and scheduled sampling deliberately
makes the teacher-forced objective harder, so selecting on `val_nll` would prefer exactly
the runs that did the least of what was turned on.

### 5. Calibrate

Both stages are calibrated post hoc against proper scoring rules, fitting on one half of
the legs and reporting on the other, so a correction is never graded on its own data.

```bash
python -m atc_forecast.evaluation.stage2 --model runs/stage2.pt --data data/windows_raw \
    --out runs/lambda.json --n-bins 6
python -m atc_forecast.evaluation.stage1 --model runs/stage1.pt \
    --meta data/dataset/meta_grounded.parquet --fit-temperature
```

Stage 1 fits a **sampling temperature** minimising the energy score. Because that score is
proper, its minimiser is the calibrated spread and cannot be gamed by collapsing to a point.

Stage 2 fits a **spread inflation lambda, per lead time**. A scalar lambda can only rescale
the whole path about its mean, so it cannot fix a forecast whose dispersion has the wrong
*shape* over the horizon — and this one does: with a fitted scalar applied, coverage ran
around 0.60 in the first ten seconds and 0.94 in the last ten, badly over-confident early
and over-dispersed late, averaging out to something that looked fine.

### 6. The experiment

```bash
python -m atc_forecast.evaluation.compose --stage1 runs/stage1.pt --stage2 runs/stage2.pt \
    --data data/windows_raw --meta data/dataset/meta_grounded.parquet \
    --lambda-json runs/lambda.json --out results/compose.csv
```

Composition is done **per clearance, not pooled**: for each sampled clearance, `exec_k`
execution trajectories are drawn and *that* clearance's ensemble is inflated before the
clouds are pooled. The pooled cloud is then communication spread plus calibrated execution
spread. Inflating an already-pooled, communication-dominated cloud would rescale the wrong
thing.

### 7. Look at it

```bash
streamlit run src/atc_forecast/app/viewer.py -- --stage1 runs/stage1.pt \
    --stage2 runs/stage2.pt --data data/windows_raw \
    --meta data/dataset/meta.parquet --lambda-json runs/lambda.json
```

Pick a window and see the transcript the channel produced, the clearance posterior Stage 1
inferred from it, and the trajectory distribution under each condition, with metrics.

### Optional: real audio

```bash
python -m atc_forecast.asr.transcribe audio/ --out transcripts.jsonl \
    --model jlvdoorn/whisper-large-v3-atco2-asr --references refs.json
```

The pipeline does not need audio. This is for fitting a channel to your own recognizer and
for running Stage 1 on real recognizer output. Use an ATC-finetuned model; a
general-purpose one is far worse on exactly the tokens that matter, spoken digit strings and
callsigns.

**One caveat before you try to compare synthetic against real performance.** A corpus of
real utterances matched to trajectories by callsign and time does *not* give aligned
`(transcript, clearance)` pairs. Measured on one such set, the matched utterance mentioned
the labelled heading in 0.2% of windows and the labelled altitude in 1.3%, both at chance:
the clearance label comes from the trajectory while the utterance is usually a *different*
instruction. Such a set cannot serve as a real-transcript test set for Stage 1. Measuring
real-transcript performance needs utterances whose clearance label comes from the **same**
utterance.

## Layout

```
src/atc_forecast/
  units.py            physical constants and normalisation scales
  collect/            opensky.py (historical ADS-B), aircraft.py (wake/engine metadata)
  data/               segmentation, phraseology, legs, windows, lexicon, text
  channel/            model.py (the channel), fit.py (fit + fidelity), profiles/*.json
  models/             backbone.py (shared), stage1.py (posterior), stage2.py (flow)
  training/           stage1.py, stage2.py
  evaluation/         scores.py, stage1.py, stage2.py, compose.py
  asr/                transcribe.py
  app/                viewer.py
scripts/build_dataset.py
tests/
```

Normalisation uses **fixed physical constants**, never dataset statistics. A z-score on the
sample mean moves when the data moves, silently invalidating previously trained models, and
it does not transfer to another airspace. The divisors in `units.py` are properties of civil
aviation, so they are stable across datasets, regions and years.

## Design notes

Choices that are load-bearing, and would look arbitrary without the reason:

**Stage 1 is autoregressive, not per-field independent.** A factorized model draws each
field independently, so a sampled clearance can be incoherent (mishearing the altitude but
not its presence) and the posterior can never be jointly multimodal, which is exactly what
"turn left 270 or turn right 090" requires. Each conditional here reads a GRU state that has
consumed every previously decoded field, so the fields co-vary, and the likelihood is exact,
which is what makes the posterior directly calibratable.

**Heading is categorical over 72 bins.** Digit-by-digit corruption of a three-digit heading
has a multimodal posterior (200 becoming 290, 100 or 300) that a unimodal circular
distribution cannot represent, which is why its concentration parameter never calibrated.

**Stage 2 predicts controls, not positions.** A clearance commands a rate or a target, not a
coordinate. Positions are dead-reckoned from the controls for the distance metrics. Because
the loss is on controls and every metric is on their running sum, per-step weights scale
with how many later positions each step still affects; weighting steps equally
under-penalises early error by up to a factor of the horizon.

**The future attention mask is used in training only.** `fut_mask` is built from the next
clearance's onset and post-hoc coverage, so passing it at sampling time tells the forecaster
how long the maneuver lasted, which is not knowable at prediction time. Its legitimate job
was keeping valid queries from attending to fabricated interpolated values; that need is
removed instead, by overwriting invalid steps with the source sample so they carry no target
information.

**`last_valid_index`, never `mask.sum(1) - 1`.** The latter is correct only when the valid
steps form an unbroken run from step 0. Windows with an internal coverage hole do occur, and
on those the count-based expression is short by a mean of 18 steps and sometimes indexes a
step that is itself invalid, scoring the forecast against a fabricated position.

**The CFG dropout rate is not an ordinary hyperparameter.** The unconditional branch *is*
the no-instruction baseline that every reported clearance gain is measured against. Too low
and the gain is flattered; too high and it is understated. Change it as a deliberate
ablation, never alongside anything else.

**Steady segments are kept with an empty clearance.** Each leg is then a complete ordered
trajectory with clearances at maneuver onsets and nothing between, which is the structure a
comms-conditioned forecaster consumes. It also teaches Stage 1 that most radio traffic
carries no instruction.

**`is_vfr` is flagged, not filtered.** A VFR aircraft is self-navigating, so a synthesised
clearance for it is fiction. Squawk is often missing, so "unknown" is an honest third level
and the decision is left to the user.

## License

See `LICENSE`.
