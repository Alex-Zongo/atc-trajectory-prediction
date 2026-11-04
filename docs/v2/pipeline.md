# v2 pipeline: data, training, evaluation

How to rebuild every v2 result from raw ADS-B, and where each piece lives. To run the trained models on your own data instead, see [real-data.md](real-data.md); for the results themselves, [results.md](results.md).

- [Setup](#setup)
- [Code map](#code-map)
- [Reproduce the main model](#reproduce-the-main-model)
- [1. Data](#1-data)
- [2. Training](#2-training)
- [3. Weights & Biases](#3-weights--biases)
- [4. Evaluation](#4-evaluation)
- [5. Diagnosis](#5-diagnosis)
- [6. Tests](#6-tests)
- [The whole plan in one script](#the-whole-plan-in-one-script)
- [Hyperparameters that matter](#hyperparameters-that-matter)

## Setup

```bash
uv sync --all-extras          # or --extra track --extra dev for W&B and tests only
wandb login                   # once, if you log online
```

Device selection is automatic (CUDA, then Apple MPS, then CPU); override it with `--device`. With `scripts/artifacts.py fetch` (see the [README](../../README.md#get-the-trained-models)), `data/v2/` and `runs/v2/` are already in place, and every evaluation below runs without retraining.

## Code map

| module | role |
|---|---|
| `data/kalman.py` | batched constant-acceleration Kalman filter and RTS smoother on irregular samples |
| `data/resample.py` | clean a track, split at t0, gap rules, resample each side at 1 Hz; `windows_from_adsb` is the entry point for any ADS-B, `history_windows` the live (history-only) variant |
| `data/frames.py` | track frame, features, centering velocity, residual targets, per-step scale, PCA basis |
| `data/atoms.py` | atom vocabularies (status 3, lateral 146, vertical 106, speed 112), labels to atoms, relative features |
| `data/windows_v2.py` | the dataset build |
| `data/messages.py` | precomputed messages: four variants per window, through the fitted recognizer channel |
| `data/primitives.py` | maneuver primitives and physical kinds (design A) |
| `channel/model.py`, `channel/profiles/` | the recognizer channel and its fitted rates per recognizer |
| `channel/word_conf.py` | per-word confidence distributions fitted to real recognizers |
| `models/encoders.py` | history encoder, instruction tokens, text encoder |
| `models/sbm.py` | SBM: any-order, independent or mixture heads; sampling; certified beam search |
| `models/ctp.py` | CTP: basis or per-step output; atoms, none or text conditioning |
| `models/baselines.py` | MDN, MultiPath-style anchors, kinematic rule |
| `models/prim.py` | design A: kind head, flow on primitives, kinematic rule |
| `training/data_v2.py` | batches from memory-mapped arrays; the interpretation policy |
| `training/sbm_v2.py`, `training/traj_v2.py` | training loops for the SBM and for the CTP and its alternatives |
| `training/tracking.py` | optional W&B logging |
| `evaluation/metrics_v2.py` | ADE/FDE, min-ADE, energy, variogram, coverage, PIT, rank, spread-skill, raw-ADS-B error |
| `evaluation/compose_v2.py` | every condition on fixed test messages |
| `evaluation/addressee_v2.py` | who the message is for: resolution over the aircraft on frequency |
| `evaluation/real_v2.py` | the same conditions on real matched transmissions and ADS-B |
| `inference/` | ADS-B readers (OpenSky, readsb/adsb.lol), transcript normalization, `Forecaster` |
| `scripts/run_v2.sh` | the whole plan, by stage |
| `scripts/inspect_v2.py` | data figures and statistics |
| `scripts/diagnose_v2.py`, `scripts/altitude_noise_v2.py` | under-dispersion hypotheses H1–H5, SBM checks, altitude noise |
| `scripts/analyze_composition_v2.py` | composition by SBM confidence, right/wrong mode, paired intervals |
| `scripts/build_primitives_v2.py` | design A targets |
| `scripts/collect_results_v2.py` | every result into `results/v2/results.json` |
| `scripts/wandb_backfill.py` | push logs of runs trained without `--wandb` |
| `scripts/artifacts.py` | publish and fetch the release assets |

## Reproduce the main model

```bash
bash scripts/run_v2.sh data                     # windows, word confidence, messages (~12 min)
uv run python -m atc_forecast.training.sbm_v2 --profile jacktol --factorization anyorder \
    --out runs/v2/sbm_jacktol_anyorder_s0.pt --seed 0 --wandb
uv run python -m atc_forecast.training.traj_v2 --model ctp --output basis --condition atoms \
    --out runs/v2/ctp_basis_atoms_zfix_s0.pt --seed 0 --wandb
uv run python -m atc_forecast.training.traj_v2 --model ctp --output basis --condition none \
    --out runs/v2/ctp_basis_none_s0.pt --seed 0 --wandb
uv run python -m atc_forecast.training.traj_v2 --model anchors --condition atoms \
    --out runs/v2/anchors_atoms_zfix_s0.pt --seed 0 --wandb          # optional fast head
uv run python -m atc_forecast.evaluation.compose_v2 --sbm runs/v2/sbm_jacktol_anyorder_s0.pt \
    --ctp runs/v2/ctp_basis_atoms_zfix_s0.pt --hist-only runs/v2/ctp_basis_none_s0.pt \
    --direct runs/v2/ctp_basis_text_s0.pt --anchors runs/v2/anchors_atoms_zfix_s0.pt \
    --split test --n 5000 --k 12 --s 6 --nfe 12 \
    --per-message results/v2/eval_zfix_s0.npz --out results/v2/eval_zfix_s0.json --wandb
uv run python -m atc_forecast.evaluation.addressee_v2 --sbm runs/v2/sbm_jacktol_anyorder_s0.pt \
    --ctp runs/v2/ctp_basis_atoms_zfix_s0.pt --hist-only runs/v2/ctp_basis_none_s0.pt \
    --profile jacktol_conc --n-aircraft 8 --n 1000 --out results/v2/addressee_old_N8.json
uv run python scripts/collect_results_v2.py --runs runs/v2 --results results/v2 --out results/v2/results.json
```

The `--direct` model is the transcript-reading CTP (`--condition text`, [training](#2-training)). The addressee evaluation reads the concurrent-distractor messages (`--tag _conc`, [evaluation](#who-is-the-message-for)). The `--p-null 0.2 --p-maintain 0.2` defaults of `traj_v2` are the corrected "not for us" labels; checkpoints without `zfix` in their name predate them.

### The final SBM: callsign contrast, then handoff phrases

The SBM above is the *base* SBM. The paper's main results use the *final* SBM, built in two steps on top of the same trajectory models (results: [results.md](results.md#the-sbm-versions)):

```bash
# 1. contrast messages (own instruction, other callsign, labeled "other aircraft") and phrasing
#    heard on the recorded NorCal frequency; vocabulary grown from the base one, old ids kept
uv run python -m atc_forecast.data.messages --data data/v2 --profile jacktol --tag _cs --swap \
    --real-phrasing 0.15 --vocab-extend data/v2/messages_jacktol/vocab.json
uv run python -m atc_forecast.training.sbm_v2 --profile jacktol_cs --cs-match \
    --out runs/v2/sbm_jacktol_cs_s0.pt --seed 0 --resume --wandb            # ~1.7 h, seeds 0-2

# 2. generated handoffs (random facility and frequency), then a 10k-step fine-tune
uv run python -m atc_forecast.data.messages --data data/v2 --profile jacktol --tag _ho --swap \
    --real-phrasing 0.15 --handoff 0.5 --vocab-extend data/v2/messages_jacktol_cs/vocab.json
uv run python -m atc_forecast.training.sbm_v2 --profile jacktol_ho --cs-match \
    --init runs/v2/sbm_jacktol_cs_s0.pt --steps 10000 --lr 1e-4 --warmup 500 --eval-every 2000 \
    --out runs/v2/sbm_jacktol_ho_s0.pt --seed 0 --resume --wandb            # ~30 min

# checks: callsign probe, turn-direction probe, test conditions that depend on the SBM
uv run python scripts/diagnose_v2.py callsign --sbm runs/v2/sbm_jacktol_ho_s0.pt --profile jacktol_cs --n 600 --tag _ho_s0
uv run python scripts/diagnose_v2.py turn     --sbm runs/v2/sbm_jacktol_ho_s0.pt --tag _ho_s0
uv run python -m atc_forecast.evaluation.compose_v2 --sbm runs/v2/sbm_jacktol_ho_s0.pt \
    --ctp runs/v2/ctp_basis_atoms_zfix_s0.pt --anchors runs/v2/anchors_atoms_zfix_s0.pt --split test --n 5000 \
    --only composed mode anchors_composed --per-message results/v2/eval_ho_s0.npz --out results/v2/eval_ho_s0.json
```

`--cs-match` adds two deterministic features per transcript token: whether it belongs to the best in-order alignment of the aircraft's own callsign within the first eight words, and the share of callsign tokens that alignment matches. `--resume` continues from `<out>.resume.pt`, written at every validation, so a run survives a restart. Each checkpoint records its own addressee prior (`addressed_prior`, 0.7 with contrast messages), which `addressee_v2` and `real_v2` read.

## 1. Data

```bash
# trajectory windows: segmentation on pressure altitude, delayed t0, strict 1 Hz, leg split (~6.5 min)
uv run python -m atc_forecast.data.windows_v2 --adsb "data/adsb/adsb_*.parquet" --out data/v2

# word confidence fitted to recognizer benchmarks (needs data/asr, internal; optional)
uv run python -m atc_forecast.channel.word_conf --out data/v2/word_conf.json

# messages for one channel profile (~4 min on 8 workers); one folder per profile
uv run python -m atc_forecast.data.messages --data data/v2 --profile jacktol

# look at it: data/v2/figures/*.png and stats.json
uv run python scripts/inspect_v2.py --data data/v2
```

Raw ADS-B comes from OpenSky (`collect/opensky.py`, which needs historical access) as one parquet file per day. Any other source works through `inference/adsb.py` ([real-data.md](real-data.md#ads-b)).

Without `data/asr`, word confidences come from a parametric fallback with the same shape as the fitted curves, and the build says so.

Build flags:
- `--delays`: draws per maneuver (3);
- `--delay-lo` / `--delay-hi`: delay range in seconds (2 / 12);
- `--gap`: largest raw gap in seconds (4);
- `--altitude baro|geo`: altitude source (baro).

Output in `data/v2/`:
- `arrays/*.npy`: `hist`, `ctx`, `center`, `fut_vel`, and `raw_future` (validation and test only);
- `windows.parquet`: one row per window, with ids, split, kind, delay, labels, atoms, state and `next_onset_rel`;
- `basis.npz`: target scale, mean, components and coefficient std, all from the training split;
- `manifest.json`: counts, rejection reasons, build config;
- `messages_<profile>/`: tokens, confidences, own callsign, `messages.parquet`, vocabulary, summary.

## 2. Training

Every trajectory model gets the same budget: 60k steps at batch 256, about 40 min on an M4 Pro. The SBM takes 40k steps at batch 512, about 1.7 h. Checkpoints and `*.log.jsonl` go to `runs/v2/`.

```bash
# SBM (main: any-order heads, measured channel)
uv run python -m atc_forecast.training.sbm_v2 --profile jacktol --factorization anyorder \
    --out runs/v2/sbm_jacktol_anyorder_s0.pt --seed 0 --wandb --wandb-group core

# CTP (main: basis flow conditioned on atoms)
uv run python -m atc_forecast.training.traj_v2 --model ctp --output basis --condition atoms \
    --out runs/v2/ctp_basis_atoms_zfix_s0.pt --seed 0 --wandb --wandb-group core

# alternatives and ablations
uv run python -m atc_forecast.training.traj_v2 --model ctp --output basis --condition none  --out runs/v2/ctp_basis_none_s0.pt     # history only
uv run python -m atc_forecast.training.traj_v2 --model ctp --output steps --condition atoms --out runs/v2/ctp_steps_atoms_s0.pt    # per-step flow
uv run python -m atc_forecast.training.traj_v2 --model ctp --output basis --condition text  --out runs/v2/ctp_basis_text_s0.pt     # reads the transcript
uv run python -m atc_forecast.training.traj_v2 --model ctp --condition text --init-sbm runs/v2/sbm_jacktol_anyorder_s0.pt \
    --out runs/v2/ctp_basis_textinit_s0.pt
uv run python -m atc_forecast.training.traj_v2 --model mdn     --out runs/v2/mdn_atoms_s0.pt
uv run python -m atc_forecast.training.traj_v2 --model anchors --out runs/v2/anchors_atoms_zfix_s0.pt
uv run python -m atc_forecast.training.traj_v2 --basis sine    --out runs/v2/ctp_basis_atoms_sine_s0.pt
uv run python -m atc_forecast.training.traj_v2 --time-sampling uniform --out runs/v2/ctp_basis_atoms_unif_s0.pt
uv run python -m atc_forecast.training.sbm_v2 --factorization independent --out runs/v2/sbm_jacktol_independent_s0.pt
uv run python -m atc_forecast.training.sbm_v2 --factorization mixture     --out runs/v2/sbm_jacktol_mixture_s0.pt
uv run python -m atc_forecast.training.sbm_v2 --no-history                --out runs/v2/sbm_jacktol_nohist_s0.pt

# design A (explicit maneuvers): targets, then the kind-conditioned and primitive models
uv run python scripts/build_primitives_v2.py --clean --bank --out primitives_v3.npz
uv run python -m atc_forecast.training.traj_v2 --model kindctp --kinds joint --kind-readout --out runs/v2/kindctp_ro_atoms_s0.pt
uv run python -m atc_forecast.training.traj_v2 --model prim --kinds joint --prim-file primitives_v3.npz --out runs/v2/prim_v3_atoms_s0.pt
```

For a quick check that everything runs, add `--steps 1000 --eval-every 1000 --eval-n 500`.

## 3. Weights & Biases

Logging is off by default. Turn it on per run with `--wandb`, or for every run of a stage with `WANDB=1 bash scripts/run_v2.sh <stage>`. Evaluations log progress and running scores as they go, so a long evaluation is visible before it finishes.

- **Project and grouping:** `--wandb-project` (default `atc-v2`), `--wandb-group` (core, baselines, seeds, ...), `--run-name` (default: the checkpoint name).
- **Offline:** `WANDB_MODE=offline` logs locally; push later with `wandb sync <dir>`.
- **Runs started without `--wandb`:** `uv run python scripts/wandb_backfill.py runs/v2/*.log.jsonl --group core`.

| key | what it tells you |
|---|---|
| `train/loss`, `train/grad_norm`, `train/lr`, `train/ms_per_step` | optimization health |
| `train/ce_{status,lateral,vertical,speed}` (SBM) | which slot is hard; speed stays highest, since the channel drops half the speed values |
| `val/nll`, `val/acc_*`, `val/ece_*` (SBM) | selection metric; accuracy and calibration per slot |
| `final/T_*`, `final/beam_certified` (SBM) | fitted temperatures; share of exact joint modes |
| `val/loss` against `train/loss` (CTP) | the overfitting gap |
| `val/energy_h`, `val/energy_v`, `val/select` (CTP) | proper scores; `select` picks the checkpoint |
| `val/ade_h` against `val/minade6_h` | a gap means several modes are kept; no gap means mode collapse |
| `val/cov50`, `val/cov90`, `cal/coverage90`, `cal/spread_skill` | calibration overall and by lead time; a dip at short leads points to the noise floor |
| `progress/*`, `running/*` (evaluations) | messages done, running energy per condition |
| `eval/scores` table, `eval/coverage_*` | every condition × message variant |

## 4. Evaluation

```bash
uv run python -m atc_forecast.evaluation.compose_v2 \
    --sbm runs/v2/sbm_jacktol_anyorder_s0.pt --ctp runs/v2/ctp_basis_atoms_zfix_s0.pt \
    --hist-only runs/v2/ctp_basis_none_s0.pt --direct runs/v2/ctp_basis_text_s0.pt \
    --direct-init runs/v2/ctp_basis_textinit_s0.pt --anchors runs/v2/anchors_atoms_zfix_s0.pt \
    --split test --n 5000 --k 12 --s 6 --nfe 12 --out results/v2/eval_zfix_s0.json \
    --per-message results/v2/eval_zfix_s0.npz --wandb
uv run python scripts/analyze_composition_v2.py --pm results/v2/eval_zfix_s0.npz \
    --out results/v2/composition_zfix.json
```

Options:
- `--only <conditions>` restricts a run (for sweeps);
- `--method heun` switches the flow sampler;
- `--per-message` keeps one record per message: the SBM's mode probability and margin, whether the mode is right, the distinct interpretations among the K draws, and each condition's energy score and ADE.

`analyze_composition_v2.py` bins those records by SBM confidence, splits them by right or wrong mode, and gives paired differences between conditions with bootstrap intervals.

| condition | what it is |
|---|---|
| `oracle` | the CTP given the true interpretation |
| `composed` | K SBM samples × S CTP samples |
| `mode` | the SBM's certified joint mode |
| `hist_only` | the separate history-only model |
| `direct`, `direct_init` | the CTP reading the transcript (from scratch; text encoder from the SBM) |
| `kinematic_mode`, `kinematic_true` | the instruction-aware rule, on the SBM mode and on the true interpretation |
| `mdn_*`, `anchors_*` | the mixture heads, given the true interpretation (`oracle`) or on the same K SBM samples (`composed`) |

Every condition is scored by message variant (own, maintain, own_other, other), with calibration by lead time and PIT and rank histograms. All noise is seeded per (window, interpretation, sample), so results are the same across batch sizes and devices. The estimators are described in [results.md](results.md#metrics).

Sweeps change one factor at a time: K × S at 72 samples, Heun at 32 steps, the per-step CTP, and each SBM ablation. They run on 2,000 test windows; `compose_full_*` repeats the K × S and Heun runs on the full test set.

```bash
WANDB=1 bash scripts/run_v2.sh sweeps        # -> results/v2/compose_<tag>.json
```

### Who is the message for

On a real frequency each transmission is for one of N aircraft. `addressee_v2.py` works in three steps:
1. it scores the message against every aircraft on frequency with the SBM;
2. it divides out the training prior (the share of training messages that are "for me": 0.8 for the base SBM, 0.7 with contrast messages);
3. joint addressee inference: each message addresses at most one aircraft, `P(a) = L_a / (Σ_b L_b + L_off)`, and each aircraft's composition is gated by its share.

Scenes are built from the N aircraft whose windows are closest in time on the same day.

```bash
# "other" messages drawn from concurrent traffic (within 2 min), same vocabulary
uv run python -m atc_forecast.data.messages --profile jacktol --tag _conc \
    --vocab-from data/v2/messages_jacktol/vocab.json
uv run python -m atc_forecast.training.sbm_v2 --profile jacktol_conc --out runs/v2/sbm_jacktol_conc_anyorder_s0.pt
uv run python -m atc_forecast.evaluation.addressee_v2 --sbm runs/v2/sbm_jacktol_conc_anyorder_s0.pt \
    --ctp runs/v2/ctp_basis_atoms_zfix_s0.pt --hist-only runs/v2/ctp_basis_none_s0.pt \
    --profile jacktol_conc --n-aircraft 8 --n 1000 --out results/v2/addressee_conc_N8.json
```

Conditions:
- `independent`: each aircraft reads the message alone;
- `prior`: the 1/N prior, without competition;
- `scene`: the resolution;
- `scene_hist`: the resolution, with no-motion readings sent to the history-only model;
- `known`: the addressee is given;
- `oracle` and `hist_only`.

The on-air score weights the addressee 1 : N − 1 against bystanders.

### Real transmissions

`evaluation/real_v2.py` runs the same conditions on real matched transmissions and ADS-B; see [real-data.md](real-data.md).

## 5. Diagnosis

```bash
uv run python scripts/diagnose_v2.py nfe       --ctp runs/v2/ctp_basis_atoms_s0.pt   # H2: NFE 1..64, Euler/Heun
uv run python scripts/diagnose_v2.py overfit   --ctp runs/v2/ctp_basis_atoms_s0.pt   # H3: train vs validation
uv run python scripts/diagnose_v2.py time      --ctp runs/v2/ctp_basis_atoms_s0.pt runs/v2/ctp_basis_atoms_unif_s0.pt  # H4
uv run python scripts/diagnose_v2.py hindsight --ctp runs/v2/ctp_basis_atoms_s0.pt   # H5: by delay, by later onset
uv run python scripts/diagnose_v2.py noise     --ctp runs/v2/ctp_basis_atoms_s0.pt   # H1: smoothed vs raw spread-skill
uv run python scripts/diagnose_v2.py sbm       --sbm runs/v2/sbm_jacktol_anyorder_s0.pt  # order spread, beam certification
uv run python scripts/diagnose_v2.py callsign  --sbm runs/v2/sbm_jacktol_ho_s0.pt --profile jacktol_cs  # which cue decides the status
uv run python scripts/diagnose_v2.py turn      --sbm runs/v2/sbm_jacktol_ho_s0.pt   # unheard turn direction vs track
uv run python scripts/altitude_noise_v2.py                                            # level-flight altitude wiggle vs filter
```

Results go to `results/v2/diagnose_<cmd>.json`, each with a figure next to it; `--wandb` logs them. `scripts/collect_results_v2.py` gathers checkpoints' validation scores and curves, evaluations and diagnoses into `results/v2/results.json`.

## 6. Tests

```bash
uv run pytest -q              # all tests, a few seconds; no data, checkpoints or network
```

- **Resampling:**
  - 1 Hz on both sides;
  - no leakage either way;
  - gaps refused;
  - exact integration;
  - turn tracking;
  - the external interface.
- **Atoms and messages:**
  - grids and relative turns;
  - word errors;
  - confusable callsigns;
  - labels are what was said;
  - determinism.
- **Models:**
  - basis round trip;
  - torch features equal the numpy reference;
  - every CTP, SBM and MDN variant trains and samples;
  - beam scores equal log-probabilities;
  - seeded noise is independent of the batch.
- **Metrics:**
  - the nested energy score is unbiased;
  - calibrated coverage reads 0.5 / 0.9 and rank_out10 reads 0.1;
  - overconfidence is flagged.
- **Real-data path:**
  - transcript normalization;
  - history-only windows equal the full ones;
  - readsb units;
  - the callsign at t0;
  - the `Forecaster`;
  - the evaluation command line end to end.

## The whole plan in one script

```bash
bash scripts/run_v2.sh data        # windows, word confidence, messages, data figures
bash scripts/run_v2.sh core        # SBM + CTP (basis, atoms) + history-only + per-step   (~3.6 h)
bash scripts/run_v2.sh baselines   # MDN, anchors, direct transcript (scratch + SBM init)
bash scripts/run_v2.sh ablations   # SBM factorizations, no history; uniform flow time
bash scripts/run_v2.sh seeds       # seeds 1, 2 for the main models
bash scripts/run_v2.sh channels    # other recognizer profiles
bash scripts/run_v2.sh eval        # every model on test + per-message composition analysis
bash scripts/run_v2.sh diagnose    # H1-H5 and the SBM checks
bash scripts/run_v2.sh sweeps      # composition sweeps (K x S, sampler, CTP and SBM variants)
# prefix any stage with WANDB=1 to log it
```

## Hyperparameters that matter

| | default | effect |
|---|---|---|
| CTP `--output` | basis | 18 coefficients (smooth, cheap) or 180 per-step values |
| CTP `--nfe`, `--method` | 12, euler | sampler steps; too few pull samples toward the mean; Heun with 32 steps is calibrated |
| CTP/MDN `--basis` | pca | PCA of the training residuals, or quarter-wave sines with the same count per axis |
| CTP `--time-sampling` | logit_normal | more training at mid flow times |
| CTP `--p-null` / `--p-maintain` | 0.2 / 0.2 | how often any window is shown a message that is not about its motion, independently of the window (so "not ours" teaches the history-only answer) / how often a maneuver's instruction carries a MAINTAIN clause on a steady axis |
| CTP `--ema` | 0.999 | the evaluated weights; validation lags early |
| SBM `--factorization` | anyorder | exact joint; independent is the cheap ablation |
| SBM `--profile` | jacktol | which recognizer channel's messages |
| composition `--k`, `--s` | 12, 6 | interpretations and futures per interpretation; anything above one interpretation does about as well |
| all `--lr`, `--steps`, `--batch`, `--d` | 3e-4, 60k/40k, 256/512, 128 | standard |
