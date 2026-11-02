# v2 results

All numbers come from `results/v2/` (re-created by the commands in [pipeline.md](pipeline.md)). Unless stated otherwise, scores are horizontal energy scores in metres over the 60 s horizon, lower is better.

Every row is one training seed (seed 0), so differences under about 2 % are within seed variation. Seeds 1 and 2 of the five models behind the close comparisons are being trained; this page will report mean ± sd when they finish.

- [Metrics](#metrics)
- [Test results](#test-results)
- [What works](#what-works)
- [What does not matter, or does not work](#what-does-not-matter-or-does-not-work)
- [Where the error is](#where-the-error-is)
- [Explicit maneuver representations (design A)](#explicit-maneuver-representations-design-a)

## Metrics

Each forecast is 72 futures: K = 12 interpretations × S = 6 futures. Futures that share an interpretation are not independent draws from the composed forecast, and the estimators account for that (`evaluation/metrics_v2.py`, `tests/test_metrics_v2.py`):

- **Energy score:** the spread term pairs only futures drawn under different interpretations. That keeps it unbiased for the composed distribution; pairing all 72 inflates it.
- **Coverage (`cov50`, `cov90`):** the share of true coordinates inside the central predictive interval, per coordinate and lead time. It is read from the PIT of sub-ensembles with one future per interpretation, so a calibrated forecast reads exactly 0.50 and 0.90. Each sub-ensemble has K members, so compare coverage only at equal K. The older quantile-based value is kept as `cov90_q` (calibrated: 0.875).
- **`rank_out10`:** the share of messages whose whole true trajectory falls in the most outlying tenth of the forecast, by multivariate rank. It is 0.10 when calibrated, and higher means overconfident.
- **PIT and rank histograms:** 10 bins each, stored in `calibration.<condition>.hist` of every evaluation JSON.

Results computed before this correction are kept in `results/v2/pre_metricfix/`. The correction lowered composed energy scores by about 0.5 % and left the rankings unchanged.

## Test results

5,000 test windows, four messages each:
- `own`: an instruction for this aircraft's maneuver;
- `maintain`: for it, but "keep doing what you are doing";
- `non-motion`: for it, but not about motion (frequency change, traffic);
- `other`: an instruction for another aircraft.

`all` weights them as they occur (1,705 / 295 / 2,000 / 2,000).

| condition | own | maintain | non-motion | other | all | own ADE / FDE (m) | own cov90 | own rank_out10 |
|---|---|---|---|---|---|---|---|---|
| **SBM + CTP, composed (main)** | 1,675 | 1,490 | 3,126 | 3,289 | **2,688** | 236 / 597 | 0.86 | 0.16 |
| SBM best reading + CTP | 1,714 | 1,489 | 3,127 | 3,348 | 2,719 | 235 / 593 | 0.84 | 0.19 |
| CTP given the true interpretation (bound) | 1,554 | 1,489 | 3,127 | 3,126 | 2,599 | 215 / 542 | 0.85 | 0.18 |
| SBM + anchors (mixture head), composed | 1,666 | 1,486 | 3,259 | 3,407 | 2,769 | 238 / 606 | 0.91 | 0.09 |
| CTP reading the transcript (SBM-initialized) | 2,108 | 1,465 | 3,010 | 3,129 | 2,718 | 292 / 785 | 0.85 | 0.17 |
| CTP reading the transcript (from scratch) | 2,187 | 1,478 | 2,995 | 3,173 | 2,751 | 303 / 814 | 0.85 | 0.17 |
| history only (ignores the radio) | 3,202 | 1,733 | 2,988 | 2,988 | 2,988 | 435 / 1,166 | 0.85 | 0.17 |
| kinematic rule on the true interpretation | – | – | – | – | – | 399 / 940 | – | – |

The SBM's best reading equals the true interpretation for 93 % of messages. The CTP, anchors and history-only rows come from `eval_zfix_s0`; the SBM-initialized transcript model comes from `eval_test_s0`, on the same windows and messages.

## What works

1. **Reading the instruction.** Own-maneuver error drops from 3,202 (history only) to 1,675, and ADE from 435 m to 236 m. The instruction carries most of the information about the next minute.
2. **The interpretation bottleneck** (transcript → SBM → Z → CTP) beats a CTP that reads the transcript directly, on the messages that matter: 1,675 against 2,108–2,187. It also gives readable intermediate outputs: interpretations with probabilities. Initializing the direct model's text encoder from the SBM closes only about a sixth of the gap.
3. **Composition over the SBM's samples** rather than its best reading: 1,675 against 1,714 on own messages (paired difference −39 m, 95 % CI [−61, −17]), and 3,289 against 3,348 on messages for others.
   - **Where the gain is:** on the 16 % of messages whose best reading is wrong, composition lowers the error from 2,514 to 2,097. That closes 42 % of the gap to the true-interpretation bound (1,516).
   - **What it costs:** where the best reading is right, it costs 2.1 % (1,561 → 1,594).
4. **The allocation matters little beyond one interpretation** (full test set):

   | K × S | 12 × 6 | 36 × 2 | 4 × 18 | 1 × 72 |
   |---|---|---|---|---|
   | own | 1,675 | 1,669 | 1,673 | 1,803 |

5. **History in the SBM.** Without it (2,000-window subset), the best reading is right 90 % of the time instead of 94 %, and composed own error rises from 1,590 to 1,721. The aircraft's state is what tells *descend four thousand* from *climb four thousand* when the verb is misheard.
6. **Window-independent "not for us" training labels.** These are drawn independently of the window, and add a MAINTAIN clause only on steady axes. Maintain-message error falls from 1,573 to 1,490, and all-message error by 1.2 %.
7. **Who the message is for** (8 aircraft per frequency, 1,000 messages, `addressee_v2`):
   - **Bystanders:** when the aircraft compete for each message, the other aircraft's forecasts improve 44 % (5,490 → 3,091), because they stop following instructions whose callsign was clipped.
   - **On-air score:** routing no-motion readings to the history-only model brings the weighted on-air score to 2,788. That is better than giving every aircraft its true interpretation (2,890).
   - **Cost to the addressee:** when its callsign is clipped (22 % of messages), its error rises from 1,929 to 2,635, which is the honest answer to an ambiguous message.
   - **Ranking:** the addressee ranks first for 82 % of messages with 8 aircraft. With the SBM trained on concurrent traffic, it ranks first for 90 %, 84 % and 79 % with 4, 8 and 16 aircraft.
8. **The flow sampler.**
   - **Euler steps:** with 12 steps the flow is mildly overconfident (cov90 0.86, rank_out10 0.16).
   - **Heun, 32 steps:** on the same weights, cov90 reaches 0.90 and rank_out10 0.105, with own error 1,662 (full test set). Most of the overconfidence therefore comes from the coarse integration, not from the learned field; Heun costs about five times the network calls.
   - **Too few steps:** samples collapse toward the mean. With one Euler step, 90 % coverage falls to 0.40 (older quantile definition, `diagnose_v2.py nfe`).

## What does not matter, or does not work

| change | result | verdict |
|---|---|---|
| SBM: independent slots, or a mixture over joint readings (2,000 windows) | composed own 1,600 / 1,597 against 1,590 | no difference once composed; any-order kept for its exact joint and certified mode |
| SBM trained on concurrent distractors (other aircraft's real instructions within 2 min) | addressee first 84 % against 82 %; on-air 2,812 against 2,788 | no gain: clipped callsigns, not distractors, are the bottleneck |
| mixture head (anchors; MDN from the earlier-label run) instead of the flow | own 1,666–1,669 against 1,675; all 2,750–2,769 against 2,688; calibrated (cov90 0.90–0.91, rank_out10 0.09–0.10); one network call instead of twelve | a valid fast alternative; the flow wins on non-instruction messages |
| flow on the 180 per-second values instead of 18 coefficients (2,000 windows) | own 1,606 against 1,590, all +5 %, cov90 0.88 against 0.86, 2.4× the training time | not worth it |
| sine basis instead of PCA (same 6 / 4 / 8 shapes per axis) | own 1,674 against 1,675, all +1.3 %, cov90 0.83 against 0.86; anchors on sines 1,713 | the basis is not the accuracy bottleneck; PCA calibrates better |
| uniform flow-time sampling instead of logit-normal | 1,666 against 1,631 (validation) | logit-normal kept |
| hand-written kinematic rule on the true interpretation | own ADE 399 m against 215 m | a learned model halves the error |

## Where the error is

- **Messages for other aircraft and non-motion messages** carry the largest errors. On these the history-only model is still better (2,988 against 3,126–3,289), which is why routing no-motion readings to it helps on a frequency. A CTP trained with more such examples is the clean fix.
- **A second maneuver later in the minute:** windows with one score 1,724 against 1,392 without. No message announced it.
- **Generalization:** train 1,432 against validation 1,631, a modest gap.
- **Altitude noise:** ADS-B altitude is quantized to 25 ft and wiggles about 10 ft (median) in level flight. Vertical errors of that size are measurement, not model, error.

## Explicit maneuver representations (design A)

PCA coefficients are compact but not directly readable. Design A replaces them with quantities a controller could inspect:
- a **kind**, from 8 physical maneuver kinds;
- **primitives** per axis: delay, rate (bank angle for turns), ramp and capture;
- a **kinematic rule** that flies them.

With the true interpretation, own error:

| flow on | joint kinds, fed to the flow | per-axis kinds | kinds as a readout only |
|---|---|---|---|
| PCA weights (`--model kindctp`) | 1,605 | 1,596 | 1,584 |
| primitives (`--model prim`) | 1,633 | 1,652 | – |
| primitives with bank angle (`primitives_v3`) | 1,624 | – | 1,631 |

The CTP without kinds scores 1,554. Composed, the bank-angle primitives reach own 1,733 and all 2,736, against 1,675 / 2,688.

- **Kinds cost about 3 % when fed to the flow,** and less than 2 % as a readout. The readout gives kind probabilities almost for free.
- **Primitives cost 4–5 %** with or without kinds, and buy calibration (cov90 0.88 against 0.86).
- **The main model stays the CTP.** The cheapest readable add-on is a kind readout on it. The 1–2 % differences here are what seeds 1 and 2 will settle.

Targets come from `scripts/build_primitives_v2.py` (`--clean --bank --out primitives_v3.npz` for the bank-angle state); models are trained with `traj_v2 --model prim` or `--model kindctp`, with `--kind-readout` for the readout variant.
