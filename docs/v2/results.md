# v2 results

All numbers come from `results/v2/` (re-created by the commands in [pipeline.md](pipeline.md)). Unless stated otherwise, scores are horizontal energy scores (ES) in metres over the 60 s horizon, lower is better.

**Main results are means over three training runs** (seeds 0, 1, 2) of the SBM, the CTP and the mixture head, with the standard deviation across runs where it matters. The ADS-B-only and transcript-conditioned models are trained once. The SBM is the **final SBM**: trained with callsign-contrast transmissions, then fine-tuned on generated handoff phrases (see [the SBM versions](#the-sbm-versions)). Sensitivity studies and explicit maneuver representations use the initial SBM (no callsign contrast) and the first run, and are compared within a group.

- [Terms](#terms)
- [Metrics](#metrics)
- [Test results](#test-results)
- [The SBM versions](#the-sbm-versions)
- [What works](#what-works)
- [What does not matter, or does not work](#what-does-not-matter-or-does-not-work)
- [Recorded controller speech](#recorded-controller-speech)
- [Where the error is](#where-the-error-is)
- [Explicit maneuver representations (design A)](#explicit-maneuver-representations-design-a)

## Terms

| term | meaning |
|---|---|
| structured instruction representation `Z` | what a transmission means for one aircraft: a transmission status (own-aircraft motion instruction, own-aircraft transmission without a motion instruction, other-aircraft transmission) and a lateral, vertical and speed field |
| semantic label `Z*` | the representation a synthetic test transmission was generated from; conditioning the CTP on it is a reference that is not available in operation, and not a bound on the error |
| most probable instruction | the SBM's most probable representation, found by beam search and certified exact |
| composed prediction | 12 representations sampled from the SBM, 6 future trajectories under each |
| trajectory segment | one test sample: 30 s of history and the next 60 s |
| recipient identification using multiple aircraft | on a frequency, a transmission addresses at most one aircraft: the SBM's evidence that each aircraft is addressed is normalized across the aircraft |

## Metrics

Each prediction is 72 future trajectories: K = 12 sampled representations × S = 6 trajectories. Trajectories that share a representation are not independent draws from the composed prediction, and the estimators account for that (`evaluation/metrics_v2.py`, `tests/test_metrics_v2.py`):

- **Energy score:** the spread term pairs only trajectories drawn under different sampled representations. That keeps it unbiased for the composed distribution, while pairing all 72 inflates it.
- **Coverage (`cov50`, `cov90`):** the share of true coordinates inside the central predictive interval, per coordinate and lead time. It is read from the PIT of sub-ensembles with one trajectory per sampled representation, so a calibrated prediction reads exactly 0.50 and 0.90. Each sub-ensemble has K members, so compare coverage only at equal K. The older quantile-based value is kept as `cov90_q` (calibrated: 0.875).
- **`rank_out10`:** the share of transmissions whose whole true trajectory falls in the most outlying tenth of the prediction, by multivariate rank. It is 0.10 when calibrated, and higher means overconfident.
- **PIT and rank histograms:** 10 bins each, stored in `calibration.<condition>.hist` of every evaluation JSON.

## Test results

5,000 test segments, three transmissions each, in four categories:
- `own`: an instruction for this aircraft's maneuver;
- `maintain`: for it, but "keep doing what you are doing";
- `non-motion`: for it, but not about motion (frequency change, traffic);
- `other`: an instruction for another aircraft.

`all` weights them as they occur (1,705 / 295 / 2,000 / 2,000).

| condition | own | maintain | non-motion | other | all | own ADE / FDE (m) | own cov90 |
|---|---|---|---|---|---|---|---|
| **SBM + CTP, composed (main)** | **1,682** ± 5 | 1,497 | 3,152 | 3,296 | **2,701** | 238 / 604 | 0.86 |
| SBM + CTP, most probable instruction | 1,727 ± 2 | 1,496 | 3,151 | 3,353 | 2,732 | 236 / 600 | 0.84 |
| SBM + mixture-density CTP (anchors), composed | 1,672 ± 2 | 1,488 | 3,254 | 3,388 | 2,762 | 240 / 613 | 0.90 |
| CTP conditioned on the transcript (one run) | 2,187 | 1,478 | 2,995 | 3,173 | 2,751 | 303 / 814 | 0.85 |
| ADS-B only (ignores the radio) | 3,202 | 1,733 | 2,988 | 2,988 | 2,988 | 435 / 1,166 | 0.85 |
| CTP given the semantic label (reference) | 1,556 ± 2 | 1,496 | 3,151 | 3,149 | 2,615 | 215 / 545 | 0.85 |
| kinematic rule on the semantic label | – | – | – | – | – | 399 / 940 | – |

Paired difference, composed minus most probable instruction, own transmissions: −46 ± 3 m across runs (first run: 95 % bootstrap interval [−66, −30] m). Composition is ahead in every run. With an SBM-initialized text encoder, the transcript-conditioned CTP reaches 2,108 on own transmissions.

## The SBM versions

| version | training transmissions | used for |
|---|---|---|
| base | own 50 %, own without motion 20 %, other aircraft 20 %, maintain 10 % | sensitivity studies, explicit representations |
| + callsign contrast | adds 15 % *contrast* transmissions: the aircraft's own instruction under another aircraft's callsign, labeled "other aircraft" (`messages.py --swap`), and phrasing heard on the recorded NorCal frequency (`--real-phrasing 0.15`) | ablation |
| **final** | the callsign-contrast SBM, fine-tuned 10,000 steps on a set with generated handoff phrases: random facility names and frequencies (`messages.py --handoff 0.5`) | **main results** |

The final SBM also reads two callsign-alignment features per transcript token (`--cs-match`), and its prior share of transmissions addressed to the aircraft is 0.7 instead of 0.8.

**Callsign probe** (`diagnose_v2.py callsign`, 600 own transmissions with the callsign received, share still assigned to the aircraft):

| probe | base | + callsign contrast | final |
|---|---|---|---|
| A: original transmission | 99.8 % | 99.7 % | 99.7 % |
| B: callsign of another aircraft | **68.8 %** | 14.5 % | 14.9 % |
| C: history of another aircraft | 83.6 % | 83.9 % | 84.9 % |
| E: own instruction, other callsign (held out) | – | 4.4 % | 4.5 % |

In the base training set, every transmission for another aircraft carried an instruction that did not fit the receiving aircraft's history, so the history alone separated the classes and the SBM barely compared callsigns. The contrast transmissions remove that shortcut. On the single-aircraft test, the final SBM scores 1,682 on own transmissions against 1,678 for the base SBM.

## What works

1. **Reading the instruction.** Own-maneuver error drops from 3,202 (ADS-B only) to 1,682, which retains 92 % of the gain that the semantic label gives (1,556). ADE drops from 435 m to 238 m.
2. **An explicit structured representation** (transcript → SBM → Z → CTP) beats a CTP that reads the transcript directly, on the transmissions that matter: 1,682 against 2,108–2,187. It also gives readable intermediate outputs: instruction representations with probabilities.
3. **Composition over the SBM's samples** rather than the most probable instruction: −46 ± 3 m on own transmissions.
   - **Where the gain is:** on the 16 % of transmissions whose most probable instruction is wrong, composition lowers the error from 2,520 to 2,116, closing 38 % of the gap to the semantic label (1,471).
   - **What it costs:** where the most probable instruction is right, 1.3 % (1,570 → 1,590).
4. **Turn direction from the aircraft's track** (`diagnose_v2.py turn`). On 6,963 test transmissions whose transcript kept the heading but lost "left"/"right", the SBM's direction matches the shorter turn 95 % of the time and the flown direction 93 % (the aircraft flew the shorter turn 92 % of the time). Mirroring the aircraft's track about the instructed heading reverses its choice for 87 % of transmissions, and for 99 % of 30–90° turns. Near 180°, where geometry says little, it keeps both directions.
5. **The allocation matters little beyond one sampled representation** (initial SBM, full test set):

   | K × S | 12 × 6 | 36 × 2 | 4 × 18 | 1 × 72 |
   |---|---|---|---|---|
   | own | 1,675 | 1,669 | 1,673 | 1,803 |

6. **History in the SBM.** Without it (2,000-segment subset), the most probable instruction is right 90 % of the time instead of 94 %, and composed own error rises from 1,590 to 1,721.
7. **Segment-independent no-motion training labels** for the CTP. Maintain-transmission error falls from 1,573 to 1,490, and all-transmission error by about 1 %.
8. **Who the transmission is for** (8 aircraft per scene, 1,000 transmissions, `addressee_v2`, final SBM):
   - **Other aircraft:** with each aircraft judged alone they score 3,874, already 29 % below the initial SBM's 5,464. Recipient identification using multiple aircraft brings this to 3,088, but ADS-B only for every aircraft scores 2,876, so without routing the radio does not help the aircraft it does not address.
   - **Scene score:** routing samples without motion content to the ADS-B-only model brings the weighted scene score to 2,773, the only configuration below ADS-B only for every aircraft (2,936). It is also better than giving every aircraft its semantic label (2,890), because the CTP's no-motion answer is weaker than the ADS-B-only model's.
   - **Cost to the recipient:** when its callsign is clipped, its error rises from 1,924 to 2,634 (2,590 with routing), the honest answer to an ambiguous transmission.
   - **Ranking:** the recipient ranks first for 91 %, 87 % and 83 % of transmissions with 4, 8 and 16 aircraft. With 16 aircraft the gap to a given recipient falls from 451 m (initial SBM) to 313 m.
9. **The flow sampler.**
   - **Euler steps:** with 12 steps the flow is mildly overconfident (cov90 0.86, rank_out10 0.15).
   - **Heun, 32 steps:** on the same weights, cov90 reaches 0.90 and rank_out10 0.10, with own error 1,662 (base SBM, full test set). Most of the overconfidence therefore comes from the coarse integration, not from the learned field. Heun costs about five times the network calls.

## What does not matter, or does not work

| change | result | verdict |
|---|---|---|
| SBM: independent fields, or a mixture over fields (2,000 segments) | composed own 1,600 / 1,597 against 1,590 | no difference once composed. Any-order kept for its exact joint and certified most probable instruction |
| SBM trained on concurrent transmissions for other aircraft (real instructions within 2 min) | recipient first 84 % against 82 % | no gain. The callsign comparison, not more negatives, was missing (see callsign contrast) |
| mixture-density head (anchors) instead of the flow | own 1,672 ± 2 against 1,682 ± 5 (paired −9 ± 4); all 2,762 against 2,701; calibrated (cov90 0.90, rank_out10 0.09); one network call instead of twelve | a valid fast alternative. The flow wins on non-instruction transmissions |
| flow on the 180 per-second values instead of 18 coefficients (2,000 segments) | own 1,606 against 1,590, all +5 %, cov90 0.88 against 0.86, 2.4× the training time | not worth it |
| sine basis instead of PCA (same 6 / 4 / 8 shapes per axis) | own 1,674 against 1,675, all +1.3 %, cov90 0.83 against 0.86 | the basis is not the accuracy bottleneck. PCA calibrates better |
| uniform flow-time sampling instead of logit-normal | 1,666 against 1,631 (validation) | logit-normal kept |
| hand-written kinematic rule on the semantic label | own ADE 399 m against 215 m | a learned model halves the error |

## Recorded controller speech

### July 2026, San Francisco and Boston (paper's evaluation)

12,260 transmissions recorded from public LiveATC feeds at KSFO and KBOS on 10 and 14 July 2026, transcribed by two ATC-fine-tuned Whisper large-v3 recognizers (one per airport) and paired with one aircraft's OpenSky track by callsign. The set belongs to a research collaboration: it is used for evaluation only (no training, tuning or model selection) and is not released, so only aggregates are reported. It has no human labels: transcripts are machine output, and scores are measured against the trajectory each aircraft then flew. 9,010 transmissions meet the window requirements; 4,863 are controller transmissions. Boston lies outside the region the trajectory models were trained on. Means ± sd over three training runs; ADS-B only is one run.

| controller transmissions | n | ADS-B only | most probable instruction | **composed** | vertical ES (ft), composed (ADS-B only) |
|---|---|---|---|---|---|
| all | 4,863 | 4,400 | 3,649 ± 15 | **3,452 ± 46** | 1,050 (1,006) |
| heading, altitude, speed instructions | 1,851 | 5,064 | 3,582 ± 37 | **3,360 ± 26** | 1,001 (1,106) |
| handoffs | 1,180 | 3,953 | 3,850 ± 58 | **3,738 ± 66** | 1,203 (1,109) |
| KSFO | 2,353 | 3,667 | 3,426 ± 40 | **3,228 ± 42** | 1,023 (930) |
| KBOS (outside the training region) | 2,510 | 5,088 | 3,858 ± 65 | **3,661 ± 85** | 1,076 (1,078) |
| tower | 767 | **2,841** | 3,288 ± 119 | 3,146 ± 124 | 1,069 (1,013) |

- **The interface transfers:** composition improves on ADS-B only by 22 % on all controller transmissions and by 34 % on heading, altitude and speed instructions, and on the most probable instruction by 197 m (95 % interval 176–221 m). On the 1,352 controller transmissions whose audio the recognizers never saw in training, the improvement is 23 %. The addressed aircraft ranks first among nearby aircraft for 93 % of controller transmissions.
- **Limits:** tower transmissions carry takeoff and landing clearances outside the representation and are worse than ADS-B only; the vertical score is slightly worse overall; and the prediction is overconfident: over all 9,010 transmissions, nominal 90 % intervals cover 0.81 of flown positions and 24 % of flown trajectories fall in the outer tenth.

### Earlier set: NorCal Departure

A private evaluation set (not fully independent of the contrast and final SBMs, whose training phrasing was taken from it): 72.9 h of a public LiveATC feed of NorCal Departure (29 April to 2 May 2026, 298 files of 15 minutes), transcribed by a general-purpose Whisper large-v3 and matched to adsb.lol ADS-B by callsign. 1,929 controller transmissions are matched, and 338 have ADS-B coverage that meets the training requirements. The recordings and transcripts are for personal research use, so only aggregates are reported. How to run the same evaluation on your own data: [real-data.md](real-data.md).

| SBM | composed ES | most probable | matched aircraft first | handoffs read as "own, no motion" |
|---|---|---|---|---|
| base | 4,786 ± 323 | 5,460 | 39 % | 23 % (70 % read as climbs) |
| + callsign contrast | 4,805 ± 300 | 5,441 | 96 % | 26 % |
| **final** | 4,906 ± 52 | 5,454 | **98 %** | **87 %** |

ADS-B only: 5,397 m. Mean ± sd over three SBM runs, with the same CTP throughout. "base" is the initial SBM.

- **The interface transfers:** with the final SBM, composition beats ADS-B only by 9 % and the most probable instruction by 10 % (3 % on synthetic transmissions). The prediction is overconfident: 30 % of flown trajectories fall in the outer tenth.
- **The callsign comparison transfers:** the matched aircraft, among about 12 within 40 nm, is ranked first for 98 % of transmissions, against 39 % for the base SBM.
- **Handoffs:** the base SBM's lower handoff score (4,578 m, but 3,917–5,329 across runs) comes from misreading frequencies as climbs. Under those climbs the horizontal error falls and the vertical error rises, in every run. The final SBM reads them correctly and scores 5,328 ± 39 m, below ADS-B only (5,569 m). The overall change from the callsign-contrast SBM is −171, +131 and +344 m in the three runs, so there is no consistent effect on the mean, but the spread across runs falls from ±300 to ±52 m.
- **Open:** the vertical score is worse than ADS-B only (1,360 against 1,221 m), and altitude instructions barely improve it.

## Where the error is

- **Transmissions for other aircraft and non-motion transmissions** carry the largest errors. On these the ADS-B-only model is still better (2,988 against 3,152–3,296). The CTP conditioned on the semantic label scores about the same as composition here (3,151 and 3,149), so the gap is in the CTP's no-motion answer, not in the SBM; this is also why routing samples without motion content to the ADS-B-only model helps on a frequency. A CTP trained with more such examples is the clean fix.
- **A second maneuver later in the minute:** segments with one score 1,724 against 1,392 without. No transmission announced it.
- **Generalization:** train 1,432 against validation 1,631, a modest gap.
- **Altitude noise:** ADS-B altitude is quantized to 25 ft and wiggles about 10 ft (median) in level flight. Vertical errors of that size are measurement error, not model error.
- **Recorded speech:** overconfidence, vertical prediction, and phrases outside the training phraseology.

## Explicit maneuver representations (design A)

PCA coefficients are compact but not directly readable. Design A replaces them with quantities a controller could inspect:
- a **maneuver type**, from 8 physical types;
- **primitives** per axis: delay, rate (bank angle for turns), transition duration and capture;
- a **kinematic rule** that flies them.

Given the semantic label, own error (initial SBM, first run):

| flow on | joint types, conditioning the flow | per-axis types | types as a readout only |
|---|---|---|---|
| PCA coefficients (`--model kindctp`) | 1,605 | 1,596 | 1,584 |
| primitives (`--model prim`) | 1,633 | 1,652 | – |
| primitives with bank angle (`primitives_v3`) | 1,624 | – | 1,631 |

The CTP without maneuver types scores 1,554. Composed, the bank-angle primitives reach own 1,733 and all 2,736, against 1,675 / 2,688.

- **Maneuver types cost about 3 % when passed to the flow,** and less than 2 % as a readout. The readout gives type probabilities almost for free.
- **Primitives cost 4–5 %** with or without types, and improve coverage (cov90 0.88 against 0.86).
- **The main model stays the CTP.** The cheapest readable add-on is a maneuver-type readout on it.

Targets come from `scripts/build_primitives_v2.py` (`--clean --bank --out primitives_v3.npz` for the bank-angle state). Models are trained with `traj_v2 --model prim` or `--model kindctp`, with `--kind-readout` for the readout variant.
