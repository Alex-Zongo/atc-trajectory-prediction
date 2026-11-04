---
license: mit
library_name: pytorch
tags:
  - trajectory-prediction
  - air-traffic-control
  - speech-recognition
  - flow-matching
  - uncertainty-quantification
  - ads-b
pipeline_tag: other
---

# ATC instruction-conditioned trajectory prediction (v2)

Checkpoints for [atc-trajectory-prediction](https://github.com/Alex-Zongo/atc-trajectory-prediction): prediction of an aircraft's next 60 s from its ADS-B history and a controller instruction known only through an imperfect speech transcript.

```
p(Y | O, X)  ≈  Σ_Z  q_φ(Z | O, X) · p_θ(Y | X, Z)
```

- **Semantic belief model (SBM)** `q_φ(Z | O, X)`: a distribution over message interpretations (status, lateral, vertical and speed instruction) from the transcript, each word's recognizer confidence, the aircraft's own ADS-B callsign and its 30 s history.
- **Conditional trajectory predictor (CTP)** `p_θ(Y | X, Z)`: a rectified flow on 18 PCA trajectory-shape coefficients.
- **Composition**: 12 interpretations drawn from the SBM, 6 future trajectories under each.

## Files

| file | contents |
|---|---|
| `runs-v2.tar.gz` | the main model and its baselines, first training run: final SBM, CTP, history-only predictor, transcript-conditioned predictor, mixture-density (anchors) head |
| `SHA256SUMS` | checksums, verified by `scripts/artifacts.py fetch` |

Install into a clone of the repository:

```bash
uv run python scripts/artifacts.py fetch --hub Alex-Zongo --models-only
```

Then, for example:

```python
from atc_forecast.inference import Forecaster, load_states
fc = Forecaster.load("runs/v2/sbm_jacktol_ho_s0.pt", "runs/v2/ctp_basis_atoms_zfix_s0.pt")
```

## Results (synthetic test set, three training runs)

Energy score in metres over 60 s, lower is better, on instructions for the aircraft's own maneuver: history only 3,202; most probable interpretation 1,727 ± 2; **composed 1,682 ± 5**; reference interpretation given 1,556 ± 2. On a private set of 338 recorded controller transmissions matched to ADS-B, composition improves on history only by 9 %, and the addressed aircraft is ranked first among about 12 for 98 % of transmissions. Details: `docs/v2/results.md` in the repository.

## Intended use and limits

Research on trajectory prediction under communication uncertainty. Not for operational use.

- Trained on synthetic messages built from real ADS-B over the San Francisco Bay Area (five days), with transcripts simulated from a fitted recognizer channel.
- The interpretation space covers heading, altitude and speed instructions. Route, conditional and procedure clearances are outside it.
- The flow with 12 Euler steps is moderately overconfident (90 % intervals cover 0.86). On recorded speech the prediction is more overconfident, and its vertical score does not beat the history-only model.
- Speed targets are compared with ground speed, and headings with track.

## Citation

```bibtex
@software{zongo2026atc,
  author  = {Zongo, Alex and Petit, Mathurin and Brusset, Louis and Kam, Jordan and
             Torun, Emir and Ghazanfari, Mahyar and Wei, Peng and Bayen, Alexandre},
  title   = {Aircraft Trajectory Prediction under {ATC} Instruction Uncertainty},
  year    = {2026},
  note    = {Software; paper in preparation}
}
```
