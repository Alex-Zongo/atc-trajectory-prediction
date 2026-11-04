---
license: other
license_name: opensky-derived
license_link: https://opensky-network.org/about/terms-of-use
tags:
  - air-traffic-control
  - trajectory-prediction
  - ads-b
  - synthetic-speech-transcripts
size_categories:
  - 1M<n<10M
---

# ATC trajectory segments and synthetic controller transmissions (v2)

Processed data for [atc-trajectory-prediction](https://github.com/Alex-Zongo/atc-trajectory-prediction).

## Contents (`dataset-v2.tar`)

| path | contents |
|---|---|
| `data/v2/arrays/` | 441,454 trajectory segments at 1 Hz: 30 s history (track frame), context, centering, 60 s future velocities |
| `data/v2/windows.parquet` | one row per segment: day, aircraft, maneuver labels and atoms, prediction time, split |
| `data/v2/segments.parquet` | maneuver segmentation of the source tracks |
| `data/v2/basis.npz`, `manifest.json` | per-step scales, PCA basis, build configuration |
| `data/v2/messages_jacktol/` | base synthetic transmissions (2.47 M): tokens, word confidences, labels |
| `data/v2/messages_jacktol_ho/` | final transmission set (2.91 M): adds callsign-contrast transmissions, real-radio phrasing and generated handoffs |

Access is gated: requests are approved by hand, for research use under the OpenSky terms below.

The synthetic transmissions are stored in the `messages_*` folders, the name the code uses for them.

Splits are by flight leg: about 80 % train, 10 % validation, 10 % test. Install with `uv run python scripts/artifacts.py fetch --hub <owner>`.

## Source and terms

- **ADS-B:** derived from OpenSky Network state vectors over the San Francisco Bay Area on five days in 2026. Raw state vectors are not included. Use of this derived data is subject to the [OpenSky terms of use](https://opensky-network.org/about/terms-of-use), and work using it should cite Schäfer et al., "Bringing Up OpenSky", IPSN 2014.
- **Transmissions:** generated from the segments' maneuvers and passed through a simulated recognizer channel. No recorded speech is included. The fitted word-confidence file that the channel uses derives from ATCO2 and is not included either. Without it, the transmission builder falls back to a parametric word-confidence model of the same shape.
- **Not included:** recorded controller speech, transcripts of it, and the matched recorded-speech evaluation set.
