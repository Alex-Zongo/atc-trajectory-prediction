---
title: Aircraft trajectory prediction from ADS-B and ATC transcripts
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Predict 60 s of flight from ADS-B and a noisy ATC transcript
---

# Aircraft trajectory prediction from ADS-B and imperfect ATC transcripts

Interactive demo of a semantic belief model (SBM) and a conditional trajectory predictor
(CTP). The SBM reads a controller transcript with word confidences, the aircraft's own
callsign and its last 30 s of ADS-B, and returns a distribution over structured instruction
representations. The CTP samples future trajectories under each sampled representation.

Views:

- **Live sector**: a few aircraft grouped by time on one day; transmissions arrive on the
  frequency, the recipient is identified across the aircraft, and the addressed aircraft's
  prediction updates. Type your own transmission.
- **Try a transmission**: edit a transcript word by word, lower word confidences, and compare
  the composed prediction with the most probable representation, the semantic label, the
  ADS-B-only predictor, the transcript-conditioned flow and a kinematic rule.
- **Compare models**: any two SBM, trajectory-predictor and sampler choices side by side.

All models are the first training run of each configuration. Flights are held-out recorded
ADS-B segments shown in relative coordinates; transmissions are synthetic. No recorded speech
is used, and typed transmissions are not stored.

API: `GET /api/variants`, `GET /api/cases`, `POST /api/predict`, `POST /api/scene`
(see `/api/docs`).

A research prototype, not an operational tool.
