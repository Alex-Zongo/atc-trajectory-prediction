# Web demo

A FastAPI service that runs the v2 models, and a static front end with three views: a live
sector (several aircraft, recipient identification, replay clock), a single-transmission
editor, and a side-by-side model comparison. Every model variant of the paper can be
selected (`server/variants.py`).

```
demo/
  build_bundle.py   held-out windows, synthetic transmissions and scenes -> demo/bundle/ (gitignored)
  server/           variants.py (checkpoint registry), engine.py (model execution), app.py (HTTP)
  web/              index.html, app.js, style.css
  space_files/      Dockerfile and Space card for Hugging Face
  deploy.py         assembles demo/space/ (gitignored), a Docker folder for hosting; uploads nothing
  build_static.py   static version: the same front end on precomputed predictions, for a website
```

Run locally, from a clone with the release assets:

```bash
uv sync --extra demo                                           # in a fresh clone; otherwise --all-extras
uv run python scripts/artifacts.py fetch --models-only --demo  # checkpoints and demo/bundle
uv run uvicorn demo.server.app:app --port 7860
# open http://localhost:7860
```

Maintainers can rebuild the bundle with `uv run python demo/build_bundle.py` (needs data/v2 and the
raw ADS-B day files) and publish it with `scripts/artifacts.py publish --models-only --demo`.

Checkpoints are read from `runs/v2` (`DEMO_MODELS` overrides), the bundle from `demo/bundle`
(`DEMO_BUNDLE`). Only variants whose checkpoints are installed are offered. A prediction (K = 12,
S = 6, 12 Euler steps) takes about 0.2 s on a laptop CPU once the models are loaded; a scene of
five aircraft about 0.6 s.

Static version: `uv run python demo/build_static.py` writes `output/static_demo/` (about 33 MB):
every case under every shown model pair and 15 live-sector scenes, precomputed with the default
sampler. Copy the folder to any static host (the project page uses `/demos/atc-traj/`). Typed
transmissions and word edits need the model service.

Hosting: `deploy.py` assembles a Docker folder (`demo/space/`) that runs anywhere Docker does. On
Hugging Face, Docker Spaces now need a paid plan.

What the bundle contains: test-split windows (validation-split windows as other aircraft in
scenes), airline callsigns only, derived model inputs and the recorded future, positions
relative to the aircraft or to the scene centre, and synthetic transmissions. No absolute
positions, times, icao24 addresses, raw ADS-B or recorded speech.
