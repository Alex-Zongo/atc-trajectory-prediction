"""HTTP service for the demo: the static front end and a small JSON API.

    uv run --extra demo uvicorn demo.server.app:app --port 7860
    # environment: DEMO_BUNDLE (default demo/bundle), DEMO_MODELS (default runs/v2)

    GET  /api/health      models installed, bundle size
    GET  /api/variants    the selectable models and samplers
    GET  /api/cases       single-aircraft cases and multi-aircraft scenes
    POST /api/predict     one aircraft, one transcript
    POST /api/scene       several aircraft, one transcript

Nothing typed by a visitor is stored: requests are processed in memory and not logged.
"""

import os
import threading
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import variants as V
from .engine import BadRequest, Engine

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path(os.environ.get("DEMO_BUNDLE", ROOT / "bundle"))
MODELS = Path(os.environ.get("DEMO_MODELS", ROOT.parent / "runs" / "v2"))
torch.set_num_threads(int(os.environ.get("DEMO_THREADS", min(4, os.cpu_count() or 1))))

if not (BUNDLE / "windows.npz").exists():
    raise SystemExit(f"no demo bundle in {BUNDLE}: run `uv run python scripts/artifacts.py fetch --models-only --demo`"
                     " (or build it with demo/build_bundle.py)")
engine = Engine(BUNDLE, MODELS)
lock = threading.Lock()                 # one model call at a time on a small CPU
app = FastAPI(title="Aircraft trajectory prediction from ADS-B and imperfect ATC transcripts: demo", docs_url="/api/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"],
                   allow_headers=["*"])


class Word(BaseModel):
    w: str = Field(max_length=40)
    c: float | None = Field(default=None, ge=0.0, le=1.0)


class Common(BaseModel):
    words: list[Word] = Field(max_length=60)
    sbm: str = V.DEFAULTS["sbm"]
    ctp: str = V.DEFAULTS["ctp"]
    sampler: str = V.DEFAULTS["sampler"]
    K: int = V.DEFAULTS["K"]
    S: int = V.DEFAULTS["S"]
    seed: int = Field(default=0, ge=0, le=10_000)


class PredictReq(Common):
    window: int
    conditions: list[str] = ["composed", "most_probable", "adsb_only"]


class SceneReq(Common):
    windows: list[int] = Field(min_length=1, max_length=16)
    mode: str = "normalized"
    routing: bool = False


def run(fn, **kw):
    try:
        with lock:
            return fn(**kw)
    except BadRequest as e:
        raise HTTPException(400, str(e)) from None
    except Exception as e:  # noqa: BLE001 -- report any model failure as a readable error
        raise HTTPException(500, f"model error: {type(e).__name__}: {e}") from None


@app.get("/api/health")
def health():
    files = V.files()
    have = [f for f in files if (MODELS / f).exists()]
    return {"ok": True, "models": f"{len(have)}/{len(files)}", "missing": sorted(set(files) - set(have)),
            "windows": len(engine.meta), "scenes": len(engine.scenes), "cases": len(engine.cases)}


@app.get("/api/variants")
def variants(all: bool = False):
    return V.public(everything=all, available={f.name for f in MODELS.glob("*.pt")})


@app.get("/api/cases")
def cases():
    return {"cases": engine.cases, "scenes": engine.scenes, "windows": engine.meta}


@app.post("/api/predict")
def predict(r: PredictReq):
    return run(engine.predict, wid=r.window, words=[w.model_dump() for w in r.words], sbm=r.sbm,
               ctp=r.ctp, sampler=r.sampler, K=r.K, S=r.S, seed=r.seed,
               conditions=tuple(r.conditions))


@app.post("/api/scene")
def scene(r: SceneReq):
    return run(engine.scene, wids=r.windows, words=[w.model_dump() for w in r.words], sbm=r.sbm,
               ctp=r.ctp, sampler=r.sampler, K=r.K, S=r.S, seed=r.seed, mode=r.mode,
               routing=r.routing)


WEB = ROOT / "web"
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")
