"""The demo engine reproduces the paper's evaluation on a showcase case (skipped without
the bundle and checkpoints, which are not in the repository)."""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE, MODELS = ROOT / "demo/bundle", ROOT / "runs/v2"
pytestmark = pytest.mark.skipif(not (BUNDLE / "windows.npz").exists() or not MODELS.exists(),
                                reason="demo bundle or checkpoints not installed")


@pytest.fixture(scope="module")
def engine():
    import sys
    sys.path.insert(0, str(ROOT))
    from demo.server.engine import Engine
    return Engine(BUNDLE, MODELS)


def test_fig6a_scores(engine):
    case = next((c for c in engine.cases if c["id"] == "rescue"), None)
    if case is None:
        pytest.skip("paper case not in the bundle")
    r = engine.predict(case["window"], case["message"]["words"], conditions=("composed", "most_probable"))
    assert abs(r["predictions"]["composed"]["scores"]["es_h_m"] - 2182) < 10       # Fig. 6(a)
    assert abs(r["predictions"]["most_probable"]["scores"]["es_h_m"] - 10583) < 30
    assert len(r["predictions"]["composed"]["tracks"]) == 72


def test_scene_probabilities(engine):
    sc = engine.scenes[0]
    tx = next(t for t in sc["transmissions"] if t["recipient"] is not None)
    r = engine.scene([a["window"] for a in sc["aircraft"]], tx["message"]["words"])
    p = np.array([a["p_addressed"] for a in r["aircraft"]])
    assert p.sum() + r["p_outside"] == pytest.approx(1.0, abs=1e-6)
    assert r["aircraft"][[a["window"] for a in r["aircraft"]].index(tx["recipient"])]["p_addressed"] == p.max()


def test_bad_variant(engine):
    from demo.server.engine import BadRequest
    with pytest.raises(BadRequest):
        engine.predict(0, [{"w": "hello"}], sbm="nope")
