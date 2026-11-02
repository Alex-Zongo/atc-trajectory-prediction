"""v2 models on tiny configs: shapes, exactness of the bookkeeping, reproducibility."""

import numpy as np
import torch

from atc_forecast.data import atoms as A
from atc_forecast.models.baselines import MDN, MDNConfig, kinematic_rule
from atc_forecast.models.ctp import CTP, CTPConfig, TargetSpace, seeded_noise
from atc_forecast.models.encoders import InstructionTokens
from atc_forecast.models.sbm import SBM, SBMConfig

B, V = 5, 40


def basis(k=(6, 4, 8)):
    rng = np.random.default_rng(0)
    comps = [np.linalg.qr(rng.normal(size=(60, n)))[0].T.astype(np.float32) for n in k]
    return {"scale": np.full((60, 3), 2.0, np.float32), "mean": np.zeros((60, 3), np.float32),
            "lateral_along": comps[0], "lateral_right": comps[1], "vertical": comps[2]}


def batch(with_text=False):
    g = torch.Generator().manual_seed(0)
    b = {"hist": torch.randn(B, 30, 9, generator=g), "ctx": torch.randn(B, 5, generator=g),
         "center": torch.tensor([[120.0, 0.0, -5.0]] * B), "fut_vel": torch.randn(B, 60, 3) + 100,
         "wake": torch.zeros(B, dtype=torch.long), "engine": torch.zeros(B, dtype=torch.long),
         "n_engines": torch.full((B,), 2.0), "state": torch.tensor([[90.0, 9000.0, 250.0]] * B),
         "z": torch.tensor([[0, 50, 20, 0]] * B), "window_id": np.arange(B)}
    if with_text:
        b["tokens"] = torch.randint(1, V, (B, 20), generator=g)
        b["kappa"] = torch.rand(B, 20, generator=g)
        b["cs_tokens"] = torch.randint(1, V, (B, 6), generator=g)
    return b


def test_target_space_round_trip():
    bs = basis()
    R = torch.randn(4, 60, 3)
    steps = TargetSpace(bs, "steps")
    assert torch.allclose(steps.decode(steps.encode(R)), R, atol=1e-5)
    full = basis((60, 60, 60))
    sp = TargetSpace(full, "basis")
    assert sp.shape == (180, 1)
    assert torch.allclose(sp.decode(sp.encode(R)), R, atol=1e-3)


def test_instruction_features_match_numpy():
    it = InstructionTokens(16)
    rng = np.random.default_rng(0)
    for slot in ("lateral", "vertical", "speed"):
        a = rng.integers(0, A.SIZES[slot], 64)
        st = np.stack([rng.uniform(0, 360, 64), rng.uniform(1000, 40000, 64),
                       rng.uniform(100, 500, 64)], 1).astype(np.float32)
        ref = A.relative_features(slot, a, st[:, 0], st[:, 1], st[:, 2])
        got = it.features(slot, torch.tensor(a), *torch.tensor(st).T).numpy()
        assert np.allclose(ref, got, atol=1e-4), slot


def test_ctp_variants_train_and_sample():
    for output in ("basis", "steps"):
        for cond in ("atoms", "none", "text"):
            m = CTP(CTPConfig(d=32, dec_layers=1, hist_layers=1, text_layers=1, output=output,
                              condition=cond, vocab_size=V), basis())
            b = batch(cond == "text")
            loss = m.loss(b)
            loss.backward()
            assert torch.isfinite(loss)
            m.eval()
            v = m.sample(b, 3, nfe=2)
            assert v.shape == (B, 3, 60, 3)
    m = CTP(CTPConfig(d=32, dec_layers=1, hist_layers=1), basis()).eval()
    z = torch.tensor([[[0, 50, 20, 0], [2, 0, 0, 0]]] * B)
    assert m.sample(batch(), 4, z=z, nfe=2).shape == (B, 2, 4, 60, 3)


def test_zero_initialised_flow_returns_the_centering_velocity_plus_scaled_noise_mean():
    m = CTP(CTPConfig(d=32, dec_layers=1, hist_layers=1, output="steps"), basis()).eval()
    b = batch()
    noise = torch.zeros(B, 2, 60, 3)
    v = m.sample(b, 2, noise=noise, nfe=4)
    assert torch.allclose(v, b["center"][:, None, None].expand_as(v))


def test_seeded_noise_is_independent_of_batch_composition():
    a = seeded_noise([7, 3, 11], 4, (18, 1), run_seed=5)
    b = seeded_noise([11, 7], 4, (18, 1), run_seed=5)
    assert torch.equal(a[0], b[1]) and torch.equal(a[2], b[0])


def test_sbm_factorizations():
    b = batch(True)
    z = torch.tensor([[0, 50, 20, 0], [2, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 3], [0, 0, 70, 0]])
    b["z"] = z
    for f in ("anyorder", "independent", "mixture"):
        m = SBM(SBMConfig(vocab_size=V, d=32, text_layers=1, hist_layers=1, slot_blocks=1,
                          factorization=f, n_mix=3))
        loss, _ = m.loss(b)
        loss.backward()
        m.eval()
        lp = m.log_prob(b, z)
        assert lp.shape == (B,) and (lp < 0).all()
        s = m.sample(b, 6)
        assert s.shape == (B, 6, 4)
        if f != "mixture":
            zb, lpb, cert = m.beam(b, width=4)
            assert zb.shape == (B, 4, 4) and cert.dtype == torch.bool
            # the beam's own score is the model's log-probability of what it returns
            assert torch.allclose(lpb[:, 0], m.log_prob(b, zb[:, 0]), atol=1e-4)


def test_sbm_status_marginal_normalizes():
    m = SBM(SBMConfig(vocab_size=V, d=32, text_layers=1, hist_layers=1, slot_blocks=1)).eval()
    b = batch(True)
    tot = 0
    for s in range(3):
        z = torch.tensor([[s, 0, 0, 0]] * B)
        enc = m.encode(b)
        lg = m.logits(m.slots(enc, torch.zeros(B, 4, dtype=torch.bool), z))[0]
        tot = tot + torch.softmax(lg, -1)[:, s]
    assert torch.allclose(tot, torch.ones(B), atol=1e-5)


def test_sbm_sampling_is_reproducible_with_generators():
    from atc_forecast.models.sbm import window_generators
    m = SBM(SBMConfig(vocab_size=V, d=32, text_layers=1, hist_layers=1, slot_blocks=1)).eval()
    b = batch(True)
    s1 = m.sample(b, 5, generators=window_generators(range(B), 3))
    s2 = m.sample(b, 5, generators=window_generators(range(B), 3))
    assert torch.equal(s1, s2)


def test_mdn_trains_and_samples():
    for anchors in (False, True):
        K = 4
        m = MDN(MDNConfig(d=32, hist_layers=1, n_comp=K, anchors=anchors), basis(),
                np.zeros((K, 18), np.float32) if anchors else None)
        loss = m.loss(batch())
        loss.backward()
        assert torch.isfinite(loss)
        assert m.eval().sample(batch(), 3).shape == (B, 3, 60, 3)


def test_kinematic_rule():
    center = np.array([[128.6, 0.0, 0.0]] * 3)
    state = np.array([[90.0, 10000.0, 250.0]] * 3)
    r270 = A.lateral_atom(["R"], [180])[0]
    d8 = A.vertical_atom(["descend"], [8000])[0]
    z = np.array([[0, 0, 0, 0], [0, r270, 0, 0], [0, 0, d8, 0]])
    v = kinematic_rule(z, state, center)
    assert np.allclose(v[0], [128.6, 0, 0])                 # nothing said: keep going
    hdg = np.degrees(np.arctan2(v[1, :, 1], v[1, :, 0]))
    assert np.isclose(hdg[-1], 90.0, atol=0.5)               # 90 deg right turn completed
    alt = 10000 + np.cumsum(v[2, :, 2]) / 0.3048
    assert alt.min() >= 8000 - 1 and alt[-1] < 10000


def test_zpolicy_mirrors_messages():
    """'Not ours' must not tell the CTP whether the window maneuvers, and a MAINTAIN on
    its own must only come from a steady window, as in data/messages.py."""
    from types import SimpleNamespace
    from atc_forecast.training.data_v2 import ZPolicy
    rng = np.random.default_rng(0)
    n = 20000
    man = rng.random(n) < 0.8
    z_true = np.zeros((n, 4), np.int64)
    z_true[man, 1] = 10                                   # a lateral instruction
    steady = np.stack([~man, np.ones(n, bool), np.ones(n, bool)], 1)
    store = SimpleNamespace(z_true=z_true, is_man=man, steady=steady)
    z = ZPolicy(p_null=0.3).draw(store, np.arange(n), rng)
    m = z[:, 1:]
    null = z[:, 0] != 0
    assert (m[null] == A.NONE).all()
    assert abs(man[null].mean() - man.mean()) < 0.02      # independent of the window
    alone = ~null & ((m == A.MAINTAIN).sum(1) == 1) & ((m == A.NONE).sum(1) == 2)
    assert alone.any() and not man[alone].any()
    assert not (~null & (m == A.NONE).all(1)).any()        # an own message says something
    assert (m[man & ~null, 0] == 10).all()                 # maneuvers keep their atoms


def test_primitive_fit_and_decode_roundtrip():
    """A synthetic turn (delay, ramp, new rate, capture) is fitted back to its parameters, and
    the design-A decoder turns the parameters into the same velocities as the numpy rule."""
    import torch

    from atc_forecast.data.primitives import curve, fit_free, signals, velocities
    from atc_forecast.models.prim import PrimSpace

    true = np.array([[0.0, 5.0, 2.5, 6.0, 30.0]])                  # v0, d, r, tau, e
    y = curve(true)
    P, rms = fit_free(y)
    assert rms[0] < 0.5 and abs(P[0, 1] - 5) <= 1 and abs(P[0, 2] - 2.5) < 0.2
    flat = np.zeros((1, 60))
    center = np.array([[120.0, 0.0, 0.0]])
    v = velocities(y, flat, flat, center)
    psi, z, gs = signals(v, center)
    assert np.allclose(psi, y, atol=1e-4) and np.allclose(z, 0, atol=1e-6)
    Pall = np.stack([true[0], np.zeros(5), np.zeros(5)])[None].reshape(1, 15)
    Pall[0, 8] = Pall[0, 13] = 61.0                                  # no capture on the other axes
    prim = {"mu": np.zeros(21), "sd": np.ones(21), "res_comp": np.zeros((3, 2, 60)),
            "res_scale": np.ones(3), "centers": np.zeros((8, 4))}
    x = torch.tensor(np.concatenate([Pall, np.zeros((1, 6))], 1), dtype=torch.float32)
    V = PrimSpace(prim).decode(x, torch.tensor(center, dtype=torch.float32)).numpy()
    assert np.allclose(V, v, atol=1e-3)


def test_bank_decoder_matches_numpy():
    """With bank-angle targets, the torch decoder and the numpy rule give the same path, and a
    turn at constant bank turns faster as the aircraft slows down."""
    import torch

    from atc_forecast.data.primitives import bank_velocities
    from atc_forecast.models.prim import PrimSpace

    P = np.zeros((1, 3, 5))
    P[0, 0] = [0.0, 3.0, 25.0, 4.0, 61.0]                          # roll into 25 deg of bank
    P[0, 1] = [0.0, 0.0, 0.0, 0.0, 61.0]
    P[0, 2] = [0.0, 5.0, -1.0, 3.0, 61.0]                          # decelerate 1 kt/s
    center = np.array([[130.0, 0.0, 0.0]])
    v = bank_velocities(P, center)
    prim = {"mu": np.zeros(21), "sd": np.ones(21), "res_comp": np.zeros((3, 2, 60)),
            "res_scale": np.ones(3), "centers": np.zeros((8, 4)), "bank": np.float32(1)}
    x = torch.tensor(np.concatenate([P.reshape(1, 15), np.zeros((1, 6))], 1), dtype=torch.float32)
    V = PrimSpace(prim).decode(x, torch.tensor(center, dtype=torch.float32)).numpy()
    assert np.allclose(V, v, atol=1e-3)
    psi = np.degrees(np.unwrap(np.arctan2(v[0, :, 1], v[0, :, 0])))
    rate = np.diff(psi)
    assert rate[50] > rate[15] > 0
