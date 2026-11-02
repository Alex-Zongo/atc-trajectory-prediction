"""Scores on nested (K interpretations x S futures) samples from a known mixture: the energy
score must be unbiased and the calibration checks must read their nominal values when the
truth is drawn from the same mixture."""
import numpy as np

from atc_forecast.evaluation.metrics_v2 import coverage, coverage_quantile, energy, nested, rank

K, S, B, T = 12, 6, 4000, 5


def mixture(rng, n):
    """[B, n, T, 3]: a bimodal forecast, two well separated interpretations."""
    z = rng.choice([-3.0, 3.0], size=(B, n, 1, 1))
    return z + rng.normal(size=(B, n, T, 3))


def nested_samples(rng):
    z = rng.choice([-3.0, 3.0], size=(B, K, 1, 1, 1))                 # one per interpretation
    X = z + rng.normal(size=(B, K, S, T, 3))
    return X.reshape(B, K * S, T, 3)


def test_energy_nested_is_unbiased():
    rng = np.random.default_rng(0)
    y = mixture(rng, 1)[:, 0]
    ref = energy(mixture(rng, K * S), y).mean()                   # independent samples
    X = nested_samples(rng)
    unbiased = energy(X, y, nested(K, S)).mean()
    pooled = energy(X, y).mean()                                  # all pairs: biased high
    assert abs(unbiased - ref) < 0.02 * ref
    assert pooled - ref > 0.03 * ref


def test_calibration_nominal_for_a_calibrated_forecast():
    rng = np.random.default_rng(1)
    y = mixture(rng, 1)[:, 0]
    X = nested_samples(rng)
    g = nested(K, S)
    c50, c90 = coverage(X, y, groups=g).mean((1, 2))
    assert abs(c50 - 0.5) < 0.02 and abs(c90 - 0.9) < 0.015
    out10 = (rank(X, y, g) > 0.9).mean()
    assert abs(out10 - 0.1) < 0.015
    # the quantile definition reads below nominal even for independent samples
    q90 = coverage_quantile(mixture(rng, K * S), y)[1].mean()
    assert abs(q90 - 0.9 * (K * S - 1) / (K * S + 1)) < 0.015


def test_overconfident_forecast_is_flagged():
    rng = np.random.default_rng(2)
    y = mixture(rng, 1)[:, 0] * 1.5                                # truth more spread out
    X = nested_samples(rng)
    g = nested(K, S)
    assert coverage(X, y, groups=g)[1].mean() < 0.85
    assert (rank(X, y, g) > 0.9).mean() > 0.15
