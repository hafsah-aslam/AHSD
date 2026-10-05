"""S2-dir (EXPLORATORY) building blocks."""

import numpy as np

from nids import s2dir, spectral


def test_whitener_gives_unit_covariance_without_shrinkage():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6))
    z = rng.normal(size=(4000, 8, 6)) @ A.T
    W = s2dir.whitener(z, shrinkage=0.0)
    u = z.reshape(-1, 6) @ W.T
    assert np.allclose(np.cov(u, rowvar=False), np.eye(6), atol=1e-6)


def test_channel_power_parseval_per_channel():
    rng = np.random.default_rng(1)
    u = rng.normal(size=(50, 32, 4))
    assert np.allclose(s2dir.channel_power(u).sum(0), (u ** 2).mean((0, 1)))
    # channel average equals S2's window_power averaged over windows
    assert np.allclose(s2dir.channel_power(u).mean(1), spectral.window_power(u).mean(0))


def test_s2dir_zero_for_same_distribution_positive_for_shift():
    rng = np.random.default_rng(2)
    g2 = np.ones(17)
    E0 = np.zeros(5)
    zb_train = rng.normal(size=(800, 32, 5))
    zb = rng.normal(size=(800, 32, 5))
    same = s2dir.s2dir_raw(rng.normal(size=(800, 32, 5)), zb, zb_train, E0, g2, seed=0)
    shifted = s2dir.s2dir_raw(rng.normal(size=(800, 32, 5)) + 1.0, zb, zb_train, E0, g2, seed=0)
    assert shifted["raw"] > 50 * same["raw"]
    assert same["Q_benign_halves"] > 0 and same["Q_benign_halves"] < shifted["raw"]
