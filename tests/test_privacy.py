"""DP-FedAvg and privacy accounting tests."""
import numpy as np
import pytest

from src import config
from src import differential_privacy as dp
from src import privacy_accounting as pa


def test_clipping_bounds_norm():
    rng = np.random.default_rng(0)
    delta = rng.normal(size=50) * 10
    clipped, norm = dp.clip_update(delta, clip_norm=1.0)
    assert np.linalg.norm(clipped) <= 1.0 + 1e-12
    delta2 = rng.normal(size=50) * 0.01
    c2, n2 = dp.clip_update(delta2, clip_norm=1.0)
    assert np.allclose(c2, delta2)


def test_noise_shape():
    rng = np.random.default_rng(1)
    vec = np.zeros(16)
    noisy = dp.add_gaussian_noise(vec, scale=0.1, rng=rng)
    assert noisy.shape == (16,)
    # scale matches empirical std roughly
    samples = np.array([dp.add_gaussian_noise(np.zeros(1000), 0.5,
                                              np.random.default_rng(s)).std()
                        for s in range(200)])
    assert abs(samples.mean() - 0.5) < 0.05


def test_dp_client_aggregate():
    rng = np.random.default_rng(3)
    deltas = [np.full(16, 2.0), np.full(16, -1.0), np.full(16, 0.5),
              np.full(16, 3.0), np.full(16, -0.2)]
    weights = np.ones(5) / 5
    agg, scale = dp.dp_client_aggregate(deltas, weights, sigma=0.5,
                                        clip_norm=1.0, rng=rng)
    assert agg.shape == (16,)
    # noise scale is sigma * C (aggregate noise multiplier convention)
    assert np.isclose(scale, 0.5 * 1.0)


def test_dp_noise_scale_is_sigma_times_clip_norm():
    """The mechanism and the RDP accountant must share one definition of sigma.

    privacy_accounting treats sigma as the Gaussian noise multiplier relative
    to the clip bound C (per-round RDP = alpha / (2 sigma^2) for a release
    with L2 sensitivity <= C).  The mechanism must therefore add noise with
    std sigma * C to the released aggregate -- NOT sigma * C / K.  This test
    fails if the old 1/K factor is reintroduced.
    """
    for sigma in config.DP_SIGMAS:
        rng = np.random.default_rng(0)
        deltas = [np.zeros(16) for _ in range(config.N_SILOS)]
        weights = np.ones(config.N_SILOS) / config.N_SILOS
        _, scale = dp.dp_client_aggregate(deltas, weights, sigma=sigma,
                                          clip_norm=config.DP_CLIP_NORM,
                                          rng=rng)
        assert np.isclose(scale, sigma * config.DP_CLIP_NORM)
        # the accountant's sigma equals noise std / sensitivity bound C
        assert np.isclose(scale / config.DP_CLIP_NORM, sigma)


def test_dp_noise_empirical_std_matches_accounting():
    """Empirical noise std on the release ~= sigma * C.

    With all-zero client deltas the release is pure noise, so the empirical
    spread of the released aggregate directly estimates the mechanism's
    noise scale, which must equal the sigma * C scale the accountant assumes.
    """
    sigma, C, dim, reps = 0.5, config.DP_CLIP_NORM, 8, 400
    deltas = [np.zeros(dim) for _ in range(config.N_SILOS)]
    weights = np.ones(config.N_SILOS) / config.N_SILOS
    samples = np.empty((reps, dim))
    for r in range(reps):
        noisy, scale = dp.dp_client_aggregate(deltas, weights, sigma=sigma,
                                              clip_norm=C,
                                              rng=np.random.default_rng(r))
        samples[r] = noisy
        assert np.isclose(scale, sigma * C)
    # 400 reps x 8 dims -> std estimate accurate to a few percent
    assert abs(samples.std() - sigma * C) < 0.05 * sigma * C


def test_epsilon_finite_and_monotone():
    eps = [pa.compute_epsilon(s) for s in [0.1, 0.25, 0.5, 1.0, 2.0]]
    assert all(np.isfinite(e) for e in eps)
    assert eps == sorted(eps, reverse=True)  # larger sigma -> smaller eps


def test_epsilon_reasonable_magnitude():
    """Sanity bounds consistent with the paper's formula."""
    e = pa.compute_epsilon(1.0)
    assert e < 500.0  # well below the sigma=0.1 value (~2000+)
    e2 = pa.compute_epsilon(0.1)
    assert e2 > 1000.0


def test_rdp_formula():
    alpha = np.array([2.0])
    sigma = 1.0
    rdp = pa.rdp_compose(alpha, sigma, rounds=40)
    assert np.isclose(rdp[0], 40 * 2 / 2.0)
    eps = pa.convert_to_epsilon(rdp, alpha, delta=1e-5)
    assert np.isclose(eps[0], 40 * 2 / 2.0 + np.log(1e5) / 1.0)