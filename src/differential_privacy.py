"""
Client-level differentially private FedAvg (DP-FedAvg).

Noise convention (ONE definition of sigma, shared by this mechanism and by
the RDP accountant in src/privacy_accounting.py):

    sigma is the aggregate Gaussian noise multiplier relative to the clip
    bound C.  The released quantity is the sample-weighted average of the
    clipped client deltas,

        agg = sum_i (n_i / N) * delta_i,        ||delta_i||_2 <= C,

    and the mechanism outputs agg + z with z ~ N(0, (sigma * C)^2) drawn
    i.i.d. per coordinate.

Consistency with the accountant: privacy_accounting computes, per round and
for any Renyi order alpha > 1,

    RDP(alpha) = alpha / (2 * sigma^2),

which is the RDP of the Gaussian mechanism whose noise standard deviation is
sigma times the L2 sensitivity of the released quantity.  Each client
contributes w_i * delta_i with ||delta_i|| <= C and w_i = n_i / N <= 1, so
the client-level (add/remove-one-client) L2 sensitivity of the release is at
most C, and adding N(0, (sigma*C)^2) noise realizes exactly the mechanism
that RDP(alpha) = alpha / (2 sigma^2) accounts for (a conservative bound
whenever every w_i < 1).

NOTE (convention fix): an earlier revision added noise with scale
sigma * C / K (K = number of silos).  That injects K times less noise than
the sigma-relative-to-C convention above; under that scaling the realized
per-round RDP was K^2 = 25x the accounted RDP.  A 1/K factor on the noise is
only correct when sigma is calibrated to the summed release sum_i delta_i
(sensitivity C) with the 1/K normalization applied afterwards as
post-processing; it is inconsistent when -- as here and in the accountant --
sigma is defined relative to C on the released weighted-average aggregate
itself.
"""

from __future__ import annotations

import numpy as np

from . import config


def clip_update(delta: np.ndarray, clip_norm: float = None) -> np.ndarray:
    """L2-norm clip a client delta."""
    clip_norm = clip_norm if clip_norm is not None else config.DP_CLIP_NORM
    delta = np.asarray(delta, dtype=float)
    norm = float(np.linalg.norm(delta))
    if norm > clip_norm and norm > 0.0:
        return delta * (clip_norm / norm), norm
    return delta.copy(), norm


def add_gaussian_noise(vector: np.ndarray, scale: float, rng) -> np.ndarray:
    """Add isotropic Gaussian noise with the given standard deviation."""
    return vector + rng.normal(0.0, scale, size=vector.shape)


def dp_client_aggregate(clipped_deltas, weights: np.ndarray, sigma: float,
                        clip_norm: float, rng):
    """Weighted average of clipped deltas + Gaussian noise N(0, (sigma*C)^2).

    sigma is the aggregate Gaussian noise multiplier relative to the clip
    bound C -- the same sigma the RDP accountant uses -- so the noise scale
    on the released aggregate is sigma * C (NOT sigma * C / K; see the
    module docstring for the sensitivity argument).
    """
    clip_norm = clip_norm if clip_norm is not None else config.DP_CLIP_NORM
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    agg = np.zeros_like(clipped_deltas[0], dtype=float)
    for delta, w in zip(clipped_deltas, weights):
        agg = agg + w * delta
    noise_scale = sigma * clip_norm
    agg = add_gaussian_noise(agg, noise_scale, rng)
    return agg, noise_scale