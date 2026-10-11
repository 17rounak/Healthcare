"""
Renyi differential privacy (RDP) accounting for the Gaussian mechanism.

The paper's stated accounting uses, for any Renyi order alpha > 1:

    RDP(alpha)   = T * alpha / (2 * sigma^2)

and converts to epsilon via

    epsilon(alpha) = T*alpha/(2*sigma^2) + log(1/delta)/(alpha - 1)

We search a fine grid of alpha values and take the minimum epsilon.  This is
the exact formula implemented here; grid discretization is the only source of
numerical inexactness.  T = 40 communication rounds, full participation
(K = 5, no subsampling amplification), delta = 1e-5.
"""

from __future__ import annotations

import numpy as np

from . import config


def rdp_epsilon_per_round(alpha: np.ndarray, sigma: float) -> np.ndarray:
    """RDP epsilon for one Gaussian mechanism application (one round)."""
    return alpha / (2.0 * sigma ** 2)


def rdp_compose(alpha: np.ndarray, sigma: float, rounds: int = None) -> np.ndarray:
    """Composed RDP across all communication rounds (pure composition)."""
    rounds = rounds if rounds is not None else config.ROUNDS
    return rounds * alpha / (2.0 * sigma ** 2)


def convert_to_epsilon(rdp: np.ndarray, alpha: np.ndarray,
                       delta: float = None) -> np.ndarray:
    delta = delta if delta is not None else config.DP_DELTA
    return rdp + np.log(1.0 / delta) / (alpha - 1.0)


def compute_epsilon(sigma: float, delta: float = None, rounds: int = None,
                    alpha_grid: np.ndarray = None) -> float:
    """Minimize epsilon over the alpha grid.

    The formula is minimized where alpha is close to 1 (e.g. alpha ~= 1.08
    for sigma = 0.1), so the grid is log-spaced to resolve that region.
    """
    delta = delta if delta is not None else config.DP_DELTA
    rounds = rounds if rounds is not None else config.ROUNDS
    if alpha_grid is None:
        alpha_grid = np.exp(np.linspace(
            np.log(config.RDP_ALPHA_MIN),
            np.log(config.RDP_ALPHA_MAX),
            config.RDP_N_ALPHAS))
    alpha = alpha_grid[alpha_grid > 1.0]
    rdp = rdp_compose(alpha, sigma, rounds)
    eps = convert_to_epsilon(rdp, alpha, delta)
    return float(np.min(eps))


def epsilon_table(sigmas=None, delta: float = None) -> list:
    """[(sigma, epsilon, delta), ...] for all configured noise multipliers."""
    sigmas = sigmas if sigmas is not None else config.DP_SIGMAS
    delta = delta if delta is not None else config.DP_DELTA
    return [(s, compute_epsilon(s, delta), delta) for s in sigmas]


def best_alpha(sigma: float, delta: float = None) -> float:
    """Return the alpha that achieves the minimum epsilon (diagnostics)."""
    delta = delta if delta is not None else config.DP_DELTA
    alpha_grid = np.exp(np.linspace(np.log(config.RDP_ALPHA_MIN),
                                    np.log(config.RDP_ALPHA_MAX),
                                    config.RDP_N_ALPHAS))
    alpha = alpha_grid[alpha_grid > 1.0]
    eps = convert_to_epsilon(rdp_compose(alpha, sigma), alpha, delta)
    return float(alpha[np.argmin(eps)])