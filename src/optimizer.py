"""
Mini-batch SGD shared by every training regime.

Optimizer configuration is fully specified in config.py and is used verbatim
by centralized, local-only, FedAvg, FedProx and DP-FedAvg.
"""

from __future__ import annotations

import numpy as np

from . import config


def train_sgd(model, X: np.ndarray, y: np.ndarray, rng,
              epochs: int = None, learning_rate: float = None,
              batch_size: int = None, w_global: np.ndarray = None,
              mu: float = 0.0, track_losses=None):
    """Run local mini-batch SGD and return the trained parameter vector.

    model        : model object exposing gradient()/get_parameters()/
                   set_parameters()
    w_global     : if not None, the FedProx proximal anchor. The local
                   objective becomes L + (mu/2)*||w - w_global||^2 so the
                   gradient receives an extra mu*(w - w_global) term.
    """
    epochs = epochs if epochs is not None else config.LOCAL_EPOCHS
    learning_rate = learning_rate if learning_rate is not None else config.LEARNING_RATE
    batch_size = batch_size if batch_size is not None else config.BATCH_SIZE

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]

    params = model.get_parameters()
    if track_losses is not None:
        track_losses.append(model.loss(X, y))

    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            Xb = X[idx]
            yb = y[idx]
            model.set_parameters(params)
            grad = model.gradient(Xb, yb)
            if mu > 0.0 and w_global is not None:
                grad = grad + mu * (params - w_global)
            params = params - learning_rate * grad
        if track_losses is not None:
            model.set_parameters(params)
            track_losses.append(model.loss(X, y))

    model.set_parameters(params)
    return params