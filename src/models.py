"""
Model implementations (pure NumPy; no FL framework, no sklearn inside the
training loop).

Model A: class-weighted logistic regression. 16 parameters
         (15 feature weights + 1 bias).

Model B: MLP 15 -> 16 -> 1 with ReLU hidden activation and sigmoid output.
         273 parameters (15*16 + 16 + 16 + 1).

Both models expose a flat parameter vector so the same federated aggregation
logic applies to either family.
"""

from __future__ import annotations

import numpy as np

from . import config


def sample_weights(y: np.ndarray) -> np.ndarray:
    """Positive-class weight 5.59, negative-class weight 1.0."""
    return np.where(y == 1.0, config.POSITIVE_CLASS_WEIGHT,
                    config.NEGATIVE_CLASS_WEIGHT)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(p: np.ndarray, y: np.ndarray,
                         weights: np.ndarray, eps: float = 1e-12) -> float:
    """Mean weighted binary cross entropy."""
    p = np.clip(p, eps, 1.0 - eps)
    bce = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(np.mean(weights * bce))


class LogisticRegression:
    """Class-weighted logistic regression using mini-batch SGD training."""

    def __init__(self, n_features: int = None):
        self.n_features = n_features if n_features is not None else len(
            config.FEATURE_COLUMNS)
        self.params = np.zeros(self.n_features + 1)

    # -- parameter management ------------------------------------------------
    def initialize(self, seed: int):
        rng = np.random.default_rng(seed)
        self.params = rng.normal(0.0, 0.01, size=self.n_features + 1)
        return self

    def get_parameters(self) -> np.ndarray:
        return self.params.copy()

    def set_parameters(self, flat: np.ndarray):
        flat = np.asarray(flat, dtype=float).reshape(-1)
        if flat.size != self.n_features + 1:
            raise ValueError(
                f"Expected {self.n_features + 1} parameters, got {flat.size}")
        self.params = flat.astype(np.float64).copy()

    @property
    def n_params(self) -> int:
        return self.n_features + 1

    # -- forward --------------------------------------------------------------
    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.params[:self.n_features] + self.params[self.n_features]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.predict_logits(X))

    def predict(self, X: np.ndarray, threshold: float = config.THRESHOLD) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # -- training losses ------------------------------------------------------
    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        p = self.predict_proba(X)
        return binary_cross_entropy(p, y, sample_weights(y))

    def gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Mean gradient of the weighted BCE with respect to the flat vector."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        p = self.predict_proba(X)
        sw = sample_weights(y)
        d = sw * (p - y)                      # (n,)
        g_w = X.T @ d / len(y)                # (d,)
        g_b = float(np.mean(d))
        return np.concatenate([g_w, [g_b]])
class MLP:
    """Two-layer MLP 15 -> 16 -> 1 with ReLU hidden, sigmoid output.

    Parameter layout (flat vector, 273 = 15*16 + 16 + 16 + 1):
      [W1 (15,16), b1 (16,), W2 (16,1), b2 (1,)]
    """

    def __init__(self, n_features: int = None, hidden: int = None):
        self.n_features = n_features if n_features is not None else len(
            config.FEATURE_COLUMNS)
        self.hidden = hidden if hidden is not None else config.MLP_HIDDEN_UNITS
        self.params = np.zeros(self.n_params)
        self._cache = None

    # -- parameter management ------------------------------------------------
    def initialize(self, seed: int):
        rng = np.random.default_rng(seed)
        a1 = np.sqrt(6.0 / (self.n_features + self.hidden))
        a2 = np.sqrt(6.0 / (self.hidden + 1))
        W1 = rng.uniform(-a1, a1, size=(self.n_features, self.hidden))
        W2 = rng.uniform(-a2, a2, size=(self.hidden, 1))
        b1 = np.zeros(self.hidden)
        b2 = np.zeros(1)
        self.params = _pack([W1, b1, W2, b2])
        return self

    def get_parameters(self) -> np.ndarray:
        return self.params.copy()

    def set_parameters(self, flat: np.ndarray):
        flat = np.asarray(flat, dtype=float).reshape(-1)
        if flat.size != self.n_params:
            raise ValueError(
                f"Expected {self.n_params} parameters, got {flat.size}")
        self.params = flat.astype(np.float64).copy()

    @property
    def n_params(self) -> int:
        return self.n_features * self.hidden + self.hidden + self.hidden + 1

    def _unpack(self, params: np.ndarray):
        split1 = self.n_features * self.hidden
        split2 = split1 + self.hidden
        W1 = params[:split1].reshape(self.n_features, self.hidden)
        b1 = params[split1:split2]
        W2 = params[split2:self.n_params - 1].reshape(self.hidden, 1)
        b2 = params[self.n_params - 1:]
        return W1, b1, W2, b2

    # -- forward --------------------------------------------------------------
    def forward(self, X: np.ndarray):
        W1, b1, W2, b2 = self._unpack(self.params)
        h1 = X @ W1 + b1
        a1 = np.maximum(h1, 0.0)              # ReLU
        z2 = a1 @ W2 + b2
        p = sigmoid(z2.reshape(-1))
        self._cache = (X, h1, a1, W2)
        return p

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(np.asarray(X, dtype=float))

    def predict(self, X: np.ndarray,
                threshold: float = config.THRESHOLD) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # -- training -------------------------------------------------------------
    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        p = self.predict_proba(X)
        return binary_cross_entropy(p, y, sample_weights(y))

    def gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Mean gradient of the weighted BCE w.r.t. the flat vector."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        p = self.forward(X)
        _, h1, a1, W2 = self._cache
        sw = sample_weights(y)
        dz = sw * (p - y) / len(y)            # (n,)
        dW2 = a1.T @ dz.reshape(-1, 1)        # (h,1)
        db2 = np.array([dz.sum()])
        da1 = dz.reshape(-1, 1) * W2.T        # (n,h)
        dh1 = da1 * (h1 > 0.0)
        # dz already carries the 1/len(y) batch-mean factor, so the
        # first-layer equations must NOT be normalized by len(y) again
        # (a finite-difference gradient check confirms W2/b2 and these
        # equations; dividing here would shrink dW1/db1 by 1/batch_size).
        dW1 = X.T @ dh1                       # (d,h)
        db1 = dh1.sum(axis=0)                 # (h,)
        return _pack([dW1, db1, dW2, db2])


def _pack(arrays) -> np.ndarray:
    return np.concatenate([a.reshape(-1) for a in arrays]).astype(np.float64)


def make_model(model_name: str, seed: int):
    """Factory returning an initialized model instance."""
    if model_name == config.MODEL_LOGISTIC:
        model = LogisticRegression()
    elif model_name == config.MODEL_MLP:
        model = MLP()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model.initialize(seed)
    return model