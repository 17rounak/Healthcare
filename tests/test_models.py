"""Model tests: parameter counts, forward pass, gradients."""
import numpy as np
import pytest

from src import config
from src.models import LogisticRegression, MLP, make_model
from src import optimizer


def _synthetic(n=256, d=15):
    rng = np.random.default_rng(123)
    X = rng.normal(size=(n, d))
    z = X[:, :3].sum(axis=1)
    p = 1 / (1 + np.exp(-z))
    y = (rng.uniform(size=n) < p).astype(float)
    return X, y


def test_logistic_parameter_count():
    m = LogisticRegression(config.EXPECTED_N_FEATURES)
    assert m.n_params == 16
    m.initialize(0)
    assert m.get_parameters().shape == (16,)


def test_mlp_parameter_count():
    m = MLP(config.EXPECTED_N_FEATURES, config.MLP_HIDDEN_UNITS)
    assert m.n_params == 273
    m.initialize(0)
    assert m.get_parameters().shape == (273,)


def test_logistic_forward():
    X, y = _synthetic()
    m = LogisticRegression(15).initialize(0)
    proba = m.predict_proba(X)
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()
    logits = m.predict_logits(X)
    pred = m.predict(X)
    assert pred.shape == (len(X),)
    assert set(np.unique(pred)) <= {0, 1}


def test_logistic_gradient_dimensions():
    X, y = _synthetic()
    m = LogisticRegression(15).initialize(0)
    g = m.gradient(X, y)
    assert g.shape == (16,)


def test_mlp_forward():
    X, y = _synthetic()
    m = MLP(15, 16).initialize(0)
    proba = m.predict_proba(X)
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert m.predict(X).shape == (len(X),)


def test_mlp_gradient_dimensions():
    X, y = _synthetic()
    m = MLP(15, 16).initialize(0)
    g = m.gradient(X, y)
    assert g.shape == (273,)


def test_gradient_check_logistic():
    """Numerical gradient check for the logistic model."""
    X, y = _synthetic(128)
    m = LogisticRegression(15).initialize(7)
    eps = 1e-6
    g = m.gradient(X, y)
    num = np.empty_like(g)
    p0 = m.get_parameters()
    for i in range(m.n_params):
        p_plus = p0.copy()
        p_plus[i] += eps
        m.set_parameters(p_plus)
        l_plus = m.loss(X, y)
        p_minus = p0.copy()
        p_minus[i] -= eps
        m.set_parameters(p_minus)
        l_minus = m.loss(X, y)
        num[i] = (l_plus - l_minus) / (2 * eps)
    m.set_parameters(p0)
    assert np.allclose(num, g, atol=1e-3)


def test_mlp_gradient_check():
    """Finite-difference gradient check for ALL MLP parameter blocks.

    Checks W1, b1, W2 and b2 against central finite differences of the
    weighted BCE loss.  This guards against re-introducing the double
    batch-normalization bug in the first-layer equations: dz already carries
    the 1/len(y) mean factor, so dW1 = X.T @ dh1 and db1 = dh1.sum(axis=0)
    must NOT be normalized by len(y) again (the old bug made W1/b1 gradients
    exactly 1/batch_size of the true gradient).
    """
    # Small deterministic batch: cheap enough to differentiate all 273 params.
    rng = np.random.default_rng(2024)
    n, d = 8, 15
    X = rng.normal(size=(n, d))
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    m = MLP(d, 16).initialize(7)

    eps = 1e-6
    g = m.gradient(X, y)
    num = np.empty_like(g)
    p0 = m.get_parameters()
    for i in range(m.n_params):
        p_plus = p0.copy()
        p_plus[i] += eps
        m.set_parameters(p_plus)
        l_plus = m.loss(X, y)
        p_minus = p0.copy()
        p_minus[i] -= eps
        m.set_parameters(p_minus)
        l_minus = m.loss(X, y)
        num[i] = (l_plus - l_minus) / (2 * eps)
    m.set_parameters(p0)

    assert np.allclose(num, g, rtol=1e-4, atol=1e-6)


def test_sgd_trains_logistic():
    X, y = _synthetic(512)
    m = LogisticRegression(15).initialize(0)
    rng = np.random.default_rng(0)
    loss0 = m.loss(X, y)
    optimizer.train_sgd(m, X, y, rng, epochs=10)
    assert m.loss(X, y) < loss0