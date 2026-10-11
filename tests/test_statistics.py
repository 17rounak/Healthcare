"""Statistical test utilities."""
import numpy as np
import pytest

from src import statistics as st


def test_paired_t_test():
    # identical data => p=1, stat=0
    t, p = st.paired_t_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert p == 1.0
    # shifted data with low n gives a small-ish p
    a = np.array([0.50, 0.55, 0.60, 0.65, 0.70])
    b = a - 0.08
    t, p = st.paired_t_test(a, b)
    assert p < 0.05


def test_wilcoxon_test():
    t, p = st.wilcoxon_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert p == 1.0


def test_bootstrap_ci():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, 20)
    b = np.zeros(20)
    lo, hi = st.paired_bootstrap_ci(a, b, n_boot=500, rng=rng)
    assert lo < hi
    assert lo < 0.5 < hi


def test_cohens_d():
    a = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    b = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    assert st.paired_cohens_d(a, b) == 0.0
    # non-degenerate shift (constant shifts leave sd(diff)==0, so add noise)
    shifted = a + 0.1 + np.linspace(-0.01, 0.01, len(a))
    d = st.paired_cohens_d(shifted, a)
    assert d > 0


def test_holm_stepdown():
    p = np.array([0.001, 0.02, 0.04, 0.5, 0.9])
    adj = st.holm_stepdown(p)
    assert adj.min() <= p.min() * len(p)
    # the first (smallest) p is multiplied by m
    assert np.isclose(adj[np.argmin(p)], p.min() * len(p) + 1e-15) or \
        np.isclose(adj[np.argmin(p)], min(p.min() * len(p), 1.0))


def test_practical_classification():
    assert st.classify_practical_significance(0.005) == "negligible"
    assert st.classify_practical_significance(0.012) == \
        "meaningful/practically-relevant"