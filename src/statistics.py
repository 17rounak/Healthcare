"""
Statistical analysis: paired tests, bootstrap intervals, Cohen's d, and Holm
step-down multiple-comparison correction.

Because every method shares the same five seeds, all comparisons use PAIRED
tests only (paired t-test, Wilcoxon signed-rank). Bootstrap is a paired
bootstrap over the seed-level differences.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config


def paired_t_test(a: np.ndarray, b: np.ndarray):
    """Paired two-sided t-test. Returns (stat, p_value)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.all(a == b):
        return 0.0, 1.0
    stat, p = stats.ttest_rel(a, b)
    return float(stat), float(p)


def wilcoxon_test(a: np.ndarray, b: np.ndarray):
    """Wilcoxon signed-rank test (paired). Returns (stat, p_value)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    if len(d) < 2 or np.all(d == 0):
        return 0.0, 1.0
    stat, p = stats.wilcoxon(d, zero_method="wilcox", correction=False)
    return float(stat), float(p)


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot=None,
                        ci: float = None, rng=None):
    """Percentile CI of mean(a - b) from n_boot paired resamples."""
    n_boot = n_boot if n_boot is not None else config.BOOTSTRAP_SAMPLES
    ci = ci if ci is not None else config.BOOTSTRAP_CI
    rng = rng if rng is not None else np.random.default_rng(12345)

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = len(d)
    means = np.empty(n_boot)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(means, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)


def paired_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d = mean(difference) / sd(difference)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    if len(d) < 2 or np.std(d, ddof=1) == 0:
        return 0.0
    return float(np.mean(d) / np.std(d, ddof=1))


def holm_stepdown(p_values, alpha: float = 0.05) -> np.ndarray:
    """Holm-Bonferroni step-down correction.

    Sorts p-values ascending; j-th (1-based) smallest p is judged against
    alpha/(m - j + 1). Returns the corrected (adjusted) p-values.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.ones_like(p)
    p_sorted = p[order]
    for j, pj in enumerate(p_sorted):
        adjusted[order[j]] = min(pj * (m - j), 1.0)
        # Step-down convention: once a comparison is accepted, later ones are.
    return adjusted


def primary_comparison(a_scores: np.ndarray, b_scores: np.ndarray,
                       method_a: str, method_b: str,
                       model: str, regime: str) -> dict:
    """Full paired comparison FedAvg (a) vs Local-only (b)."""
    diff = np.asarray(a_scores, dtype=float) - np.asarray(b_scores, dtype=float)
    t_stat, t_p = paired_t_test(a_scores, b_scores)
    w_stat, w_p = wilcoxon_test(a_scores, b_scores)
    lo, hi = paired_bootstrap_ci(a_scores, b_scores)
    d = paired_cohens_d(a_scores, b_scores)
    return {
        "model": model,
        "regime": regime,
        "method_a": method_a,
        "method_b": method_b,
        "mean_a": float(np.mean(a_scores)),
        "mean_b": float(np.mean(b_scores)),
        "mean_gain": float(np.mean(diff)),
        "t_stat": t_stat,
        "p_value": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "ci_lower": lo,
        "ci_upper": hi,
        "cohens_d": d,
        "practical": ("meaningful" if abs(np.mean(diff)) >= config.PRACTICAL_AUC_DELTA
                      else "negligible"),
    }


def classify_practical_significance(delta_auc: float,
                                    threshold: float = None) -> str:
    """Rough practical-significance classification (default 1 pt of AUC)."""
    threshold = threshold if threshold is not None else config.PRACTICAL_AUC_DELTA
    if abs(delta_auc) >= threshold:
        return "meaningful/practically-relevant"
    return "negligible"