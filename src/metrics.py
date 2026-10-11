"""
Evaluation metrics.

For every trained model we compute accuracy, precision, recall, F1,
ROC-AUC and PR-AUC (average-precision) on the shared global test set.
Threshold-based metrics use threshold = 0.5.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)

from . import config


def evaluate_model(model, X: np.ndarray, y: np.ndarray,
                   threshold: float = config.THRESHOLD) -> dict:
    """Full metric panel for a trained model on (X, y)."""
    y = np.asarray(y, dtype=float).astype(int)
    proba = model.predict_proba(X)
    pred = (proba >= threshold).astype(int)

    def _safe(fn):
        try:
            return float(fn())
        except (ValueError, ZeroDivisionError):
            return float("nan")

    return {
        "accuracy": _safe(lambda: accuracy_score(y, pred)),
        "precision": _safe(lambda: precision_score(y, pred, zero_division=0)),
        "recall": _safe(lambda: recall_score(y, pred, zero_division=0)),
        "f1": _safe(lambda: f1_score(y, pred, zero_division=0)),
        "roc_auc": _safe(lambda: roc_auc_score(y, proba)),
        "pr_auc": _safe(lambda: average_precision_score(y, proba)),
    }


def mean_metrics(metric_list) -> dict:
    """Element-wise mean and std across a list of metric dicts."""
    keys = metric_list[0].keys() if metric_list else []
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in metric_list], dtype=float)
        vals = vals[~np.isnan(vals)]
        out[f"{k}_mean"] = float(vals.mean()) if len(vals) else float("nan")
        out[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return out