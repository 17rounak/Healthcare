"""
Privacy-respecting preprocessing.

1. Missing values are imputed with the variable median computed from the
   *training* partition only (no test leakage). Median is a non-linear
   statistic so it is intentionally computed on pooled training rows; the
   standardization below is where the paper's secure-aggregation-compatible
   sufficient-statistics protocol applies.

2. Feature standardization is computed from per-silo sufficient statistics
   (n, sum, sum of squares) that are aggregated across silos. The aggregate
   is mathematically identical to pooling the raw rows, but the raw records
   themselves never need to leave the silos. Actual cryptographic secure
   aggregation is NOT implemented - the protocol here is conceptually
   compatible with it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------
def compute_medians(df: pd.DataFrame, columns=None) -> dict:
    """Median per column, computed from the provided (training) data only."""
    columns = columns if columns is not None else config.FEATURE_COLUMNS
    medians = {}
    for col in columns:
        vals = df[col].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        medians[col] = float(np.median(vals)) if len(vals) else 0.0
    return medians


def impute_with_medians(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    """Fill missing values with previously computed medians."""
    out = df.copy()
    for col in config.FEATURE_COLUMNS:
        if col in medians:
            out[col] = out[col].fillna(medians[col])
    return out


# ---------------------------------------------------------------------------
# Sufficient-statistics standardization (secure-aggregation-compatible)
# ---------------------------------------------------------------------------
def compute_local_statistics(X: np.ndarray) -> dict:
    """n, sum, sum-of-squares per feature for one data partition."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    return {
        "n": n,
        "sum": X.sum(axis=0),
        "sum_sq": np.einsum("ij,ij->j", X, X) if n else np.zeros(X.shape[1]),
    }


def aggregate_statistics(stats_list) -> dict:
    """Combine per-silo sufficient statistics into global statistics."""
    n = sum(s["n"] for s in stats_list)
    s = sum((s["sum"] for s in stats_list), np.zeros_like(stats_list[0]["sum"]))
    s2 = sum((s["sum_sq"] for s in stats_list), np.zeros_like(stats_list[0]["sum_sq"]))
    return {"n": n, "sum": s, "sum_sq": s2}


def derive_global_standardization(stats: dict) -> dict:
    """Population-style mean and standard deviation from sufficient stats."""
    n = stats["n"]
    mean = stats["sum"] / n
    # Population variance (divide by n) to keep client-side estimates stable;
    # matches the convention used by the paper's aggregation description.
    var = np.maximum(stats["sum_sq"] / n - mean ** 2, 0.0)
    std = np.sqrt(var)
    # Guard against zero variance (constant features).
    std[std < 1e-12] = 1.0
    return {"mean": mean.astype(np.float64), "std": std.astype(np.float64)}


def standardize(X: np.ndarray, global_stats: dict) -> np.ndarray:
    """Apply standardization derived from aggregated training statistics."""
    X = np.asarray(X, dtype=float)
    return (X - global_stats["mean"]) / global_stats["std"]


def prepare_features(df: pd.DataFrame, medians: dict, global_stats: dict):
    """Vectorised feature matrix + target from an imputed dataframe."""
    X = df[config.FEATURE_COLUMNS].to_numpy(dtype=float)
    X = standardize(X, global_stats)
    y = df[config.TARGET_COLUMN].to_numpy(dtype=float)
    return X, y


def build_preprocessing(train_df: pd.DataFrame):
    """Compute the training-only preprocessing state once.

    Returns (medians, global_stats, standardization_scalars).
    """
    medians = compute_medians(train_df)
    train_imputed = impute_with_medians(train_df, medians)
    X = train_imputed[config.FEATURE_COLUMNS].to_numpy(dtype=float)
    # For the pooling-free path we compute global stats directly from the whole
    # training matrix here; the silo-level sufficient-statistics path is used
    # inside the partitioning/experiment layer (identical result by linearity).
    stats = compute_local_statistics(X)
    gstats = derive_global_standardization(stats)
    return medians, gstats