"""
Silo construction for the three heterogeneity regimes.

Regime A (IID):          uniform random assignment; the control condition.
Regime B (label skew):   per-class Dirichlet(alpha=0.5) proportions; produces
                         simultaneous prevalence and quantity skew.
Regime C (covariate):    contiguous age bands after sorting by patient age;
                         clinically motivated covariate heterogeneity.

All partitions preserve every training record exactly once, never overlap,
and are reproducible given (seed, regime). Partitions are cached so that all
models and strategies share the exact same partition for a given seed/regime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _split_sizes(n_total: int, k: int) -> list:
    """Balanced sizes for n_total records across k silos.

    3392 -> [679, 679, 678, 678, 678]
    """
    base = n_total // k
    rem = n_total % k
    sizes = [base + 1 if i < rem else base for i in range(k)]
    return sizes


def _balanced_assignment_sizes(n_total: int, k: int):
    """Assign blocks of balanced size given the (seed, n_total)."""
    sizes = _split_sizes(n_total, k)
    boundaries = np.cumsum([0] + sizes)
    return sizes, boundaries


def partition_iid(train_indices: np.ndarray, seed: int, k: int = None):
    """Regime A: uniform random assignment."""
    k = k if k is not None else config.N_SILOS
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_indices))
    sizes, boundaries = _balanced_assignment_sizes(len(train_indices), k)
    silos = []
    for i in range(k):
        silos.append(train_indices[perm[boundaries[i]:boundaries[i + 1]]])
    return silos


def partition_dirichlet(train_indices: np.ndarray, y: np.ndarray, seed: int,
                        alpha: float = None, k: int = None):
    """Regime B: Dirichlet-based per-class label skew.

    For each class the proportion of its samples going to each silo is drawn
    from Dirichlet(alpha * ones(k)). Every sample is assigned exactly once.
    """
    alpha = alpha if alpha is not None else config.DIRICHLET_ALPHA
    k = k if k is not None else config.N_SILOS
    rng = np.random.default_rng(seed)

    classes = np.unique(y)
    assignment = np.full(len(train_indices), -1, dtype=int)

    for cls in classes:
        cls_pos = np.flatnonzero(y == cls)
        n_c = len(cls_pos)
        if n_c == 0:
            continue
        prop = rng.dirichlet(np.full(k, alpha)).astype(np.float64)
        counts = np.floor(prop * n_c).astype(int)
        # Distribute the remainder by largest fractional part.
        frac = prop * n_c - counts
        remainder = n_c - counts.sum()
        order = np.argsort(-frac)
        for r in range(int(remainder)):
            counts[order[r % k]] += 1

        perm = rng.permutation(n_c)
        cursor = 0
        for silo in range(k):
            if counts[silo] == 0:
                continue
            block = cls_pos[perm[cursor:cursor + counts[silo]]]
            assignment[block] = silo
            cursor += counts[silo]

    silos = [train_indices[assignment == i] for i in range(k)]
    return silos


def partition_covariate(train_indices: np.ndarray, age: np.ndarray, seed: int,
                        k: int = None):
    """Regime C: contiguous age bands (youngest -> oldest)."""
    del seed  # deterministic given the age ordering
    k = k if k is not None else config.N_SILOS
    order = np.argsort(age, kind="stable")
    sizes, boundaries = _balanced_assignment_sizes(len(train_indices), k)
    silos = []
    for i in range(k):
        block = order[boundaries[i]:boundaries[i + 1]]
        silos.append(train_indices[block])
    return silos


# ---------------------------------------------------------------------------
# Cache: partitions are identical for every model/strategy at a given seed.
# ---------------------------------------------------------------------------
_PARTITION_CACHE = {}


def make_partition(train_df, regime: str, seed: int, medians: dict):
    """Return cached list of numpy index arrays into train_df for the regime."""
    key = (regime, seed)
    if key in _PARTITION_CACHE:
        return _PARTITION_CACHE[key]

    train_indices = np.arange(len(train_df))
    y = train_df[config.TARGET_COLUMN].to_numpy(dtype=float)

    if regime == config.REGIME_IID:
        silos = partition_iid(train_indices, seed)
    elif regime == config.REGIME_LABEL_SKEW:
        silos = partition_dirichlet(train_indices, y, seed)
    elif regime == config.REGIME_COVARIATE_SKEW:
        age = train_df["age"].to_numpy(dtype=float)
        # Impute age with its median if a missing (NA) value sneaks in.
        if np.any(np.isnan(age)):
            age[np.isnan(age)] = medians["age"]
        silos = partition_covariate(train_indices, age, seed)
    else:
        raise ValueError(f"Unknown regime: {regime}")

    _PARTITION_CACHE[key] = silos
    return silos


def describe_partition(train_df, silos: list) -> pd.DataFrame:
    """Heterogeneity summary: counts, CHD prevalence, mean age, age z-score.

    Returns a pandas DataFrame with one row per silo.
    """
    rows = []
    all_ages = train_df["age"].to_numpy(dtype=float)
    age_mean = np.nanmean(all_ages)
    age_std = np.nanstd(all_ages)
    for i, idx in enumerate(silos):
        sub = train_df.iloc[idx]
        n = len(idx)
        pos = int(sub[config.TARGET_COLUMN].sum())
        prev = pos / n if n else float("nan")
        mean_age = float(sub["age"].mean()) if n else float("nan")
        if n > 1:
            z = (mean_age - age_mean) / max(age_std, 1e-12)
        else:
            z = float("nan")
        rows.append({
            "silo": i + 1,
            "n_records": n,
            "chd_positives": pos,
            "chd_prevalence": prev,
            "mean_age": mean_age,
            "age_z": z,
        })
    return pd.DataFrame(rows)