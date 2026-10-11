"""
Data loading and reproducible stratified train/test splitting for the
Framingham Heart Study dataset.

The key invariant: the train/test split is stratified on the target, driven
only by the experiment seed, and identical across every regime / model /
strategy for a given seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_framingham(path=None) -> pd.DataFrame:
    """Load the raw Framingham CSV and return a clean dataframe.

    The raw file uses bare carriage-return line separators, so we read it as
    text and re-parse to be robust to platform quirks.
    """
    path = path if path is not None else config.DATASET_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Expected the 4,240-record Framingham Heart Study CSV."
        )

    with open(path, "r", newline="", encoding="utf-8-sig") as fh:
        text = fh.read()

    # Normalise the various line-ending conventions the file is distributed in.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    import io

    df = pd.read_csv(io.StringIO(text), na_values=config.MISSING_TOKENS)

    # Keep exactly the documented columns.
    expected = config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing columns: {missing_cols}")
    df = df[expected].copy()

    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """Structural checks; returns a dict of facts for reporting."""
    n_rows = len(df)
    n_features = len(config.FEATURE_COLUMNS)
    positives = int(df[config.TARGET_COLUMN].sum())
    prevalence = positives / n_rows

    checks = {
        "n_rows": n_rows,
        "n_features": n_features,
        "n_positives": positives,
        "prevalence": prevalence,
        "has_target": config.TARGET_COLUMN in df.columns,
        "no_id_column": not any(
            c.lower() in {"id", "customerid", "row", "index"}
            for c in df.columns
        ),
        "missing_counts": df.isna().sum().to_dict(),
        "exact_features": list(df.columns),
    }
    return checks


def make_train_test_split(df: pd.DataFrame, seed: int, test_fraction=None):
    """Stratified, reproducible split.

    Returns (train_df, test_df). Stratification uses the target column.
    """
    test_fraction = test_fraction if test_fraction is not None else config.TEST_FRACTION
    rng = np.random.default_rng(seed)

    y = df[config.TARGET_COLUMN].to_numpy()
    train_idx = np.zeros(len(df), dtype=bool)
    for cls in np.unique(y):
        cls_pos = np.flatnonzero(y == cls)
        n_test = int(round(len(cls_pos) * test_fraction))
        # Guarantee a fixed, seed-controlled selection of test rows.
        perm = rng.permutation(len(cls_pos))
        test_rows = cls_pos[perm[:n_test]]
        train_idx[test_rows] = True

    test_mask = train_idx
    return df.loc[~test_mask].reset_index(drop=True), df.loc[test_mask].reset_index(drop=True)