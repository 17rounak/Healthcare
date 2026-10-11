"""
Result tables (CSV outputs for every table described in the paper).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Table 1 – Comparative positioning (static documentation data)
# ---------------------------------------------------------------------------
TABLE1_ROWS = [
    {"study": "McMahan et al. (2017), FedAvg", "scale": "Benchmark image + language",
     "data_kind": "CIFAR / Shakespeare", "privacy": "None",
     "heterogeneity": "Non-IID partitions", "notes": "Introduced FedAvg"},
    {"study": "Li et al. (2020), FedProx", "scale": "Benchmark + synthetic",
     "data_kind": "Synthetic convex", "privacy": "None",
     "heterogeneity": "Controlled non-IID", "notes": "Proximal term mu"},
    {"study": "McMahan et al. (2018), DP-FedAvg", "scale": "Benchmark",
     "data_kind": "Image + language", "privacy": "Client-level DP",
     "heterogeneity": "Non-IID", "notes": "Clipping + Gaussian noise"},
    {"study": "Rieke et al. (2020)", "scale": "Healthcare survey",
     "data_kind": "Review / position", "privacy": "Discussion",
     "heterogeneity": "Clinical", "notes": "FL in medical imaging"},
    {"study": "Ayed et al. (2020)", "scale": "Small healthcare",
     "data_kind": "Stroke + open datasets", "privacy": "None",
     "heterogeneity": "Limited", "notes": "FedAvg on clinical data"},
    {"study": "This work", "scale": "5 silos; 4,240 adult participants",
     "data_kind": "Framingham Heart Study", "privacy": "DP-FedAvg client-level",
     "heterogeneity": "IID / Dirichlet label skew / age-band",
     "notes": "Controlled evaluation; paired statistics; no FL framework"},
]


def write_table1():
    df = pd.DataFrame(TABLE1_ROWS)
    df.to_csv(config.TABLES_DIR / "table1_comparative_positioning.csv",
              index=False)
    return df


# ---------------------------------------------------------------------------
# Table 2 – Predictor variables and descriptive statistics
# ---------------------------------------------------------------------------
def write_table2(df_raw: pd.DataFrame):
    X = df_raw[config.FEATURE_COLUMNS]
    rows = []
    for col in config.FEATURE_COLUMNS:
        s = X[col].dropna()
        ftype = ("binary/categorical" if X[col].nunique() <= 5
                 else ("integer" if np.allclose(s % 1, 0) else "continuous"))
        rows.append({
            "variable": col, "type": ftype, "n": int(s.size),
            "missing": int(X[col].isna().sum()),
            "mean": round(float(s.mean()), 3),
            "median": round(float(s.median()), 3),
            "std": round(float(s.std()), 3),
            "min": round(float(s.min()), 3),
            "max": round(float(s.max()), 3),
        })
    df = pd.DataFrame(rows)
    df.to_csv(config.TABLES_DIR / "table2_predictor_descriptives.csv",
              index=False)
    return df


# ---------------------------------------------------------------------------
# Table 3 – Training strategies (static documentation)
# ---------------------------------------------------------------------------
TABLE3_ROWS = [
    {"strategy": "Centralized", "privacy": "None",
     "how_trained": "Pooled training partition (reference, not private)",
     "hyperparameters": "SGD lr=0.1, batch=64, epochs=80"},
    {"strategy": "Local-only", "privacy": "Highest (no sharing)",
     "how_trained": "Each silo trains its own model on its own data",
     "hyperparameters": "SGD lr=0.1, batch=64, epochs=80"},
    {"strategy": "FedAvg", "privacy": "Shared delta only",
     "how_trained": "Server-weighted aggregation of client deltas, 40 rounds",
     "hyperparameters": "rounds=40, local_epochs=2, lr=0.1, batch=64"},
    {"strategy": "FedProx mu=0.01", "privacy": "Shared delta only",
     "how_trained": "FedAvg local objective + (mu/2)*||w-w_global||^2",
     "hyperparameters": "rounds=40, local_epochs=2, lr=0.1, mu=0.01"},
    {"strategy": "FedProx mu=0.1", "privacy": "Shared delta only",
     "how_trained": "FedAvg local objective + (mu/2)*||w-w_global||^2",
     "hyperparameters": "rounds=40, local_epochs=2, lr=0.1, mu=0.1"},
    {"strategy": "DP-FedAvg sigma=0.1..2.0", "privacy": "Client-level DP",
     "how_trained": "FedAvg with norm-clipped deltas + Gaussian noise sigma*C",
     "hyperparameters": "C=1.0, sigma=0.1..2.0, rounds=40, local_epochs=2"},
]


def write_table3():
    df = pd.DataFrame(TABLE3_ROWS)
    df.to_csv(config.TABLES_DIR / "table3_training_strategies.csv", index=False)
# ---------------------------------------------------------------------------
# Shared helper + Table 4/5
# ---------------------------------------------------------------------------
_STRATEGY_LABELS = {"centralized": "Centralized", "local_only": "Local-only",
                    "fedavg": "FedAvg"}


def _strategy_label(row):
    if row["strategy"] == "fedprox":
        return f"FedProx mu={row['mu']}"
    if row["strategy"] == "dp_fedavg":
        return f"DP sigma={row['sigma']}"
    return _STRATEGY_LABELS.get(row["strategy"], row["strategy"])


def write_table4(profile_df: pd.DataFrame, regime: str):
    df = profile_df.copy()
    df.insert(0, "regime", regime)
    df.to_csv(config.TABLES_DIR / f"table4_silo_heterogeneity_{regime}.csv",
              index=False)
    return df


def write_table5(results_df: pd.DataFrame):
    """Test ROC-AUC by strategy x model x regime, mean (std) over seeds."""
    df = results_df[results_df["roc_auc"].notna()].copy()
    df["strategy_label"] = df.apply(_strategy_label, axis=1)
    agg = (df.groupby(["model", "regime", "strategy_label"])["roc_auc"]
           .agg(roc_auc_mean="mean", roc_auc_std=lambda s: s.std(ddof=1),
                n_seeds="count")
           .reset_index()
           .round(6))
    agg.to_csv(config.TABLES_DIR / "table5_test_roc_auc_matrix.csv",
               index=False)
    # compact wide view: rows = (model, strategy), cols = regimes
    wide = agg.pivot_table(index=["model", "strategy_label"],
                           columns="regime", values="roc_auc_mean").round(4)
    wide.to_csv(config.TABLES_DIR / "table5_test_roc_auc_wide.csv")
    return agg
# ---------------------------------------------------------------------------
# Tables 6-10
# ---------------------------------------------------------------------------
def write_table6(primary_df: pd.DataFrame):
    df = primary_df.sort_values("p_value").reset_index(drop=True)
    df.to_csv(config.TABLES_DIR / "table6_primary_significance.csv",
              index=False)
    return df


def write_table7(secondary_df: pd.DataFrame):
    df = secondary_df.copy()
    df.to_csv(config.TABLES_DIR / "table7_secondary_comparisons.csv",
              index=False)
    return df


def write_table8(per_silo_df: pd.DataFrame):
    df = per_silo_df[(per_silo_df["model"] == "logistic")].copy()
    df.to_csv(config.TABLES_DIR / "table8_persilo_logistic_auc.csv",
              index=False)
    return df


def write_table9(results_df: pd.DataFrame):
    """Full logistic metric panel at threshold 0.5 (mean over seeds)."""
    df = results_df[results_df["model"] == "logistic"].copy()
    cols = [c for c in ("accuracy", "precision", "recall", "f1",
                        "roc_auc", "pr_auc") if c in df.columns]
    out = df.groupby(["regime", "strategy", "mu", "sigma"])[cols].mean().round(4)
    out = out.reset_index().sort_values(["regime", "strategy"])
    out.to_csv(config.TABLES_DIR / "table9_logistic_metric_panel.csv",
               index=False)
    return out
# ---------------------------------------------------------------------------
# Table 11 – Complete experimental configuration (static)
# ---------------------------------------------------------------------------
def write_table11():
    rows = [
        {"parameter": "dataset", "value": "Framingham Heart Study (4,240)"},
        {"parameter": "n_silos", "value": config.N_SILOS},
        {"parameter": "test_fraction", "value": config.TEST_FRACTION},
        {"parameter": "rounds", "value": config.ROUNDS},
        {"parameter": "local_epochs", "value": config.LOCAL_EPOCHS},
        {"parameter": "learning_rate", "value": config.LEARNING_RATE},
        {"parameter": "batch_size", "value": config.BATCH_SIZE},
        {"parameter": "positive_class_weight", "value": config.POSITIVE_CLASS_WEIGHT},
        {"parameter": "negative_class_weight", "value": config.NEGATIVE_CLASS_WEIGHT},
        {"parameter": "dirichlet_alpha", "value": config.DIRICHLET_ALPHA},
        {"parameter": "dp_clip_norm_C", "value": config.DP_CLIP_NORM},
        {"parameter": "dp_sigmas", "value": ",".join(map(str, config.DP_SIGMAS))},
        {"parameter": "fedprox_mus", "value": ",".join(map(str, config.FEDPROX_MUS))},
        {"parameter": "seeds", "value": ",".join(map(str, config.SEEDS))},
        {"parameter": "dp_delta", "value": config.DP_DELTA},
        {"parameter": "bootstrap_samples", "value": config.BOOTSTRAP_SAMPLES},
        {"parameter": "practical_auc_delta", "value": config.PRACTICAL_AUC_DELTA},
        {"parameter": "mlp_hidden_units", "value": config.MLP_HIDDEN_UNITS},
        {"parameter": "logistic_n_params", "value": 16},
        {"parameter": "mlp_n_params", "value": 273},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(config.TABLES_DIR / "table11_configuration.csv", index=False)
    return df
    out.to_csv(config.TABLES_DIR / "table9_logistic_metric_panel.csv",
               index=False)
    return out


def write_table10(privacy_utility_df: pd.DataFrame):
    df = privacy_utility_df.round(4)
    df.to_csv(config.TABLES_DIR / "table10_privacy_utility.csv", index=False)
    return df
    df["label"] = df.apply(_strategy_label, axis=1)
    g = df.groupby(["regime", "model", "label"])["roc_auc"].agg(
        ["mean", "std", "count"])
    g.columns = ["roc_auc_mean", "roc_auc_std", "n_seeds"]
    g = g.round(4).reset_index()
    g.to_csv(config.TABLES_DIR / "table5_test_roc_auc.csv", index=False)
    return g
    return df