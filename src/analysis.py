"""
Statistical analysis pipeline.

1. Primary paired comparisons (FedAvg vs Local-only) with Holm correction.
2. Secondary comparisons (FedAvg vs Centralized, FedProx vs FedAvg,
   DP-FedAvg vs FedAvg).
3. Cross-silo equity (local-only).
4. Privacy-utility table (DP-FedAvg).
5. Communication cost.
6. Reproduction checks (paper values vs reproduced values).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from . import statistics as st
from . import privacy_accounting


# ---------------------------------------------------------------------------
# Communication cost
# ---------------------------------------------------------------------------
def communication_cost(n_params: int, silent=False):
    """Total uploaded+downloaded bytes across 40 rounds, 5 clients, float32.

    Upload:   each client sends a delta on every round
    Download: server broadcasts the global model on every round

    MB convention: decimal megabytes (1 MB = 10^6 bytes), matching the paper's
    reported figures (16 params -> 25,600 B = 0.026 MB; 273 params ->
    436,800 B = 0.437 MB).
    """
    n_clients = config.N_SILOS
    rounds = config.ROUNDS
    bytes_param = 4  # float32
    per_round = 2 * n_clients * n_params * bytes_param
    total_bytes = per_round * rounds
    mb = total_bytes / 1e6
    if not silent:
        print(f"  params={n_params}: upload="
              f"{n_clients*rounds*n_params*4/1e6:.4f} MB, "
              f"download={n_clients*rounds*n_params*4/1e6:.4f} MB, "
              f"total={mb:.4f} MB")
    return {
        "n_params": n_params,
        "upload_mb": n_clients * rounds * n_params * 4 / 1e6,
        "download_mb": n_clients * rounds * n_params * 4 / 1e6,
        "total_mb": mb,
    }


# ---------------------------------------------------------------------------
# Primary comparisons
# ---------------------------------------------------------------------------
def primary_comparisons(results_df: pd.DataFrame) -> pd.DataFrame:
    """FedAvg vs Local-only, paired over the shared seeds, for 2x3 cells."""
    rows = []
    for model in config.MODELS:
        for regime in config.REGIMES:
            fa = results_df[(results_df["model"] == model) &
                            (results_df["regime"] == regime) &
                            (results_df["strategy"] == "fedavg")]
            lo = results_df[(results_df["model"] == model) &
                            (results_df["regime"] == regime) &
                            (results_df["strategy"] == "local_only")]
            merged = fa[["seed", "roc_auc"]].merge(
                lo[["seed", "roc_auc"]], on="seed", suffixes=("_a", "_b"))
            a = merged["roc_auc_a"].to_numpy(dtype=float)
            b = merged["roc_auc_b"].to_numpy(dtype=float)
            rows.append(st.primary_comparison(a, b, "fedavg", "local_only",
                                              model, regime))
    df = pd.DataFrame(rows)
    pvals = df["p_value"].to_numpy(dtype=float)
    df["holm_adjusted_p"] = st.holm_stepdown(pvals)
    return df
# ---------------------------------------------------------------------------
# Secondary comparisons
# ---------------------------------------------------------------------------
def _paired_series(base: pd.DataFrame, strategy: str, mu=None,
                   sigma=None) -> dict:
    sub = base[(base["strategy"] == strategy)]
    if mu is not None:
        sub = sub[np.isclose(sub["mu"].astype(float), mu)]
    if sigma is not None:
        sub = sub[np.isclose(sub["sigma"].astype(float), sigma)]
    return dict(zip(sub["seed"].astype(int), sub["roc_auc"].astype(float)))


def _comp_row(a: dict, b: dict, model, regime, name_a, name_b) -> dict:
    seeds = sorted(set(a) & set(b))
    xa = np.array([a[s] for s in seeds], dtype=float)
    xb = np.array([b[s] for s in seeds], dtype=float)
    d = xa - xb
    if len(d) < 2:
        return {"model": model, "regime": regime, "method_a": name_a,
                "method_b": name_b, "mean_diff": float("nan"),
                "p_value": float("nan"), "t_stat": float("nan"),
                "cohens_d": float("nan"), "n_seeds": len(seeds)}
    t_stat, p = st.paired_t_test(xa, xb)
    return {"model": model, "regime": regime, "method_a": name_a,
            "method_b": name_b, "mean_diff": float(np.mean(d)),
            "p_value": p, "t_stat": t_stat,
            "cohens_d": st.paired_cohens_d(xa, xb), "n_seeds": len(seeds)}


def secondary_comparisons(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in config.MODELS:
        for regime in config.REGIMES:
            base = results_df[(results_df["model"] == model) &
                              (results_df["regime"] == regime)]
            fa = _paired_series(base, "fedavg")
            cen = _paired_series(base, "centralized")
            rows.append(_comp_row(fa, cen, model, regime, "fedavg",
                                  "centralized"))
            lo = _paired_series(base, "local_only")
            rows.append(_comp_row(fa, lo, model, regime, "fedavg",
                                  "local_only"))
            for mu in config.FEDPROX_MUS:
                fp = _paired_series(base, "fedprox", mu=mu)
                rows.append(_comp_row(fp, fa, model, regime,
                                      f"fedprox_mu{mu}", "fedavg"))
            for sigma in config.DP_SIGMAS:
                dpa = _paired_series(base, "dp_fedavg", sigma=sigma)
                rows.append(_comp_row(dpa, fa, model, regime,
                                      f"dp_sigma{sigma}", "fedavg"))
    df = pd.DataFrame(rows)
    df["practical"] = df["mean_diff"].apply(
        lambda d: st.classify_practical_significance(d))
    return df


# ---------------------------------------------------------------------------
# Cross-silo equity (local-only logistic)
# ---------------------------------------------------------------------------
def cross_silo_equity(per_silo_df: pd.DataFrame) -> pd.DataFrame:
    df = per_silo_df[(per_silo_df["model"] == "logistic")].copy()
    rows = []
    for regime in config.REGIMES:
        for seed in config.SEEDS:
            sub = df[(df["regime"] == regime) & (df["seed"] == seed)]
            aucs = sub["roc_auc"].dropna().to_numpy(dtype=float)
            if len(aucs) == 0:
                continue
            rows.append({
                "seed": seed, "regime": regime,
                "best_silo_auc": float(aucs.max()),
                "worst_silo_auc": float(aucs.min()),
                "spread": float(aucs.max() - aucs.min()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Privacy-utility table
# ---------------------------------------------------------------------------
def privacy_utility_table(results_df: pd.DataFrame) -> pd.DataFrame:
    sig_eps = {s: privacy_accounting.compute_epsilon(s)
               for s in config.DP_SIGMAS}
    ref = results_df[(results_df["strategy"] == "fedavg") &
                     (results_df["model"] == "logistic") &
                     (results_df["regime"] == "iid")]
    ref_auc = ref["roc_auc"].mean() if len(ref) else float("nan")

    rows = []
    for model in config.MODELS:
        for sigma in config.DP_SIGMAS:
            sub = results_df[(results_df["model"] == model) &
                             (results_df["strategy"] == "dp_fedavg") &
                             np.isclose(results_df["sigma"].astype(float),
                                        sigma) &
                             (results_df["regime"] == "iid")]
            auc = sub["roc_auc"].mean() if len(sub) else float("nan")
            rows.append({
                "model": model,
                "sigma": sigma,
                "epsilon": sig_eps[sigma],
                "delta": config.DP_DELTA,
                "roc_auc": auc,
                "utility_loss": ref_auc - auc,
            })
    df = pd.DataFrame(rows)
    df["fedavg_ref_auc"] = ref_auc
    return df


# ---------------------------------------------------------------------------
# Calibration-shift analysis (label skew vs IID for logistic FedAvg)
# ---------------------------------------------------------------------------
def calibration_shifts(results_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["roc_auc", "accuracy", "precision", "recall", "f1"]
    rows = []
    for regime in ["iid", "label_skew"]:
        sub = results_df[(results_df["strategy"] == "fedavg") &
                         (results_df["model"] == "logistic") &
                         (results_df["regime"] == regime)]
        row = {"regime": regime}
        for c in cols:
            row[c] = sub[c].mean() if len(sub) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reproduction checks
# ---------------------------------------------------------------------------
def reproduction_checks(results_df: pd.DataFrame, per_silo_df: pd.DataFrame,
                        dataset_facts: dict, primary_df: pd.DataFrame,
                        privacy_util: pd.DataFrame,
                        comm_costs: dict,
                        profiles: dict = None) -> pd.DataFrame:
    """Compare reproduced values against the paper's stated approximations."""
    checks = []

    def add(metric, paper, reproduced, tol=0.05, note=""):
        if reproduced is None:
            reproduced = float("nan")
        if isinstance(reproduced, float) and np.isnan(reproduced):
            within = False
            adiff = rdiff = float("nan")
        else:
            adiff = abs(float(reproduced) - paper)
            rdiff = adiff / max(paper, 1e-12)
            within = bool(adiff <= tol * max(paper, 1e-12))
        checks.append({"metric": metric, "paper_value": paper,
                       "reproduced_value": reproduced,
                       "absolute_difference": adiff,
                       "relative_difference": rdiff,
                       "within_tolerance": within, "note": note})

    # dataset
    add("dataset_n_rows", 4240, dataset_facts["n_rows"], 0.0)
    add("n_features", 15, dataset_facts["n_features"], 0.0)
    add("positive_prevalence", 0.152, dataset_facts["prevalence"], 0.01)

    # silo structure (seed 0 profiles; paper reports seed-0 values)
    if profiles:
        iid = profiles.get("iid")
        if iid is not None and len(iid):
            add("iid_mean_silo_size_seed0", 678.4,
                float(iid["n_records"].mean()), 0.01,
                note="paper: 679/679/678/678/678")
        cov = profiles.get("covariate_skew")
        if cov is not None and len(cov):
            add("covariate_min_silo_prevalence_seed0", 0.056,
                float(cov["chd_prevalence"].min()), 0.20,
                note="youngest age band (paper ~0.056)")
            add("covariate_max_silo_prevalence_seed0", 0.271,
                float(cov["chd_prevalence"].max()), 0.10,
                note="oldest age band (paper ~0.271)")
        lab = profiles.get("label_skew")
        if lab is not None and len(lab):
            rng_lab = (float(lab["chd_prevalence"].max()) -
                       float(lab["chd_prevalence"].min()))
            add("label_skew_prevalence_range_seed0", 0.778, rng_lab, 0.30,
                note="max-min silo prevalence; Dirichlet draw is "
                     "RNG-dependent, so generous tolerance")

    # centralized
    for model, paper in (("logistic", 0.7290), ("mlp", 0.7044)):
        v = results_df[(results_df["strategy"] == "centralized") &
                       (results_df["model"] == model)]
        add(f"centralized_{model}_auc", paper,
            v["roc_auc"].mean() if len(v) else None, 0.05)

    # FedAvg
    fedavg_paper = {
        "logistic": {"iid": 0.7315, "label_skew": 0.7313,
                     "covariate_skew": 0.7321},
        "mlp": {"iid": 0.7232, "label_skew": 0.7078,
                "covariate_skew": 0.7281},
    }
    for model, regmap in fedavg_paper.items():
        for regime, paper in regmap.items():
            v = results_df[(results_df["strategy"] == "fedavg") &
                           (results_df["model"] == model) &
                           (results_df["regime"] == regime)]
            add(f"fedavg_{model}_{regime}_auc", paper,
                v["roc_auc"].mean() if len(v) else None, 0.05)

    # federation gains
    gain_paper = {
        "logistic": {"iid": 0.0226, "label_skew": 0.0370,
                     "covariate_skew": 0.0380},
        "mlp": {"iid": 0.0761, "label_skew": 0.0611,
                "covariate_skew": 0.1110},
    }
    for model, regmap in gain_paper.items():
        for regime, paper in regmap.items():
            row = primary_df[(primary_df["model"] == model) &
                             (primary_df["regime"] == regime)]
            add(f"federation_gain_{model}_{regime}", paper,
                row["mean_gain"].iloc[0] if len(row) else None, 0.50)

    # local-only back-calculated
    for model, regmap in gain_paper.items():
        for regime, paper_gain in regmap.items():
            fa = results_df[(results_df["strategy"] == "fedavg") &
                            (results_df["model"] == model) &
                            (results_df["regime"] == regime)]["roc_auc"].mean()
            lo = results_df[(results_df["strategy"] == "local_only") &
                            (results_df["model"] == model) &
                            (results_df["regime"] == regime)]["roc_auc"].mean()
            if not np.isnan(fa) and not np.isnan(lo):
                add(f"local_only_{model}_{regime}_auc", fa - paper_gain,
                    lo, 0.05)

    # DP epsilon
    eps_paper = {0.1: 2315.1, 0.25: 441.6, 0.5: 140.8, 1.0: 50.4, 2.0: 20.2}
    for sigma, paper in eps_paper.items():
        v = privacy_util[privacy_util["sigma"] == sigma]
        add(f"dp_epsilon_sigma{sigma}", paper,
            v["epsilon"].iloc[0] if len(v) else None, 0.10)

    # DP utility
    dp_paper = {
        ("logistic", 0.1): 0.7288, ("mlp", 0.1): 0.7193,
        ("logistic", 0.25): 0.7208, ("mlp", 0.25): 0.7082,
        ("logistic", 0.5): 0.6979, ("mlp", 0.5): 0.6651,
        ("logistic", 1.0): 0.6543, ("mlp", 1.0): 0.5991,
        ("logistic", 2.0): 0.6032, ("mlp", 2.0): 0.5532,
    }
    for (model, sigma), paper in dp_paper.items():
        v = privacy_util[(privacy_util["model"] == model) &
                         (privacy_util["sigma"] == sigma)]
        add(f"dp_util_{model}_sigma{sigma}", paper,
            v["roc_auc"].iloc[0] if len(v) else None, 0.05)

    # communication cost
    add("comm_cost_logistic_mb", 0.026,
        comm_costs["logistic"]["total_mb"], 0.10)
    add("comm_cost_mlp_mb", 0.437, comm_costs["mlp"]["total_mb"], 0.10)

    return pd.DataFrame(checks)