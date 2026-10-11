"""
Publication-quality figures.

Uses the results stored by the experiment/analysis pipeline. Agg backend is
forced so the scripts run in any environment (no display required).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


def _save(fig, name, dpi=300):
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"{name}.png", dpi=dpi,
                bbox_inches="tight")
    for fmt in ("pdf", "svg"):
        try:
            fig.savefig(config.FIGURES_DIR / f"{name}.{fmt}",
                        bbox_inches="tight")
        except Exception as exc:  # vector backends can fail headlessly
            print(f"[plotting] skip {fmt} for {name}: {exc}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 – Federated training architecture
# ---------------------------------------------------------------------------
def figure1_architecture():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.axis("off")

    hospitals = ["Hospital 1", "Hospital 2", "Hospital 3", "Hospital 4",
                 "Hospital 5"]
    boxes = []
    for i, name in enumerate(hospitals):
        y = 4.6 - i * 1.0
        ax.add_patch(plt.Rectangle((0.02, y), 0.26, 0.72,
                                   facecolor="#dbe9f4",
                                   edgecolor="#2c3e50", lw=1.2))
        ax.text(0.15, y + 0.62, name, ha="center", va="center",
                fontsize=11, weight="bold")
        ax.text(0.15, y + 0.40, "Private EHR data", ha="center",
                va="center", fontsize=8.5)
        ax.text(0.15, y + 0.18, "Local SGD (2 epochs)", ha="center",
                va="center", fontsize=8.5)
        ax.text(0.15, y - 0.05, "parameter delta", ha="center",
                va="center", fontsize=8.5, style="italic",
                color="#c0392b")
        boxes.append((0.15, y + 0.36))

    ax.annotate("", xy=(0.34, 4.3), xytext=(0.34, 0.7),
                arrowprops=dict(arrowstyle="-", color="#7f8c8d", lw=1.5))
    ax.text(0.345, 2.5, "privacy boundary:\nonly deltas cross",
            fontsize=9, rotation=90, va="center", color="#7f8c8d")

    ax.add_patch(plt.Rectangle((0.52, 2.2), 0.30, 1.6, facecolor="#fdebd0",
                               edgecolor="#2c3e50", lw=1.4))
    ax.text(0.67, 3.6, "aggregation\nserver", ha="center", va="center",
            fontsize=11, weight="bold")
    ax.text(0.67, 3.05, "clipping (C=1.0)", ha="center", va="center",
            fontsize=8.5)
    ax.text(0.67, 2.55, "Gaussian noise\n(DP configurations)",
            ha="center", va="center", fontsize=8.5)

    ax.add_patch(plt.Rectangle((0.90, 2.4), 0.08, 1.2, facecolor="#d5f5e3",
                               edgecolor="#2c3e50", lw=1.2))
    ax.text(0.94, 3.0, "global\nmodel", ha="center", va="center", fontsize=9)

    for (x, y) in boxes:
        ax.annotate("", xy=(0.52, 3.0), xytext=(0.30, y),
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.6,
                                    connectionstyle="arc3,rad=0.05"))
    ax.annotate("", xy=(0.90, 3.0), xytext=(0.82, 3.0),
                arrowprops=dict(arrowstyle="->", color="#2e86c1", lw=1.8))
    ax.text(0.86, 3.25, "broadcast", ha="center", va="center", fontsize=9,
            color="#2e86c1")

    ax.add_patch(plt.Rectangle((0.02, 0.05), 0.26, 0.62, facecolor="#fadbd8",
                               edgecolor="#2c3e50", lw=1.2))
    ax.text(0.15, 0.42, "global held-out\ntest set (848)", ha="center",
            va="center", fontsize=9)
    ax.add_patch(plt.Rectangle((0.34, 0.05), 0.26, 0.62, facecolor="#f9e79f",
                               edgecolor="#2c3e50", lw=1.2))
    ax.text(0.47, 0.42, "ROC-AUC / PR-AUC / F1\naccuracy / precision / recall",
            ha="center", va="center", fontsize=8.5)
    ax.annotate("", xy=(0.47, 0.36), xytext=(0.15, 0.36),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5))
    ax.annotate("", xy=(0.47, 0.36), xytext=(0.90, 3.0),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5,
                                connectionstyle="arc3,rad=-0.25"))

    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.15, 5.4)
    ax.set_title("Figure 1: Federated training architecture", fontsize=13,
                 weight="bold")
    _save(fig, "figure1_architecture")
# ---------------------------------------------------------------------------
# Figure 2 – Outcome prevalence and record count per silo
# ---------------------------------------------------------------------------
def figure2_prevalence(profile: dict):
    """profile: {regime: DataFrame(silo, n_records, chd_prevalence)}"""
    regimes = list(profile.keys())
    fig, axes = plt.subplots(2, len(regimes), figsize=(5.2 * len(regimes), 7))
    labels = {"iid": "IID", "label_skew": "Label skew",
              "covariate_skew": "Covariate skew"}
    for j, reg in enumerate(regimes):
        dfreg = profile[reg]
        silos = np.arange(1, len(dfreg) + 1)
        axes[0, j].bar(silos, dfreg["n_records"], color="#5b9bd5")
        axes[0, j].set_title(labels.get(reg, reg))
        axes[0, j].set_xticks(silos)
        axes[0, j].set_ylabel("records")
        axes[1, j].bar(silos, dfreg["chd_prevalence"] * 100, color="#e05b5b")
        axes[1, j].set_xticks(silos)
        axes[1, j].set_xlabel("silo")
        axes[1, j].set_ylabel("CHD prevalence (%)")
        for i, v in enumerate(dfreg["chd_prevalence"] * 100):
            axes[1, j].text(i + 1, v + 0.4, f"{v:.1f}", ha="center",
                            fontsize=7.5)
    fig.suptitle("Figure 2: outcome prevalence and record count per silo "
                 "(seed 0)", fontsize=13, weight="bold")
    _save(fig, "figure2_silo_prevalence")


# ---------------------------------------------------------------------------
# Figure 3 – Test ROC-AUC by strategy, model family, regime
# ---------------------------------------------------------------------------
_STRAT_LABELS = {"centralized": "Centralized", "local_only": "Local-only",
                 "fedavg": "FedAvg"}


def _fig3_label(r):
    if r["strategy"] == "fedprox":
        return f"FedProx\nmu={r['mu']}"
    if r["strategy"] == "dp_fedavg":
        return f"DP\nsigma={r['sigma']}"
    return _STRAT_LABELS.get(r["strategy"], r["strategy"])


def figure3_auc_matrix(results_df: pd.DataFrame):
    df = results_df[results_df["roc_auc"].notna()].copy()
    df["label"] = df.apply(_fig3_label, axis=1)
    main = df[df["strategy"].isin(["centralized", "local_only", "fedavg",
                                   "fedprox"])]
    regs = [r for r in ["iid", "label_skew", "covariate_skew"]
            if r in set(main["regime"])] or sorted(main["regime"].unique())
    models = sorted(main["model"].unique())
    reg_labels = {"iid": "IID", "label_skew": "Label skew",
                  "covariate_skew": "Covariate skew"}

    fig, axes = plt.subplots(len(models), len(regs),
                             figsize=(5.2 * len(regs), 3.6 * len(models)),
                             squeeze=False)
    for mi, mod in enumerate(models):
        for ri, reg in enumerate(regs):
            ax = axes[mi][ri]
            sub = main[(main["model"] == mod) & (main["regime"] == reg)]
            order = ["Centralized", "Local-only", "FedAvg"]
            extra = sorted(set(sub["label"]) - set(order))
            cats = [c for c in order + extra if c in set(sub["label"])]
            means, stds, xs = [], [], []
            for j, cat in enumerate(cats):
                v = sub[sub["label"] == cat]["roc_auc"]
                if len(v) == 0:
                    continue
                means.append(v.mean())
                stds.append(v.std(ddof=1) if len(v) > 1 else 0.0)
                xs.append(j)
            ax.bar(xs, means, yerr=stds, capsize=3, color="#4d8fc9",
                   alpha=0.85)
            ax.axhline(0.5, color="grey", lw=0.8, ls="--")
            ax.set_xticks(xs, [cats[i] for i in xs], rotation=18, ha="right")
            ax.set_ylim(0.40, 0.85)
            ax.set_title(f"{mod} · {reg_labels.get(reg, reg)}")
            if ri == 0:
                ax.set_ylabel("test ROC-AUC")
    fig.suptitle("Figure 3: test ROC-AUC by strategy, model family, "
                 "heterogeneity regime (mean ± SD over 5 seeds)",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "figure3_auc_strategy_matrix")
# ---------------------------------------------------------------------------
# Figure 4 – Federation gain: FedAvg - Local-only
# ---------------------------------------------------------------------------
def figure4_federation_gain(primary_df: pd.DataFrame):
    df = primary_df[primary_df["method_a"] == "fedavg"].copy()
    reg_labels = {"iid": "IID", "label_skew": "Label skew",
                  "covariate_skew": "Covariate skew"}
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]
    for mi, mod in enumerate(models):
        ax = axes[mi]
        sub = df[df["model"] == mod]
        regs = [r for r in ["iid", "label_skew", "covariate_skew"]
                if r in set(sub["regime"])] or sorted(sub["regime"].unique())
        xs = np.arange(len(regs))
        gains = [sub[sub["regime"] == r]["mean_gain"].iloc[0] for r in regs]
        lo = [sub[sub["regime"] == r]["ci_lower"].iloc[0] for r in regs]
        hi = [sub[sub["regime"] == r]["ci_upper"].iloc[0] for r in regs]
        err = np.array([np.array(gains) - np.array(lo),
                        np.array(hi) - np.array(gains)])
        ax.bar(xs, gains, yerr=err, capsize=5, color="#4e8fc9", alpha=0.85)
        ax.axhline(0, color="grey", lw=0.9)
        for j, (g, d) in enumerate(zip(gains,
                                       sub["cohens_d"].astype(float))):
            ax.text(j, gains[j] + 0.01 * (1 if gains[j] >= 0 else -1),
                    f"d={d:.2f}", ha="center", fontsize=9)
        ax.set_xticks(xs, [reg_labels.get(r, r) for r in regs])
        ax.set_title(f"{mod} — FedAvg vs Local-only")
        ax.set_ylabel("mean paired ROC-AUC gain")
    fig.suptitle("Figure 4: federation gain with 95% paired-bootstrap CIs "
                 "and Cohen's d", fontsize=13, weight="bold")
    _save(fig, "figure4_federation_gain")


# ---------------------------------------------------------------------------
# Figure 5 – Test ROC-AUC vs communication round (logistic)
# ---------------------------------------------------------------------------
def figure5_convergence(conv_df: pd.DataFrame, model="logistic"):
    df = conv_df[(conv_df["model"] == model) &
                 (conv_df["strategy"] == "fedavg")].copy()
    reg_labels = {"iid": "IID", "label_skew": "Label skew",
                  "covariate_skew": "Covariate skew"}
    regs = [r for r in ["iid", "label_skew", "covariate_skew"]
            if r in set(df["regime"])] or sorted(df["regime"].unique())

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for reg in regs:
        sub = df[df["regime"] == reg].copy()
        pivot = sub.pivot_table(index="round", values="test_roc_auc",
                                aggfunc=["mean", "std"])
        rounds = pivot.index.values
        mean = pivot["mean"]["test_roc_auc"].values
        std = pivot["std"]["test_roc_auc"].values
        ax.plot(rounds, mean, label=reg_labels.get(reg, reg), lw=2)
        ax.fill_between(rounds, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("communication round")
    ax.set_ylabel(f"test ROC-AUC ({model})")
    ax.set_ylim(0.45, 0.85)
    ax.legend(title="regime")
    ax.set_title("Figure 5: test ROC-AUC across communication rounds "
                 "(FedAvg, mean ± SD over seeds; evaluation-only)",
                 fontsize=11, weight="bold")
    _save(fig, "figure5_convergence_logistic")


# ---------------------------------------------------------------------------
# Figure 6 – Privacy-utility tradeoff
# ---------------------------------------------------------------------------
def figure6_privacy_utility(privacy_utility_df: pd.DataFrame):
    df = privacy_utility_df.copy()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model].sort_values("epsilon")
        ax.plot(sub["epsilon"], sub["roc_auc"], marker="o", lw=2,
                label=f"{model} regression" if model == "logistic"
                else f"{model} (15-16-1)")
    # non-private FedAvg reference + chance
    ref = df[df["model"].notna()]
    fedavg_ref = df[["fedavg_ref_auc", "epsilon"]].drop_duplicates()
    if "fedavg_ref_auc" in df.columns and len(fedavg_ref):
        ax.axhline(fedavg_ref["fedavg_ref_auc"].iloc[0], color="green",
                   ls="--", lw=1.2, label="non-private FedAvg reference")
    ax.axhline(0.5, color="grey", ls=":", lw=1.0, label="chance (0.5)")
    ax.set_xscale("log")
    ax.set_xlabel("formal client-level epsilon (log scale)")
    ax.set_ylabel("test ROC-AUC")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title("Figure 6: privacy-utility tradeoff (client-level DP)",
                 fontsize=11, weight="bold")
    _save(fig, "figure6_privacy_utility")