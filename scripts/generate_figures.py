#!/usr/bin/env python
"""Generate all paper figures from the saved results.

Usage:
    python scripts/generate_figures.py
    python scripts/generate_figures.py --figure 3   # single figure
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from src import experiments as E
from src import plotting as P
from src import data_loader
from src import analysis as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", type=int, default=None)
    args = ap.parse_args()

    run_csv = config.RAW_DIR / "run_results.csv"
    if not run_csv.exists():
        sys.exit("No run_results.csv found - run scripts/run_experiments.py "
                 "first.")
    results = pd.read_csv(run_csv)
    conv = config.RAW_DIR / "convergence.csv"
    conv_df = pd.read_csv(conv) if conv.exists() else pd.DataFrame()
    per_silo = config.RAW_DIR / "per_silo_local_results.csv"
    per_silo_df = pd.read_csv(per_silo) if per_silo.exists() else pd.DataFrame()

    def want(n):
        return args.figure is None or args.figure == n

    # Figure 1 - architecture
    if want(1):
        P.figure1_architecture()

    # Figure 2 - silo prevalence/count (seed 0)
    if want(2):
        df = data_loader.load_framingham()
        profiles = {}
        for regime in config.REGIMES:
            _, _, _, _, summary = E.get_regime_data(0, regime, df)
            profiles[regime] = summary
        P.figure2_prevalence(profiles)

    # Figure 3 - AUC strategy matrix
    if want(3) and len(results):
        P.figure3_auc_matrix(results)

    # Figure 4 - federation gain
    if want(4) and len(results):
        primary = A.primary_comparisons(results)
        P.figure4_federation_gain(primary)

    # Figure 5 - convergence (logistic)
    if want(5) and len(conv_df):
        P.figure5_convergence(conv_df, model="logistic")

    # Figure 6 - privacy-utility
    if want(6) and len(results):
        priv_util = A.privacy_utility_table(results)
        P.figure6_privacy_utility(priv_util)

    print(f"Figures written to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()