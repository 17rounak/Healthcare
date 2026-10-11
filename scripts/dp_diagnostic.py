"""
Diagnostic: compare DP noise conventions for a small fixed experiment.

Conventions tested for each sigma in {0.1, 0.25, 0.5, 1.0, 2.0}:

  A. current implementation:
       noise std = sigma * C  on the released aggregate (sensitivity = C, conservative)

  B. noise added to each clipped client update before averaging:
       per-client noise std = sigma * C  on each delta_i before weighted averaging

  C. equivalent aggregate noise after averaging:
       noise std = sigma * C / sqrt(K)  on the aggregate (result of B with equal weights)

  D. aggregate noise with scale sigma * C / K:
       noise std = sigma * C / K  on the aggregate (sensitivity = C/K, tight bound)

Also derives the sensitivity and Gaussian mechanism convention algebraically.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src import config, data_loader, preprocessing, partitioning, models, optimizer, metrics

K = config.N_SILOS  # 5
C = config.DP_CLIP_NORM  # 1.0
ROUNDS = config.ROUNDS  # 40
LOCAL_EPOCHS = config.LOCAL_EPOCHS  # 2
SEED = 0
REGIME = config.REGIME_IID


def setup_data():
    """Load and partition data for the diagnostic."""
    df = data_loader.load_framingham()
    train_df, test_df = data_loader.make_train_test_split(df, SEED)
    medians = preprocessing.compute_medians(train_df)
    silos = partitioning.make_partition(train_df, REGIME, SEED, medians)

    train_imp = preprocessing.impute_with_medians(train_df, medians)
    test_imp = preprocessing.impute_with_medians(test_df, medians)
    X_train = train_imp[config.FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_imp[config.TARGET_COLUMN].to_numpy(dtype=float)
    X_test = test_imp[config.FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test = test_imp[config.TARGET_COLUMN].to_numpy(dtype=float)

    local_stats = [preprocessing.compute_local_statistics(X_train[idx]) for idx in silos]
    agg_stats = preprocessing.aggregate_statistics(local_stats)
    gstats = preprocessing.derive_global_standardization(agg_stats)

    silo_X = [preprocessing.standardize(X_train[idx], gstats) for idx in silos]
    silo_y = [y_train[idx] for idx in silos]
    test_X = preprocessing.standardize(X_test, gstats)

    return silo_X, silo_y, test_X, y_test


def compute_silo_weights(silo_X):
    """Compute sample-count weights for each silo."""
    sizes = np.array([X.shape[0] for X in silo_X], dtype=float)
    total = float(sizes.sum())
    weights = sizes / total
    return weights, sizes, total


def clip_update(delta, clip_norm):
    """L2-norm clip a client delta."""
    delta = np.asarray(delta, dtype=float)
    norm = float(np.linalg.norm(delta))
    if norm > clip_norm and norm > 0.0:
        return delta * (clip_norm / norm)
    return delta.copy()


def run_federated_dp(silo_X, silo_y, test_X, y_test, model_factory, sigma,
                     noise_convention, seed, clip_norm=C):
    """
    Run DP-FedAvg with a specified noise convention.

    noise_convention:
      'A' -> noise std = sigma * C on aggregate (current)
      'B' -> per-client noise std = sigma * C before averaging
      'C' -> noise std = sigma * C / sqrt(K) on aggregate
      'D' -> noise std = sigma * C / K on aggregate
    """
    weights, sizes, total = compute_silo_weights(silo_X)
    k = len(silo_X)

    model = model_factory()
    w = model.get_parameters()

    # Independent reproducible noise stream
    dp_rng = np.random.default_rng(int(seed * 100000 + round(sigma * 1000)))

    for r in range(1, ROUNDS + 1):
        deltas = []
        for i in range(k):
            model.set_parameters(w)
            local_rng = np.random.default_rng(np.random.SeedSequence([seed, int(r), int(i)]))
            params = optimizer.train_sgd(
                model, silo_X[i], silo_y[i], local_rng,
                epochs=LOCAL_EPOCHS)
            delta = params - w
            delta = clip_update(delta, clip_norm)
            deltas.append(delta)

        # Compute weighted average
        agg = np.zeros_like(deltas[0], dtype=float)
        for delta, wt in zip(deltas, weights):
            agg = agg + wt * delta

        # Add noise according to convention
        if noise_convention == 'A':
            # Current: noise std = sigma * C on aggregate
            noise_scale = sigma * clip_norm
            agg = agg + dp_rng.normal(0.0, noise_scale, size=agg.shape)
        elif noise_convention == 'B':
            # Per-client noise: add sigma*C noise to each delta before averaging
            noisy_deltas = []
            for delta in deltas:
                noisy_delta = delta + dp_rng.normal(0.0, sigma * clip_norm, size=delta.shape)
                noisy_deltas.append(noisy_delta)
            agg = np.zeros_like(deltas[0], dtype=float)
            for delta, wt in zip(noisy_deltas, weights):
                agg = agg + wt * delta
        elif noise_convention == 'C':
            # Equivalent aggregate noise: sigma * C / sqrt(K)
            noise_scale = sigma * clip_norm / np.sqrt(k)
            agg = agg + dp_rng.normal(0.0, noise_scale, size=agg.shape)
        elif noise_convention == 'D':
            # Old convention: sigma * C / K
            noise_scale = sigma * clip_norm / k
            agg = agg + dp_rng.normal(0.0, noise_scale, size=agg.shape)
        else:
            raise ValueError(f"Unknown convention: {noise_convention}")

        w = w + agg
        model.set_parameters(w)

    # Evaluate
    auc = metrics.roc_auc_score(y_test, model.predict_proba(test_X))
    return auc

def derive_sensitivity():
    """Print the algebraic derivation of sensitivity."""
    print("=" * 70)
    print("ALGEBRAIC SENSITIVITY ANALYSIS")
    print("=" * 70)
    print()
    print("Setup:")
    print(f"  K = {K} silos")
    print(f"  C = {C} (clip norm)")
    print(f"  delta_i = clipped client update, ||delta_i||_2 <= C")
    print(f"  w_i = n_i / N (sample-count weight, sum_i w_i = 1)")
    print()
    print("Released quantity: agg = sum_i w_i * delta_i")
    print()
    print("Sensitivity to add/remove one client k:")
    print("  Since weights are FIXED (computed from silo sizes, which don't change):")
    print("  Removing client k changes agg by: w_k * delta_k")
    print("  L2 norm of change: ||w_k * delta_k|| = w_k * ||delta_k|| <= w_k * C")
    print()
    print("  Sensitivity Delta_f = max_k w_k * C")
    print()
    print("For IID with balanced silos: w_i ~ 1/K, so Delta_f ~ C/K")
    print(f"  With K={K}, C={C}: Delta_f ~ {C/K:.4f}")
    print()
    print("The current implementation uses Delta_f = C (conservative bound)")
    print(f"  This is {K}x larger than the tight bound C/K = {C/K:.4f}")
    print()
    print("-" * 70)
    print("RDP CONSISTENCY ANALYSIS")
    print("-" * 70)
    print()
    print("RDP formula (per round): RDP(alpha) = alpha / (2 * sigma^2)")
    print()
    print("This is the RDP of the Gaussian mechanism:")
    print("  M(D) = f(D) + z,  z ~ N(0, sigma_g^2 * I)")
    print("  where sigma_g = sigma * Delta_f")
    print()
    print("So: RDP = alpha * Delta_f^2 / (2 * sigma_g^2)")
    print("        = alpha * Delta_f^2 / (2 * sigma^2 * Delta_f^2)")
    print("        = alpha / (2 * sigma^2)  CONSISTENT for ANY Delta_f")
    print()
    print("The key insight: sigma is DEFINED as sigma_g / Delta_f.")
    print("The RDP formula is independent of the actual Delta_f value.")
    print("What matters for UTILITY is sigma_g = sigma * Delta_f.")
    print()
    print("Convention comparison (sigma_g = actual noise std on release):")
    print(f"  A (current):  sigma_g = sigma * C       = sigma * {C}")
    print(f"  B (per-client): sigma_g = sigma * C / sqrt(K) = sigma * {C/np.sqrt(K):.4f}  (equal weights)")
    print(f"  C (equiv agg):  sigma_g = sigma * C / sqrt(K) = sigma * {C/np.sqrt(K):.4f}")
    print(f"  D (old):        sigma_g = sigma * C / K       = sigma * {C/K:.4f}")
    print()
    print("For the RDP formula to be consistent, the accountant must use")
    print("the SAME Delta_f as the mechanism. Since the formula only depends")
    print("on sigma (= sigma_g / Delta_f), any consistent pair works.")
    print()
    print("The paper's epsilon values are computed from sigma alone:")
    print("  epsilon = min_alpha [T*alpha/(2*sigma^2) + log(1/delta)/(alpha-1)]")
    print("These are INDEPENDENT of the sensitivity convention.")
    print()
def main():
    print("=" * 70)
    print("DP NOISE CONVENTION DIAGNOSTIC")
    print("=" * 70)
    print()

    # Derive sensitivity algebraically
    derive_sensitivity()

    # Setup data
    print("Setting up data...")
    silo_X, silo_y, test_X, y_test = setup_data()
    weights, sizes, total = compute_silo_weights(silo_X)

    print(f"Silo sizes: {sizes.astype(int)}")
    print(f"Silo weights: {weights}")
    print(f"max(w_i) = {weights.max():.4f}")
    print(f"Tight sensitivity = max(w_i) * C = {weights.max() * C:.4f}")
    print(f"Conservative sensitivity = C = {C}")
    print()

    # Model factory
    def model_factory():
        model = models.LogisticRegression()
        model.initialize(config.INIT_SEED_OFFSET + SEED)
        return model

    sigmas = config.DP_SIGMAS
    conventions = ['A', 'B', 'C', 'D']

    print("=" * 70)
    print("RESULTS: ROC-AUC by convention and sigma")
    print("=" * 70)
    print()
    print(f"{'sigma':>6} | {'A':>10} | {'B':>10} | {'C':>10} | {'D':>10} | paper")
    print("-" * 65)

    paper_logistic = {0.1: 0.7288, 0.25: 0.7208, 0.5: 0.6979, 1.0: 0.6543, 2.0: 0.6032}

    results = {conv: {} for conv in conventions}

    for sigma in sigmas:
        row = f"{sigma:>6.2f} |"
        for conv in conventions:
            auc = run_federated_dp(silo_X, silo_y, test_X, y_test,
                                   model_factory, sigma, conv, SEED)
            results[conv][sigma] = auc
            row += f" {auc:>10.4f} |"
        row += f" {paper_logistic[sigma]:.4f}"
        print(row)

    print()
    print("=" * 70)
    print("ACTUAL NOISE STANDARD DEVIATION ON RELEASED AGGREGATE")
    print("=" * 70)
    print()
    print(f"{'sigma':>6} | {'A':>10} | {'B':>10} | {'C':>10} | {'D':>10}")
    print("-" * 55)
    for sigma in sigmas:
        noise_A = sigma * C
        noise_B = sigma * C / np.sqrt(K)
        noise_C = sigma * C / np.sqrt(K)
        noise_D = sigma * C / K
        print(f"{sigma:>6.2f} | {noise_A:>10.4f} | {noise_B:>10.4f} | {noise_C:>10.4f} | {noise_D:>10.4f}")

    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()
    print("Key observations:")
    print()
    print("1. The RDP formula RDP = alpha/(2*sigma^2) is consistent with ANY")
    print("   mechanism where noise_std = sigma * sensitivity, as long as the")
    print("   accountant and mechanism agree on the sensitivity.")
    print()
    print("2. The paper's epsilon values depend ONLY on sigma, not on the")
    print("   sensitivity convention. So all conventions produce the same epsilon.")
    print()
    print("3. What differs is the ACTUAL noise magnitude (noise_std):")
    print(f"   A: noise_std = sigma * {C}")
    print(f"   B: noise_std = sigma * {C:.4f} / sqrt({K}) = sigma * {C/np.sqrt(K):.4f}")
    print(f"   C: noise_std = sigma * {C:.4f} / sqrt({K}) = sigma * {C/np.sqrt(K):.4f}")
    print(f"   D: noise_std = sigma * {C:.4f} / {K} = sigma * {C/K:.4f}")
    print()
    print("4. The current implementation (A) is CONSERVATIVE: it adds K times")
    print("   more noise than the tight-bound convention (D) would require")
    print("   for the same stated epsilon.")
    print()
    print("5. Convention D (sigma*C/K) is the tight bound for equal weights:")
    print("   sensitivity = C/K, noise_std = sigma * C/K.")
    print("   This is mathematically consistent with the RDP formula.")
    print()
    print("6. The paper's sigma=2 result matches convention A (current),")
    print("   while lower sigma results are closer to convention D (old).")
    print("   This suggests the paper may have an inconsistent noise scaling.")


if __name__ == "__main__":
    main()


