"""
Experiment orchestration.

Single runs and the full factorial experiment:

    3 regimes x 2 model families x 5 seeds x 10 strategy/configuration
    variants = 300 training runs.

Consistent with the shared-seed protocol, for every seed:
  * one stratified global train/test split shared by all regimes/strategies;
  * one silo partition per regime shared by all models/strategies;
  * the same initialization, and the same mini-batch schedule where method
    structure permits (local shuffling keyed to (seed, round, silo)).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from . import config
from . import data_loader
from . import preprocessing
from . import partitioning
from . import models
from . import optimizer
from . import federated
from . import metrics

if TYPE_CHECKING:
    import pandas as pd

# ---------------------------------------------------------------------------
# Caches used inside worker processes
# ---------------------------------------------------------------------------
_SEED_DATA_CACHE = {}
_REGIME_CACHE = {}


def _effective_epochs():
    """Centralized / local-only optimisation budget (equivalent to federated)."""
    return config.ROUNDS * config.LOCAL_EPOCHS


def load_data():
    return data_loader.load_framingham()


def get_seed_data(seed: int, df=None):
    """(train_df, test_df, medians) for a seed; cached per process."""
    if seed in _SEED_DATA_CACHE:
        return _SEED_DATA_CACHE[seed]
    if df is None:
        df = load_data()
    train_df, test_df = data_loader.make_train_test_split(df, seed)
    medians = preprocessing.compute_medians(train_df)
    _SEED_DATA_CACHE[seed] = (train_df, test_df, medians)
    return _SEED_DATA_CACHE[seed]


def get_regime_data(seed: int, regime: str, df=None):
    """Silo feature/target lists and test features per (seed, regime).

    Imputation uses training medians; standardization aggregates per-silo
    sufficient statistics (n, sum, sum^2) exactly as described in the paper,
    then every silo and the test set use the same global scaler.
    """
    key = (seed, regime)
    if key in _REGIME_CACHE:
        return _REGIME_CACHE[key]

    train_df, test_df, medians = get_seed_data(seed, df)
    silos = partitioning.make_partition(train_df, regime, seed, medians)

    train_imp = preprocessing.impute_with_medians(train_df, medians)
    test_imp = preprocessing.impute_with_medians(test_df, medians)
    X_train = train_imp[config.FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_imp[config.TARGET_COLUMN].to_numpy(dtype=float)
    X_test = test_imp[config.FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test = test_imp[config.TARGET_COLUMN].to_numpy(dtype=float)

    # --- secure-aggregation-compatible sufficient statistics -----------------
    local_stats = [preprocessing.compute_local_statistics(X_train[idx])
                   for idx in silos]
    agg_stats = preprocessing.aggregate_statistics(local_stats)
    gstats = preprocessing.derive_global_standardization(agg_stats)

    silo_X = [preprocessing.standardize(X_train[idx], gstats) for idx in silos]
    silo_y = [y_train[idx] for idx in silos]
    test_X = preprocessing.standardize(X_test, gstats)

    summary = partitioning.describe_partition(train_df, silos)
    _REGIME_CACHE[key] = (silo_X, silo_y, test_X, y_test, summary)
    return _REGIME_CACHE[key]


def _model_factory(model_name: str, seed: int):
    return lambda: models.make_model(model_name, seed)

# ---------------------------------------------------------------------------
# Single strategy runs
# ---------------------------------------------------------------------------
def run_centralized(seed, silo_X, silo_y, test_X, test_y, model_name,
                    epochs=None):
    """Upper-bound reference: train on the complete pooled training partition."""
    epochs = epochs if epochs is not None else _effective_epochs()
    X = np.concatenate(silo_X, axis=0)
    y = np.concatenate(silo_y, axis=0)
    model = models.make_model(model_name, seed)
    rng = np.random.default_rng(seed)
    optimizer.train_sgd(model, X, y, rng, epochs=epochs)
    return model, metrics.evaluate_model(model, test_X, test_y)


def run_local_only(seed, silo_X, silo_y, test_X, test_y, model_name,
                   epochs=None):
    """Isolated training: one model per silo, no communication.

    Returns (per_silo_metrics, mean_metrics).
    """
    epochs = epochs if epochs is not None else _effective_epochs()
    per_silo = []
    for i in range(len(silo_X)):
        model = models.make_model(model_name, seed)
        ss = np.random.SeedSequence([seed, 9000 + i])
        rng = np.random.default_rng(ss)
        optimizer.train_sgd(model, silo_X[i], silo_y[i], rng, epochs=epochs)
        m = metrics.evaluate_model(model, test_X, test_y)
        m["silo"] = i + 1
        per_silo.append(m)
    mean = metrics.mean_metrics(
        [{k: v for k, v in m.items() if k != "silo"} for m in per_silo])
    return per_silo, mean


def run_federated_strategy(seed, silo_X, silo_y, test_X, test_y, model_name,
                           strategy, mu=0.0, sigma=None, rounds=None,
                           local_epochs=None, verbose=False):
    """Generic FedAvg / FedProx / DP-FedAvg runner; returns (model, history, metrics)."""
    rounds = rounds if rounds is not None else config.ROUNDS
    local_epochs = local_epochs if local_epochs is not None else config.LOCAL_EPOCHS
    factory = _model_factory(model_name, seed)
    eval_fn = lambda m: metrics.evaluate_model(m, test_X, test_y)
    model, history = federated.run_federated(
        factory, silo_X, silo_y, seed, model_name, strategy=strategy, mu=mu,
        sigma=sigma, rounds=rounds, local_epochs=local_epochs, evaluate=eval_fn,
        verbose=verbose)
    m = metrics.evaluate_model(model, test_X, test_y)
    return model, history, m
# ---------------------------------------------------------------------------
# Single-cell driver
# ---------------------------------------------------------------------------
def run_single(seed, regime, model_name, strategy, mu=None, sigma=None,
               df=None, rounds=None, local_epochs=None, verbose=False):
    """Execute one cell of the factorial grid; returns a result dict.

    strategy: 'centralized' | 'local_only' | 'fedavg' |
              ('fedprox', mu) | ('dp_fedavg', sigma)
    """
    t0 = time.perf_counter()
    (silo_X, silo_y, test_X, test_y, summary) = get_regime_data(seed, regime, df)

    if isinstance(strategy, tuple):
        strategy, extra = strategy
        if strategy == config.STRATEGY_FEDPROX and mu is None:
            mu = extra
        elif strategy == config.STRATEGY_DPFEDAVG and sigma is None:
            sigma = extra

    result = {
        "seed": seed, "regime": regime, "model": model_name,
        "strategy": strategy, "mu": mu if mu is not None else 0.0,
        "sigma": sigma if sigma is not None else float("nan"),
        "silo_sizes": [int(len(x)) for x in silo_X],
        "silo_prevalence": [float(np.mean(y)) for y in silo_y],
        "runtime_seconds": 0.0,
    }

    if strategy == config.STRATEGY_CENTRALIZED:
        _, m = run_centralized(seed, silo_X, silo_y, test_X, test_y,
                               model_name)
    elif strategy == config.STRATEGY_LOCAL_ONLY:
        per_silo, m = run_local_only(seed, silo_X, silo_y, test_X, test_y,
                                     model_name)
        result["per_silo"] = per_silo
    elif strategy in (config.STRATEGY_FEDAVG, config.STRATEGY_FEDPROX,
                      config.STRATEGY_DPFEDAVG):
        _, history, m = run_federated_strategy(
            seed, silo_X, silo_y, test_X, test_y, model_name, strategy,
            mu=mu, sigma=sigma, rounds=rounds, local_epochs=local_epochs,
            verbose=verbose)
        result["history"] = history
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    metric_keys = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
    final = {}
    for k in metric_keys:
        if k in m:
            final[k] = m[k]
        elif f"{k}_mean" in m:
            final[k] = m[f"{k}_mean"]
        else:
            final[k] = float("nan")
    result.update(final)
    result["runtime_seconds"] = time.perf_counter() - t0
    return result
# ---------------------------------------------------------------------------
# Full factorial
# ---------------------------------------------------------------------------
def build_task_list(seeds=None, regimes=None, models=None, variants=None):
    """Flat list of (seed, regime, model, variant) tasks."""
    seeds = seeds if seeds is not None else config.SEEDS
    regimes = regimes if regimes is not None else config.REGIMES
    models = models if models is not None else config.MODELS
    variants = variants if variants is not None else config.STRATEGY_VARIANTS
    tasks = []
    for seed in seeds:
        for regime in regimes:
            for model in models:
                for variant in variants:
                    tasks.append((seed, regime, model, variant))
    return tasks


def _run_task_cell(task):
    """Standalone worker entry point (picklable for multiprocessing)."""
    seed, regime, model_name, variant = task
    if isinstance(variant, tuple):
        strategy, extra = variant
        if strategy == config.STRATEGY_FEDPROX:
            mu, sigma = extra, None
        else:
            mu, sigma = None, extra
    else:
        strategy, mu, sigma = variant, None, None
    return run_single(seed, regime, model_name, strategy, mu=mu, sigma=sigma)


def run_full_grid(seeds=None, regimes=None, models=None, variants=None,
                  n_jobs=None, progress_every=25, smoke=False,
                  seed_subset=None, regime_subset=None, model_subset=None,
                  variant_subset=None):
    """Execute the factorial grid, optionally restricted (smoke test)."""
    if smoke:
        seeds = seed_subset or [config.SEEDS[0]]
        regimes = regime_subset or [config.REGIME_IID]
        models = model_subset or [config.MODEL_LOGISTIC]
        variants = variant_subset or [config.STRATEGY_FEDAVG]
    else:
        seeds = seeds if seeds is not None else config.SEEDS
        regimes = regimes if regimes is not None else config.REGIMES
        models = models if models is not None else config.MODELS
        variants = variants if variants is not None else config.STRATEGY_VARIANTS

    tasks = build_task_list(seeds, regimes, models, variants)
    total = len(tasks)
    print(f"Running {total} training runs ...")

    import multiprocessing as mp
    n_jobs = n_jobs if n_jobs is not None else config.N_JOBS
    n_jobs = min(n_jobs, total)

    results = []
    if n_jobs == 1 or smoke:
        results = [_run_task_cell(t) for t in tasks]
    else:
        with mp.get_context("fork").Pool(n_jobs) as pool:
            for i, res in enumerate(pool.imap_unordered(_run_task_cell, tasks)):
                results.append(res)
                if (i + 1) % progress_every == 0:
                    print(f"  completed {i + 1}/{total}")
                    _flush_results(results)
    _flush_results(results)
    return results


def _result_frame(results) -> "pd.DataFrame":
    import pandas as pd
    rows = [{k: v for k, v in r.items()
             if k not in ("per_silo", "history")} for r in results]
    return pd.DataFrame(rows)


def _flush_results(results, filename="run_results.csv"):
    """Persist machine-readable run results (atomic overwrite of the frame)."""
    import pandas as pd
    df = _result_frame(results)
    out = config.RAW_DIR / filename
    df.to_csv(out, index=False)


def persist_convergence(results, filename="convergence.csv"):
    """Per-round history rows for every federated run."""
    import pandas as pd
    rows = []
    for r in results:
        if "history" not in r or not r["history"]:
            continue
        for h in r["history"]:
            rows.append({
                "seed": r["seed"], "regime": r["regime"], "model": r["model"],
                "strategy": r["strategy"], "mu": r["mu"], "sigma": r["sigma"],
                **h,
            })
    out = config.RAW_DIR / filename
    pd.DataFrame(rows).to_csv(out, index=False)


def persist_per_silo(results, filename="per_silo_local_results.csv"):
    """Per-silo metrics for local-only runs."""
    import pandas as pd
    rows = []
    for r in results:
        if "per_silo" not in r or not r["per_silo"]:
            continue
        for m in r["per_silo"]:
            base = {k: r[k] for k in ("seed", "regime", "model", "strategy",
                                      "mu", "sigma") if k in r}
            rows.append({**base, **m})
    out = config.RAW_DIR / filename
    pd.DataFrame(rows).to_csv(out, index=False)