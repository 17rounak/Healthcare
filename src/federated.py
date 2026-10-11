"""
Federated training loops: FedAvg, FedProx and DP-FedAvg.

Shared protocol for every federated strategy (per the paper):

    SERVER: initialise global model
    FOR round = 1 .. 40:
        broadcast global parameters
        FOR each silo (all five participate):
            train locally for 2 epochs         (mini-batch SGD)
            compute parameter delta
            return delta + number of local samples
        aggregate sample-count weighted deltas  (+ Gaussian noise with std
             sigma * C for DP-FedAvg; sigma = aggregate noise multiplier
             relative to the clip bound C, see differential_privacy.py)
        update w_global
    evaluate on the untouched global test set

The update follows w_new = w_old + weighted_average(client_deltas) with
weight_i = n_i / total_training_records.

Local mini-batch shuffling is keyed on (seed, round, silo) only -- it is
identical across every strategy for a given (seed, regime, model) so that
paired comparisons share the same optimisation schedule.
"""

from __future__ import annotations

import numpy as np

from . import config
from . import optimizer
from . import differential_privacy as dp


def _local_rng(seed: int, round_idx: int, silo: int):
    ss = np.random.SeedSequence([seed, int(round_idx), int(silo)])
    return np.random.default_rng(ss)


def _dp_stream_key(seed: int, sigma: float) -> int:
    return int(seed * 100000 + round(sigma * 1000))


def run_federated(model_factory, silo_X, silo_y, seed, model_name,
                  strategy=config.STRATEGY_FEDAVG, mu=0.0, sigma=None,
                  rounds=None, local_epochs=None, evaluate=None,
                  clip_norm=None, verbose=False):
    """Run a federated training loop.

    evaluate: optional callable(model) -> dict of metrics used to record a
              per-round convergence history. This is evaluation-only and never
              influences training.
    """
    del model_name
    rounds = rounds if rounds is not None else config.ROUNDS
    local_epochs = local_epochs if local_epochs is not None else config.LOCAL_EPOCHS
    clip_norm = clip_norm if clip_norm is not None else config.DP_CLIP_NORM
    k = len(silo_X)
    sizes = np.array([X.shape[0] for X in silo_X], dtype=float)
    total = float(sizes.sum())
    weights = sizes / total

    model = model_factory()
    w = model.get_parameters()
    is_fedprox = strategy == config.STRATEGY_FEDPROX and (mu or 0.0) > 0.0
    is_dp = strategy == config.STRATEGY_DPFEDAVG and sigma is not None

    dp_rng = None
    if is_dp:
        # Independent reproducible noise stream for the DP mechanism.
        dp_rng = np.random.default_rng(_dp_stream_key(seed, sigma))

    history = []
    for r in range(1, rounds + 1):
        deltas = []
        for i in range(k):
            model.set_parameters(w)
            local_rng = _local_rng(seed, r, i)
            params = optimizer.train_sgd(
                model, silo_X[i], silo_y[i], local_rng,
                epochs=local_epochs,
                w_global=w if is_fedprox else None,
                mu=mu if is_fedprox else 0.0)
            delta = params - w
            if is_dp:
                delta, _ = dp.clip_update(delta, clip_norm)
            deltas.append(delta)

        if is_dp:
            agg, _ = dp.dp_client_aggregate(deltas, weights, sigma,
                                            clip_norm, dp_rng)
            w = w + agg
        else:
            w = w + sum(weights[i] * deltas[i] for i in range(k))
        model.set_parameters(w)

        hist_entry = {"round": int(r)}
        if evaluate is not None:
            m = evaluate(model)
            hist_entry.update({f"test_{kk}": vv for kk, vv in m.items()})
        history.append(hist_entry)

        if verbose and (r % 10 == 0 or r == rounds):
            tag = evaluate(model) if evaluate is not None else {}
            auc = tag.get("roc_auc", None)
            desc = f"Round {r}/{rounds}"
            if auc is not None:
                desc += f"  ROC-AUC={auc:.4f}"
            print(desc)

    return model, history