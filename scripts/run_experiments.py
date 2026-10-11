#!/usr/bin/env python
"""Run the federated-learning experiment grid.

Full factorial: 3 regimes x 2 models x 5 seeds x 10 strategy variants = 300 runs.
Use --smoke for a 1-cell sanity run (1 seed, IID, logistic, FedAvg, 2 rounds).

Usage:
    python scripts/run_experiments.py                # full 300-run grid
    python scripts/run_experiments.py --smoke        # tiny smoke test
    python scripts/run_experiments.py --n-jobs 4     # 4 parallel workers
    python scripts/run_experiments.py --single 0 iid logistic fedavg
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import experiments as E
from src import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="single-cell smoke test (seed 0, iid, logistic, "
                         "FedAvg, 2 rounds)")
    ap.add_argument("--single", nargs=4, metavar=("SEED", "REGIME", "MODEL",
                                                  "STRATEGY"),
                    help="run a single cell, e.g. 0 iid logistic fedavg")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()

    if args.single:
        seed, regime, model, strategy = args.single
        if strategy == "fedprox":
            variant = ("fedprox", 0.01)
        elif strategy == "dp_fedavg":
            variant = ("dp_fedavg", 0.5)
        else:
            variant = strategy
        res = E.run_single(int(seed), regime, model, variant,
                           rounds=args.rounds, verbose=True)
        print("\n=== single-cell result ===")
        for k, v in res.items():
            if k in ("per_silo", "history"):
                continue
            print(f"  {k}: {round(v, 5) if isinstance(v, float) else v}")
        return

    print(f"Full experiment grid (smoke={args.smoke})")
    if args.smoke:
        results = [E.run_single(0, "iid", "logistic", "fedavg",
                                rounds=args.rounds or 2)]
        print("\n=== SMOKE RESULT ===")
        for k, v in results[0].items():
            if k in ("per_silo", "history"):
                continue
            print(f"  {k}: {round(v, 5) if isinstance(v, float) else v}")
        print(f"  history rounds: {len(results[0]['history'])}")
        print("Smoke mode: raw result files are NOT overwritten "
              "(run without --smoke for the full 300-run grid).")
        return
    results = E.run_full_grid(n_jobs=args.n_jobs, smoke=False)
    E.persist_convergence(results)
    E.persist_per_silo(results)
    n = len(results)
    print(f"\nCompleted {n} runs in {time.time() - t0:.1f}s "
          f"(mean {time.time() - t0:.1f}/{n} per run)")
    print(f"Results written to {config.RAW_DIR / 'run_results.csv'}")


if __name__ == "__main__":
    main()