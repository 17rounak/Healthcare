#!/usr/bin/env python
"""End-to-end pipeline driver.

    python run_all.py               # validate, test, run experiments, analyse, figure
    python run_all.py --smoke       # smoke-test everything on a tiny grid
    python run_all.py --skip-experiments

This script must be invoked from the project root.
"""
import argparse
import subprocess
import sys
import time
import os
from pathlib import Path

import sys as _sys

ROOT = Path(__file__).resolve().parent
_sys.path.insert(0, str(ROOT))


def run(cmd, **kw):
    print(f"\n>>> {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), **kw)
    assert r.returncode == 0, f"command failed: {cmd}"
    print(f"    (took {time.time() - t0:.1f}s)")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-experiments", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()

    run([sys.executable, "scripts/validate_data.py"])

    run([sys.executable, "-m", "pytest", "tests/", "-q"])

    if not args.skip_experiments:
        if args.smoke:
            run([sys.executable, "scripts/run_experiments.py", "--smoke",
                 "--rounds", "2"])
        else:
            cmd = [sys.executable, "scripts/run_experiments.py"]
            if args.n_jobs:
                cmd += ["--n-jobs", str(args.n_jobs)]
            run(cmd)
    else:
        print("\n>>> skipping experiments (--skip-experiments)")

    run([sys.executable, "scripts/run_analysis.py"])
    run([sys.executable, "scripts/generate_figures.py"])

    print("\n✓ All pipeline steps completed.")


if __name__ == "__main__":
    main()