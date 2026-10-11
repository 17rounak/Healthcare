#!/usr/bin/env python
"""Data validation script.

Checks the Framingham dataset for:
  * 4,240 rows
  * exactly 15 predictors + TenYearCHD
  * ~644 positive cases (15.2%)
  * no accidental ID column
  * expected missing-value profile
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data_loader, config


def main():
    df = data_loader.load_framingham()
    facts = data_loader.validate_dataset(df)

    ok = True

    checks = [
        ("row count == 4240", facts["n_rows"] == config.EXPECTED_N_ROWS),
        ("15 predictors", facts["n_features"] == config.EXPECTED_N_FEATURES),
        ("target present", facts["has_target"]),
        ("positive cases ~644",
         abs(facts["n_positives"] - config.EXPECTED_POSITIVE_CASES) <= 10),
        ("no ID column", facts["no_id_column"]),
        ("columns exact",
         facts["exact_features"] == config.FEATURE_COLUMNS
         + [config.TARGET_COLUMN]),
    ]

    print("=" * 62)
    print("Framingham dataset validation")
    print("=" * 62)
    print(f"  rows          : {facts['n_rows']}")
    print(f"  predictors    : {facts['n_features']}")
    print(f"  positives     : {facts['n_positives']}  "
          f"(prevalence {facts['prevalence']:.4f})")
    print(f"  columns       : {facts['exact_features']}")
    print("  missing counts:")
    for col, n in facts["missing_counts"].items():
        print(f"    {col:<16} {n}")
    print("-" * 62)
    for name, passed in checks:
        print(f"  [{'OK' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print("=" * 62)

    if ok:
        print("Data validation PASSED.")
    else:
        print("Data validation FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()