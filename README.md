# Federated Learning for Privacy-Preserving Healthcare Analytics Across Multiple Hospitals

A faithful, executable reproduction of the paper's experimental study:

> *"Federated Learning for Privacy-Preserving Healthcare Analytics Across Multiple Hospitals:
> A Controlled Evaluation of Statistical Heterogeneity and the Privacy-Utility Trade-off"*

The implementation is **pure NumPy / pandas / scikit-learn / SciPy / Matplotlib**.
No federated-learning framework (Flower, TensorFlow Federated, PySyft, ...) is used,
exactly as the paper states. Everything runs on CPU.

---

## 1. Project overview

The study performs a controlled, seed-locked factorial experiment on the
Framingham Heart Study dataset (n = 4,240, 15 predictors, binary 10-year CHD risk):

* **Training strategies**: Centralized (upper-bound reference), Local-only,
  FedAvg, FedProx (mu = 0.01, 0.1), DP-FedAvg (sigma = 0.1, 0.25, 0.5, 1.0, 2.0).
* **Heterogeneity regimes**: IID, Dirichlet label skew (alpha = 0.5),
  covariate skew (contiguous age bands).
* **Model families**: class-weighted logistic regression (16 parameters) and a
  15->16->1 MLP (273 parameters), both trained with mini-batch SGD.
* **Evaluation**: accuracy, precision, recall, F1, ROC-AUC, PR-AUC on a shared
  stratified 20% global holdout; paired statistics (t-test, Wilcoxon, bootstrap CIs,
  Cohen's d) with Holm correction for the 6 primary FedAvg-vs-local comparisons;
  formal client-level DP accounting and a privacy-utility analysis.

The exact configuration is in [src/config.py](src/config.py).

## 2. Research question

Does federated training (FedAvg) recover the benefit of pooled data under
controlled statistical heterogeneity — and at what measurable privacy cost —
compared with isolated per-site models? The study uses ROC-AUC as the primary
metric, plus a full metric panel at threshold 0.5.

## 3. Dataset

`data/framingham.csv` — the standard 4,240-record Framingham Heart Study CSV with:

* 15 predictors: `male, age, education, currentSmoker, cigsPerDay, BPMeds,
  prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI,
  heartRate, glucose`
* 1 target: `TenYearCHD` (644 positives, 15.2% prevalence)
* Expected missing values: glucose 388, education 105, totChol 50, BPMeds 53,
  BMI 19, cigsPerDay 29, heartRate 1.

Missing values are imputed with **training-only** variable medians before
standardization — no test information is used.

### Standardization without pooling raw records

Each silo computes local sufficient statistics `n`, `sum(x)`, `sum(x^2)`. These
are aggregated across silos to derive the global mean/variance used to
standardize every silo and the test set. This aggregation is *conceptually
compatible* with secure aggregation; actual cryptographic secure aggregation is
**NOT implemented** (see Limitations).

## 4. Installation
## 6. Configuration

All experiment configuration lives in `src/config.py`:

| parameter | value |
|---|---|
| `N_SILOS` | 5 |
| `ROUNDS` | 40 |
| `LOCAL_EPOCHS` | 2 |
| `LEARNING_RATE` | 0.1 |
| `BATCH_SIZE` | 64 |
| `POSITIVE_CLASS_WEIGHT` | 5.59 |
| `DP_CLIP_NORM` | 1.0 |
| `DP_SIGMAS` | [0.1, 0.25, 0.5, 1.0, 2.0] |
| `FEDPROX_MUS` | [0.01, 0.1] |
| `SEEDS` | [0, 1, 2, 3, 4] |
| `DELTA` (DP) | 1e-5 |
| `BOOTSTRAP_SAMPLES` | 10000 |

The data location can be overridden with the `HMS_DATASET_PATH` environment
variable.

## 7. How to run data validation

```bash
python scripts/validate_data.py
```

Checks: 4,240 rows, 15 predictors + target, 644 positives, no ID column,
expected missing counts.

## 8. How to run experiments

```bash
# full 300-run factorial (3 regimes x 2 models x 5 seeds x 10 variants)
python scripts/run_experiments.py --n-jobs 8

# single cell, e.g. FedProx mu=0.1, MLP, covariate skew, seed 3
python scripts/run_experiments.py --single 3 covariate_skew mlp fedprox

# DP-FedAvg single run
python scripts/run_experiments.py --single 0 iid logistic dp_fedavg
```

The runner logs `[Seed] / [regime] / [model] / [strategy] / Round k/40 ...`
and writes:

* `results/raw/run_results.csv` — one row per run (all six metrics, silo
  sizes/prevalences, runtime);
* `results/raw/convergence.csv` — per-round test metrics for federated runs
  (evaluation-only, never used for training);
* `results/raw/per_silo_local_results.csv` — per-silo metrics for local-only runs.

## 11. How to generate figures

```bash
python scripts/generate_figures.py            # all figures
python scripts/generate_figures.py --figure 5  # one figure
```

Figures (PNG + PDF + SVG) are written to `results/figures/`:

1. Federated training architecture
2. Outcome prevalence and record counts per silo
3. Test ROC-AUC by strategy x model x regime (mean +/- SD over seeds)
4. Federation gain (FedAvg - local-only) with bootstrap CIs and Cohen's d
5. Test ROC-AUC vs communication round (logistic FedAvg)
6. Privacy-utility trade-off (log-x epsilon vs ROC-AUC)

## 12. Expected runtime

A single 40-round federated run takes ~0.03-0.05 s (fully vectorized NumPy).
The full 300-run grid completed in ~10 s multi-core on an Apple-silicon
laptop; expect the same order of magnitude on a single core. The paper's
"~50 s per heterogeneity regime" (~2.5 min for the grid) is slower — likely a
less vectorized local-training loop. Analysis adds a few seconds
(10,000-sample paired bootstrap x 6 primary comparisons); figures ~15 s.

## 13. Reproducibility instructions

1. `python -m pytest tests/ -q` (41 tests) to verify the building blocks.
2. `python scripts/run_experiments.py --n-jobs 8` (the 300-run grid).
3. `python scripts/run_analysis.py`.
4. `python scripts/generate_figures.py`.
5. Or everything at once: `python run_all.py`.

Reproducibility guarantees: every seed controls the global stratified split,
the silo partition, the mini-batch schedule, and model initialization. Local
batching RNG is keyed by (seed, round, silo) so all strategies share the same
optimization schedule for paired comparisons. DP noise uses an independent
reproducible stream keyed by (seed, sigma).
## 15. Differences from the paper

Where the paper's stated configuration is ambiguous, or where the reproduced
results differ materially, this is documented in
`results/reproduction_report.md` (auto-generated) rather than silently changed.

Key anticipated differences (see the report for actual numbers):

* Reproduced logistic ROC-AUC values are lower (~0.70) than the paper's stated
  values (~0.729-0.732). This matches the linear-model ceiling on this exact
  dataset/split: an unregularised sklearn logistic regression on the same rows
  and features reaches ~0.70. The paper may have used a different evaluation
  convention (e.g., in-sample scoring) or additional feature processing not
  described; the held-out 20% protocol from the specification is implemented as
  specified.
* DP epsilon values follow the paper's stated Renyi formula
  `T*alpha/(2*sigma^2) + log(1/delta)/(alpha-1)` minimized over alpha > 1
  (analytic optimum: `a + 2*sqrt(a*b)` with `a = T/(2*sigma^2)`,
  `b = ln(1/delta)`). Four of five sigmas match the paper within rounding
  (441.4 / 140.7 / 50.4 / 20.2 vs 441.6 / 140.8 / 50.4 / 20.2). For
  sigma = 0.1 the exact minimum is 2303.5 vs the paper's 2315.1 (-0.5%); the
  paper's value coincides exactly with evaluating the formula at alpha = 1.1
  (2000*1.1 + 11.5129/0.1 = 2315.13), so the paper's accountant evidently used
  a coarse alpha grid that missed the true optimum. We report the exact
  minimum of the paper's own formula, which is the defensible reading.
* Communication cost assumes float32 uploads/downloads of deltas and global
  models over 40 rounds x 5 clients, counted in decimal megabytes
  (1 MB = 10^6 bytes) as in the paper (16 params -> 0.026 MB, 273 params ->
  0.437 MB); exact figures are in the report.

Ambiguities resolved toward the most defensible reading:

1. **Local-only optimization budget** is matched to the federated budget
   (40 rounds x 2 epochs = 80 SGD epochs) as an "equivalent total optimization
   budget".
2. **Dirichlet assignment** draws one per-class proportion vector per seed and
   assigns every record exactly once (no overlap, full coverage).
3. **DP noise scale** follows the paper's mechanism: Gaussian noise with
   standard deviation `sigma * C` added to the released client-weighted
   average of clipped deltas, where `sigma` is the aggregate Gaussian noise
   multiplier relative to the clip bound `C` — the SAME `sigma` used by the
   RDP accountant (`RDP(alpha) = alpha/(2*sigma^2)` per round).  The release's
   client-level L2 sensitivity is bounded by `C` because each clipped delta
   has norm `<= C` and its aggregation weight is `<= 1`.  (An earlier revision
   used `sigma * C / K`, which injects K times less noise than this
   accounting convention implies — the realized per-round RDP was then
   `K^2 = 25x` the accounted value.  The convention was corrected for
   mechanism/accountant consistency, not to fit any utility target.)
4. **RDP accounting** uses the paper's formula
   `T*alpha/(2*sigma^2) + log(1/delta)/(alpha-1)` minimized over alpha > 1.

## 14. Limitations (shared with the paper)

* **No real multi-hospital privacy.** All silos originate from one Framingham
  cohort; genuine institutional variation is not fully reproduced.
* **No cryptographic secure aggregation.** The sufficient-statistics
  aggregation is only *conceptually* compatible with secure aggregation; the
  cryptography is not implemented (the paper assumes it).
* **No defense against malicious clients, poisoning, or gradient inversion.**
* **DP is client-level accounting only** (Renyi, full participation, no
  subsampling amplification).
* Findings are specific to tabular CHD-risk prediction; they do not generalize
  to imaging, text, waveform, or other modalities.
## 9. How to run analysis

```bash
python scripts/run_analysis.py
```

Computes and writes:

* Table 1-11 (see section 10)
* Primary paired comparisons (FedAvg vs Local-only) with Holm correction
* Secondary comparisons (FedAvg vs Centralized, FedProx vs FedAvg, DP vs FedAvg)
* Cross-silo equity for local-only logistic models
* Privacy-utility table
* `results/reproduction_check.csv` and `results/reproduction_report.md`

## 10. How to reproduce all tables

Tables are written as CSV by `scripts/run_analysis.py` into `results/tables/`:

| table | file | content |
|---|---|---|
| 1 | `table1_comparative_positioning.csv` | positioning vs prior work (static) |
| 2 | `table2_predictor_descriptives.csv` | predictor descriptives (from data) |
| 3 | `table3_training_strategies.csv` | strategy descriptions (static) |
| 4 | `table4_silo_heterogeneity_<regime>.csv` | silo profile per regime (seed 0) |
| 5 | `table5_test_roc_auc.csv` | ROC-AUC by strategy/model/regime |
| 6 | `table6_primary_significance.csv` | primary paired tests |
| 7 | `table7_secondary_comparisons.csv` | secondary paired tests |
| 8 | `table8_persilo_logistic_auc.csv` | per-silo isolated logistic AUC |
| 9 | `table9_logistic_metric_panel.csv` | full logistic metric panel at 0.5 |
| 10 | `table10_privacy_utility.csv` | epsilon/utility table |
| 11 | `table11_configuration.csv` | experiment configuration (static) |

```bash
cd "HSM PROJECT"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python >= 3.9 with numpy, pandas, scikit-learn, scipy, matplotlib, pytest.

## 5. Project structure

```
data/framingham.csv
src/   config, data_loader, preprocessing, partitioning, models, optimizer,
       federated, differential_privacy, privacy_accounting, metrics,
       statistics, analysis, experiments, tables, plotting
scripts/ validate_data.py, run_experiments.py, run_analysis.py, generate_figures.py
results/ raw/ processed/ tables/ figures/
tests/  unit tests (41)
requirements.txt, run_all.py, README.md
```# Healthcare
