"""
Central configuration for the Federated Learning reproduction study.

All experiment hyperparameters live here so that no magic numbers are
scattered through the codebase.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
PROCESSED_DIR = RESULTS_DIR / "processed"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

# Make the data location configurable via an environment variable.
import os

DATASET_PATH = Path(os.environ.get("HMS_DATASET_PATH", DATA_DIR / "framingham.csv"))

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "male", "age", "education", "currentSmoker", "cigsPerDay", "BPMeds",
    "prevalentStroke", "prevalentHyp", "diabetes", "totChol", "sysBP", "diaBP",
    "BMI", "heartRate", "glucose",
]
TARGET_COLUMN = "TenYearCHD"
MISSING_TOKENS = ["NA", "NaN", "", "nan"]
EXPECTED_N_ROWS = 4240
EXPECTED_N_FEATURES = 15
EXPECTED_POSITIVE_CASES = 644

TEST_FRACTION = 0.20

# ---------------------------------------------------------------------------
# Federated setup
# ---------------------------------------------------------------------------
N_SILOS = 5
ROUNDS = 40
LOCAL_EPOCHS = 2
LEARNING_RATE = 0.1
BATCH_SIZE = 64
POSITIVE_CLASS_WEIGHT = 5.59
NEGATIVE_CLASS_WEIGHT = 1.0
INIT_SEED_OFFSET = 1000  # offset used to derive initialization RNG stream

# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------
REGIME_IID = "iid"
REGIME_LABEL_SKEW = "label_skew"
REGIME_COVARIATE_SKEW = "covariate_skew"
REGIMES = [REGIME_IID, REGIME_LABEL_SKEW, REGIME_COVARIATE_SKEW]

DIRICHLET_ALPHA = 0.5

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODEL_LOGISTIC = "logistic"
MODEL_MLP = "mlp"
MODELS = [MODEL_LOGISTIC, MODEL_MLP]

# MLP architecture
MLP_HIDDEN_UNITS = 16

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
STRATEGY_CENTRALIZED = "centralized"
STRATEGY_LOCAL_ONLY = "local_only"
STRATEGY_FEDAVG = "fedavg"
STRATEGY_FEDPROX = "fedprox"
STRATEGY_DPFEDAVG = "dp_fedavg"

# FedProx proximal term strength mu: local objective is
#   L_local(w) + (mu / 2) * ||w - w_global||^2
FEDPROX_MUS = [0.01, 0.1]

# DP-FedAvg
DP_CLIP_NORM = 1.0
DP_SIGMAS = [0.1, 0.25, 0.5, 1.0, 2.0]

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]

# The 10 strategy/configuration variants that compose the 5x3x2x10 = 300 grid.
STRATEGY_VARIANTS = [
    STRATEGY_CENTRALIZED,
    STRATEGY_LOCAL_ONLY,
    STRATEGY_FEDAVG,
    (STRATEGY_FEDPROX, 0.01),
    (STRATEGY_FEDPROX, 0.1),
    (STRATEGY_DPFEDAVG, 0.1),
    (STRATEGY_DPFEDAVG, 0.25),
    (STRATEGY_DPFEDAVG, 0.5),
    (STRATEGY_DPFEDAVG, 1.0),
    (STRATEGY_DPFEDAVG, 2.0),
]

# ---------------------------------------------------------------------------
# Privacy accounting
# ---------------------------------------------------------------------------
DP_DELTA = 1e-5
RDP_ALPHA_MIN = 1.0001
RDP_ALPHA_MAX = 100000.0
RDP_N_ALPHAS = 20000

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_CI = 0.95
PRACTICAL_AUC_DELTA = 0.01  # 1 percentage point of ROC-AUC

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
THRESHOLD = 0.5
N_JOBS = int(os.environ.get("HMS_N_JOBS", os.cpu_count() or 1))

# Directories to create on import
for _d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, TABLES_DIR, FIGURES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)