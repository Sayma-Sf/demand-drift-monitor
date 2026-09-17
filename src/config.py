"""Project-wide settings: paths, the simulated production timeline, model and monitor settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "reports" / "figures"
ARTIFACTS_DIR = ROOT / "artifacts"  # lineage/ is committed; models/ and predictions/ are rebuilt

# The first model trains on 2013-2015. Production then runs month by month through 2017.
TRAIN_START = "2013-01-01"
PRODUCTION_START = "2016-01-01"
PRODUCTION_END = "2017-12-31"

# How the forecast is kept up to date.
#   static          trained once, never touched again
#   drift_triggered retrained at the end of any month whose sales drifted from its training data
#   monthly         retrained at the end of every month, drift or not
STRATEGIES = ["static", "drift_triggered", "monthly"]

# Drift test: Evidently's normalised Wasserstein distance on daily sales, comparing the month
# that just ended with the same calendar month in the model's most recent training year.
# 0.1 is Evidently's default threshold for this test.
DRIFT_METHOD = "wasserstein"
DRIFT_THRESHOLD = 0.1
NAIVE_REFERENCE_ROWS = 60_000  # sample of the whole training window, for the naive comparison

# Gradient-boosted trees on store, item and calendar features.
SEED = 42
XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "objective": "reg:squarederror",
}
