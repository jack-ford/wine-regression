"""Shared paths and reproducible experiment settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = ROOT / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 4
UCI_FILES = {
    "red": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "white": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
    "names": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names",
}
