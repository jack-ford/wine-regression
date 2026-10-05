"""Load, split, and summarize the wine data."""
from __future__ import annotations

import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import (
    CV_FOLDS,
    DATA_PROCESSED,
    DATA_RAW,
    FIGURES,
    MODELS,
    RANDOM_STATE,
    RESULTS,
    TEST_SIZE,
    UCI_FILES,
)


def ensure_dirs() -> None:
    for path in [DATA_RAW, DATA_PROCESSED, RESULTS, FIGURES, MODELS]:
        path.mkdir(parents=True, exist_ok=True)



def download_data() -> None:
    for name, url in UCI_FILES.items():
        suffix = ".names" if name == "names" else ".csv"
        dest = DATA_RAW / f"winequality-{name}{suffix}"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, dest)



def load_combined_data() -> pd.DataFrame:
    red = pd.read_csv(DATA_RAW / "winequality-red.csv", sep=";")
    red["wine_type"] = 0
    white = pd.read_csv(DATA_RAW / "winequality-white.csv", sep=";")
    white["wine_type"] = 1
    combined = pd.concat([red, white], ignore_index=True)
    return combined



def deduplicate_data(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    return cleaned, before - len(cleaned)



def make_split(df: pd.DataFrame):
    X = df.drop(columns=["quality"])
    y = df["quality"].astype(int)
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )



def make_cv(y: pd.Series):
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return list(splitter.split(np.zeros(len(y)), y))



def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    features = df.drop(columns=["quality"])
    numeric = features.select_dtypes(include=np.number)
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    iqr_counts = ((numeric < lower) | (numeric > upper)).sum().sort_values(ascending=False)
    median = numeric.median()
    mad = (numeric - median).abs().median().replace(0, np.nan)
    robust_z_counts = (((numeric - median).abs() / (1.4826 * mad)) > 3.5).sum().sort_values(ascending=False)
    summary = pd.DataFrame({"IQR_outlier_count": iqr_counts, "robust_z_count": robust_z_counts})
    summary["IQR_outlier_pct"] = summary["IQR_outlier_count"] / len(df)
    summary.to_csv(RESULTS / "outlier_feature_summary.csv")
    return summary
