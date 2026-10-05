"""Model definitions, cross-validation, tuning, and held-out evaluation."""
from __future__ import annotations

import json
import math
import time
import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RESULTS, MODELS, RANDOM_STATE
from src.features import QuantileClipper, WineFeatureEngineer

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMRegressor = None
    LGBMClassifier = None

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


def rmse(y_true, y_pred) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))



def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MedianAE": median_absolute_error(y_true, y_pred),
        "Within_0.5": float(np.mean(np.abs(y_pred - y_true) <= 0.5)),
        "Within_1.0": float(np.mean(np.abs(y_pred - y_true) <= 1.0)),
        "Rounded_Accuracy": accuracy_score(y_true, np.rint(y_pred).clip(y_true.min(), y_true.max()).astype(int)),
    }



def make_pipeline(model, engineered: bool = False, scaler: str | None = None, clip: bool = False) -> Pipeline:
    steps = [("features", WineFeatureEngineer(enabled=engineered))]
    if clip:
        steps.append(("clip", QuantileClipper(0.01, 0.99)))
    steps.append(("imputer", SimpleImputer(strategy="median")))
    if scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler is not None:
        raise ValueError(f"Unknown scaler: {scaler}")
    steps.append(("model", model))
    return Pipeline(steps)



def candidate_models() -> list[dict]:
    specs = [
        {"name": "Dummy mean baseline", "model": DummyRegressor(strategy="mean"), "engineered": False, "scaler": None},
        {"name": "Linear Regression raw", "model": LinearRegression(), "engineered": False, "scaler": "standard"},
        {"name": "Ridge raw", "model": Ridge(alpha=10.0), "engineered": False, "scaler": "standard"},
        {
            "name": "ElasticNet raw",
            "model": ElasticNet(alpha=0.002, l1_ratio=0.2, max_iter=20000, random_state=RANDOM_STATE),
            "engineered": False,
            "scaler": "standard",
        },
        {
            "name": "Random Forest raw",
            "model": RandomForestRegressor(
                n_estimators=350,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "engineered": False,
            "scaler": None,
        },
        {
            "name": "Extra Trees raw",
            "model": ExtraTreesRegressor(
                n_estimators=450,
                min_samples_leaf=1,
                max_features=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "engineered": False,
            "scaler": None,
        },
        {
            "name": "HistGradientBoosting raw",
            "model": HistGradientBoostingRegressor(
                max_iter=550,
                learning_rate=0.045,
                max_leaf_nodes=31,
                l2_regularization=0.02,
                random_state=RANDOM_STATE,
            ),
            "engineered": False,
            "scaler": None,
        },
        {
            "name": "GradientBoosting raw",
            "model": GradientBoostingRegressor(
                n_estimators=450,
                learning_rate=0.045,
                max_depth=3,
                min_samples_leaf=8,
                subsample=0.85,
                random_state=RANDOM_STATE,
            ),
            "engineered": False,
            "scaler": None,
        },
        {
            "name": "Ridge engineered",
            "model": Ridge(alpha=20.0),
            "engineered": True,
            "scaler": "standard",
        },
        {
            "name": "Random Forest engineered",
            "model": RandomForestRegressor(
                n_estimators=350,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "engineered": True,
            "scaler": None,
        },
        {
            "name": "Extra Trees engineered",
            "model": ExtraTreesRegressor(
                n_estimators=450,
                min_samples_leaf=1,
                max_features=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "engineered": True,
            "scaler": None,
        },
        {
            "name": "HistGradientBoosting engineered",
            "model": HistGradientBoostingRegressor(
                max_iter=550,
                learning_rate=0.045,
                max_leaf_nodes=31,
                l2_regularization=0.02,
                random_state=RANDOM_STATE,
            ),
            "engineered": True,
            "scaler": None,
        },
    ]
    if XGBRegressor is not None:
        specs.extend(
            [
                {
                    "name": "XGBoost raw",
                    "model": XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=650,
                        learning_rate=0.035,
                        max_depth=3,
                        min_child_weight=3,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=4.0,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        eval_metric="rmse",
                        verbosity=0,
                    ),
                    "engineered": False,
                    "scaler": None,
                },
                {
                    "name": "XGBoost engineered",
                    "model": XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=650,
                        learning_rate=0.035,
                        max_depth=3,
                        min_child_weight=3,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=4.0,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        eval_metric="rmse",
                        verbosity=0,
                    ),
                    "engineered": True,
                    "scaler": None,
                },
            ]
        )
    if LGBMRegressor is not None:
        specs.extend(
            [
                {
                    "name": "LightGBM raw",
                    "model": LGBMRegressor(
                        n_estimators=750,
                        learning_rate=0.035,
                        num_leaves=31,
                        min_child_samples=18,
                        subsample=0.9,
                        subsample_freq=1,
                        colsample_bytree=0.9,
                        reg_lambda=2.0,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                    "engineered": False,
                    "scaler": None,
                },
                {
                    "name": "LightGBM engineered",
                    "model": LGBMRegressor(
                        n_estimators=750,
                        learning_rate=0.035,
                        num_leaves=31,
                        min_child_samples=18,
                        subsample=0.9,
                        subsample_freq=1,
                        colsample_bytree=0.9,
                        reg_lambda=2.0,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                    "engineered": True,
                    "scaler": None,
                },
            ]
        )
    return specs



def spec_family(spec_name: str) -> str | None:
    if "LightGBM" in spec_name and LGBMRegressor is not None:
        return "lightgbm"
    if "XGBoost" in spec_name and XGBRegressor is not None:
        return "xgboost"
    if "Extra Trees" in spec_name:
        return "extra_trees"
    if "HistGradientBoosting" in spec_name:
        return "hist_gb"
    if "Random Forest" in spec_name:
        return "random_forest"
    if "Ridge" in spec_name:
        return "ridge"
    return None



def evaluate_cv(name: str, pipe: Pipeline, X: pd.DataFrame, y: pd.Series, splits: list) -> dict:
    fold_rows = []
    start = time.perf_counter()
    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        estimator = clone(pipe)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        estimator.fit(X_train, y_train)
        train_pred = estimator.predict(X_train)
        val_pred = estimator.predict(X_val)
        fold_rows.append(
            {
                "fold": fold,
                "train_RMSE": rmse(y_train, train_pred),
                "CV_RMSE": rmse(y_val, val_pred),
                "CV_MAE": mean_absolute_error(y_val, val_pred),
                "CV_R2": r2_score(y_val, val_pred),
            }
        )
    elapsed = time.perf_counter() - start
    fold_frame = pd.DataFrame(fold_rows)
    row = {
        "model": name,
        "CV_RMSE_mean": fold_frame["CV_RMSE"].mean(),
        "CV_RMSE_std": fold_frame["CV_RMSE"].std(ddof=1),
        "CV_MAE_mean": fold_frame["CV_MAE"].mean(),
        "CV_R2_mean": fold_frame["CV_R2"].mean(),
        "Train_RMSE_mean": fold_frame["train_RMSE"].mean(),
        "Train_CV_gap_RMSE": fold_frame["CV_RMSE"].mean() - fold_frame["train_RMSE"].mean(),
        "training_time_sec": elapsed,
    }
    return row



def compare_models(X_train: pd.DataFrame, y_train: pd.Series, splits: list) -> pd.DataFrame:
    rows = []
    for spec in candidate_models():
        print(f"CV: {spec['name']}")
        pipe = make_pipeline(spec["model"], engineered=spec["engineered"], scaler=spec["scaler"])
        row = evaluate_cv(spec["name"], pipe, X_train, y_train, splits)
        row["engineered"] = spec["engineered"]
        row["scaler"] = spec["scaler"] or "none"
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("CV_RMSE_mean").reset_index(drop=True)
    frame.to_csv(RESULTS / "model_comparison.csv", index=False)
    return frame



def evaluate_isolation_filter(base_pipe: Pipeline, X: pd.DataFrame, y: pd.Series, splits: list, label: str) -> dict:
    rows = []
    start = time.perf_counter()
    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        detector = IsolationForest(contamination=0.03, random_state=RANDOM_STATE, n_estimators=250)
        keep = detector.fit_predict(X_train) == 1
        estimator = clone(base_pipe)
        estimator.fit(X_train.loc[keep], y_train.loc[keep])
        pred = estimator.predict(X_val)
        rows.append(
            {
                "fold": fold,
                "removed_train_rows": int((~keep).sum()),
                "CV_RMSE": rmse(y_val, pred),
                "CV_MAE": mean_absolute_error(y_val, pred),
                "CV_R2": r2_score(y_val, pred),
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "model": f"{label} contender + IsolationForest train-fold filtering",
        "CV_RMSE_mean": frame["CV_RMSE"].mean(),
        "CV_RMSE_std": frame["CV_RMSE"].std(ddof=1),
        "CV_MAE_mean": frame["CV_MAE"].mean(),
        "CV_R2_mean": frame["CV_R2"].mean(),
        "mean_removed_train_rows": frame["removed_train_rows"].mean(),
        "training_time_sec": time.perf_counter() - start,
    }



def run_outlier_experiments(best_spec: dict, X_train: pd.DataFrame, y_train: pd.Series, splits: list) -> pd.DataFrame:
    engineered = bool(best_spec["engineered"])
    label = "engineered" if engineered else "raw"
    base = make_pipeline(best_spec["model"], engineered=engineered, scaler=best_spec["scaler"])
    clipped = make_pipeline(best_spec["model"], engineered=engineered, scaler=best_spec["scaler"], clip=True)
    rows = [
        evaluate_cv(f"{label} contender + no outlier treatment", base, X_train, y_train, splits),
        evaluate_cv(f"{label} contender + 1/99% winsorization", clipped, X_train, y_train, splits),
        evaluate_isolation_filter(base, X_train, y_train, splits, label),
    ]
    frame = pd.DataFrame(rows).sort_values("CV_RMSE_mean").reset_index(drop=True)
    frame.to_csv(RESULTS / "outlier_experiments.csv", index=False)
    return frame



def tuning_pipeline_and_params(family: str, engineered: bool):
    if family == "lightgbm":
        pipe = make_pipeline(
            LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
            engineered=engineered,
            scaler=None,
        )
        params = {
            "model__n_estimators": randint(450, 1200),
            "model__learning_rate": loguniform(0.015, 0.08),
            "model__num_leaves": randint(12, 64),
            "model__min_child_samples": randint(8, 45),
            "model__subsample": uniform(0.7, 0.3),
            "model__colsample_bytree": uniform(0.65, 0.35),
            "model__reg_alpha": loguniform(1e-3, 3.0),
            "model__reg_lambda": loguniform(0.05, 12.0),
        }
    elif family == "xgboost":
        pipe = make_pipeline(
            XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="rmse",
                verbosity=0,
            ),
            engineered=engineered,
            scaler=None,
        )
        params = {
            "model__n_estimators": randint(350, 1000),
            "model__learning_rate": loguniform(0.015, 0.08),
            "model__max_depth": randint(2, 6),
            "model__min_child_weight": randint(1, 10),
            "model__subsample": uniform(0.65, 0.35),
            "model__colsample_bytree": uniform(0.65, 0.35),
            "model__reg_alpha": loguniform(1e-4, 1.5),
            "model__reg_lambda": loguniform(0.2, 12.0),
        }
    elif family == "hist_gb":
        pipe = make_pipeline(
            HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            engineered=engineered,
            scaler=None,
        )
        params = {
            "model__max_iter": randint(250, 950),
            "model__learning_rate": loguniform(0.015, 0.09),
            "model__max_leaf_nodes": randint(12, 64),
            "model__min_samples_leaf": randint(8, 45),
            "model__l2_regularization": loguniform(1e-4, 2.0),
        }
    else:
        pipe = make_pipeline(
            ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            engineered=engineered,
            scaler=None,
        )
        params = {
            "model__n_estimators": randint(350, 950),
            "model__max_features": uniform(0.45, 0.55),
            "model__min_samples_leaf": randint(1, 5),
            "model__min_samples_split": randint(2, 10),
            "model__max_depth": [None, 12, 16, 20, 28],
        }
    return pipe, params



def tune_model(family: str, engineered: bool, X_train: pd.DataFrame, y_train: pd.Series, splits: list):
    pipe, params = tuning_pipeline_and_params(family, engineered=engineered)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=params,
        n_iter=28,
        scoring="neg_root_mean_squared_error",
        cv=splits,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cv_results.to_csv(RESULTS / "tuning_results.csv", index=False)
    with open(RESULTS / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(search.best_params_, f, indent=2, default=str)
    return search



def final_evaluation(model, X_train, y_train, X_test, y_test) -> tuple[dict, pd.DataFrame]:
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    final = {
        "train": regression_metrics(y_train, train_pred),
        "test": regression_metrics(y_test, test_pred),
    }
    pred_frame = X_test.copy()
    pred_frame["quality"] = y_test.values
    pred_frame["predicted_quality"] = test_pred
    pred_frame["residual"] = pred_frame["quality"] - pred_frame["predicted_quality"]
    pred_frame.to_csv(RESULTS / "test_predictions.csv", index=False)
    with open(RESULTS / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    return final, pred_frame



def secondary_classification(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    engineered: bool,
) -> dict:
    threshold = 7
    y_train_bin = (y_train >= threshold).astype(int)
    y_true_bin = (y_test >= threshold).astype(int)

    if LGBMClassifier is not None:
        classifier = make_pipeline(
            LGBMClassifier(
                n_estimators=500,
                learning_rate=0.035,
                num_leaves=24,
                min_child_samples=18,
                subsample=0.9,
                subsample_freq=1,
                colsample_bytree=0.9,
                reg_lambda=3.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            ),
            engineered=engineered,
            scaler=None,
        )
        model_name = "LightGBMClassifier with balanced class weights"
    else:
        classifier = make_pipeline(
            ExtraTreesClassifier(
                n_estimators=650,
                max_features=0.75,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            engineered=engineered,
            scaler=None,
        )
        model_name = "ExtraTreesClassifier with balanced class weights"

    classifier.fit(X_train, y_train_bin)
    y_score = classifier.predict_proba(X_test)[:, 1]
    y_pred_bin = (y_score >= 0.5).astype(int)
    joblib.dump(classifier, MODELS / "secondary_high_quality_classifier.joblib")
    metrics = {
        "definition": f"high_quality = 1 if quality >= {threshold}",
        "model": model_name,
        "decision_threshold": 0.5,
        "positive_rate_test": float(y_true_bin.mean()),
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "roc_auc": roc_auc_score(y_true_bin, y_score),
        "confusion_matrix": confusion_matrix(y_true_bin, y_pred_bin).tolist(),
    }
    with open(RESULTS / "classification_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics



def feature_importance(fitted_pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    result = permutation_importance(
        fitted_pipe,
        X_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=12,
        random_state=RANDOM_STATE,
        # Trees already parallelize internally; avoid nested joblib workers.
        n_jobs=1,
    )
    frame = pd.DataFrame(
        {
            "feature": X_test.columns,
            "permutation_importance_rmse_increase": result.importances_mean,
            "std": result.importances_std,
        }
    ).sort_values("permutation_importance_rmse_increase", ascending=False)
    frame.to_csv(RESULTS / "permutation_importance.csv", index=False)
    return frame
