"""Run the reproducible Wine Quality experiment."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import joblib
import pandas as pd

# Support both direct script execution and python -m src.run_project.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_PROCESSED, RESULTS, MODELS
from src.data import (
    deduplicate_data,
    download_data,
    ensure_dirs,
    load_combined_data,
    make_cv,
    make_split,
    outlier_summary,
)
from src.modeling import (
    candidate_models,
    compare_models,
    evaluate_cv,
    feature_importance,
    final_evaluation,
    run_outlier_experiments,
    secondary_classification,
    spec_family,
    tune_model,
)
from src.reporting import make_eda_outputs, make_model_plots, write_docs


def main() -> None:
    ensure_dirs()
    download_data()
    raw = load_combined_data()
    raw_summary = make_eda_outputs(raw)
    deduped, dedup_removed = deduplicate_data(raw)
    outlier_summary(deduped)

    X_train, X_test, y_train, y_test = make_split(deduped)
    X_train.to_csv(DATA_PROCESSED / "X_train.csv", index=False)
    X_test.to_csv(DATA_PROCESSED / "X_test.csv", index=False)
    y_train.to_csv(DATA_PROCESSED / "y_train.csv", index=False)
    y_test.to_csv(DATA_PROCESSED / "y_test.csv", index=False)

    splits = make_cv(y_train)
    comparison = compare_models(X_train, y_train, splits)
    best_initial_name = comparison.iloc[0]["model"]
    best_family = spec_family(best_initial_name)
    if best_family not in {"extra_trees", "lightgbm", "xgboost", "hist_gb"}:
        raise ValueError(f"No tuning search configured for {best_initial_name}")
    final_engineered = bool(comparison.iloc[0]["engineered"])
    best_spec = next(s for s in candidate_models() if s["name"] == best_initial_name)
    outliers = run_outlier_experiments(best_spec, X_train, y_train, splits)

    search = tune_model(best_family, final_engineered, X_train, y_train, splits)
    final_cv = evaluate_cv(
        f"Tuned final {best_family} ({'engineered' if final_engineered else 'raw'} features)",
        search.best_estimator_,
        X_train,
        y_train,
        splits,
    )
    pd.DataFrame([final_cv]).to_csv(RESULTS / "final_model_cv.csv", index=False)
    final_model = search.best_estimator_
    final_model.fit(X_train, y_train)
    joblib.dump(final_model, MODELS / "final_wine_quality_pipeline.joblib")

    final_metrics, predictions = final_evaluation(final_model, X_train, y_train, X_test, y_test)
    class_metrics = secondary_classification(X_train, y_train, X_test, y_test, engineered=final_engineered)
    importance = feature_importance(final_model, X_test, y_test)
    make_model_plots(comparison, predictions, importance)
    write_docs(
        raw_summary=raw_summary,
        dedup_removed=dedup_removed,
        model_comparison=comparison,
        outliers=outliers,
        final_cv=final_cv,
        final_metrics=final_metrics,
        class_metrics=class_metrics,
        importance=importance,
        best_family=best_family,
        final_engineered=final_engineered,
        best_params=search.best_params_,
    )
    print("Done. Final test metrics:")
    print(json.dumps(final_metrics["test"], indent=2))
    print("Secondary classification metrics:")
    print(json.dumps(class_metrics, indent=2))



if __name__ == "__main__":
    main()
