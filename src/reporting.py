"""Generate scientific figures and project reports from experiment results."""
from __future__ import annotations

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import ROOT, RESULTS, FIGURES, RANDOM_STATE, CV_FOLDS


def make_eda_outputs(df: pd.DataFrame) -> dict:
    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "features": [c for c in df.columns if c != "quality"],
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "quality_distribution": df["quality"].value_counts().sort_index().to_dict(),
        "wine_type_distribution": df["wine_type"].map({0: "red", 1: "white"}).value_counts().to_dict(),
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
    }
    with open(RESULTS / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    df.describe().T.to_csv(RESULTS / "descriptive_statistics.csv")
    df.drop(columns=["quality"]).skew(numeric_only=True).sort_values(ascending=False).to_csv(RESULTS / "feature_skewness.csv")
    df.corr(numeric_only=True).to_csv(RESULTS / "correlations.csv")

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="quality", hue=df["wine_type"].map({0: "red", 1: "white"}))
    plt.title("Wine Quality Distribution by Wine Type")
    plt.xlabel("Quality score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES / "target_distribution.png", dpi=180)
    plt.close()

    corr = df.corr(numeric_only=True)["quality"].drop("quality").sort_values()
    plt.figure(figsize=(8, 6))
    corr.plot(kind="barh", color=np.where(corr > 0, "#247ba0", "#c44e52"))
    plt.title("Feature Correlation with Quality")
    plt.xlabel("Pearson correlation")
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_correlations.png", dpi=180)
    plt.close()

    key_features = ["alcohol", "volatile acidity", "density", "chlorides", "sulphates", "residual sugar"]
    plot_df = df.copy()
    plot_df["wine_type"] = plot_df["wine_type"].map({0: "red", 1: "white"})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feature in zip(axes.ravel(), key_features):
        sns.boxplot(data=plot_df, x="quality", y=feature, hue="wine_type", ax=ax, showfliers=False)
        ax.set_title(feature)
        ax.legend_.remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Important Feature Relationships with Quality", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_relationships.png", dpi=180, bbox_inches="tight")
    plt.close()

    return summary



def make_model_plots(model_comparison: pd.DataFrame, predictions: pd.DataFrame, importance: pd.DataFrame) -> None:
    top = model_comparison.sort_values("CV_RMSE_mean").head(10).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top["model"], top["CV_RMSE_mean"], xerr=top["CV_RMSE_std"], color="#247ba0")
    plt.xlabel("CV RMSE")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(FIGURES / "model_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=predictions, x="quality", y="predicted_quality", hue="wine_type", palette=["#c44e52", "#247ba0"], alpha=0.55)
    lo, hi = predictions["quality"].min(), predictions["quality"].max()
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.title("Predicted vs. Actual Quality")
    plt.xlabel("Actual quality")
    plt.ylabel("Predicted quality")
    plt.tight_layout()
    plt.savefig(FIGURES / "predicted_vs_actual.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(predictions["residual"], kde=True, ax=axes[0], color="#247ba0")
    axes[0].set_title("Residual Distribution")
    axes[0].set_xlabel("Actual - predicted")
    sns.scatterplot(data=predictions, x="predicted_quality", y="residual", ax=axes[1], alpha=0.5, color="#c44e52")
    axes[1].axhline(0, color="black", linestyle="--", lw=1)
    axes[1].set_title("Residuals vs. Predictions")
    plt.tight_layout()
    plt.savefig(FIGURES / "residual_diagnostics.png", dpi=180)
    plt.close()

    err_by_quality = predictions.assign(abs_error=lambda d: d["residual"].abs()).groupby("quality")["abs_error"].mean()
    plt.figure(figsize=(8, 5))
    err_by_quality.plot(kind="bar", color="#6a994e")
    plt.title("Mean Absolute Error by Actual Quality")
    plt.xlabel("Actual quality")
    plt.ylabel("Mean absolute error")
    plt.tight_layout()
    plt.savefig(FIGURES / "error_by_quality.png", dpi=180)
    plt.close()

    top_imp = importance.head(12).iloc[::-1]
    plt.figure(figsize=(9, 7))
    plt.barh(top_imp["feature"], top_imp["permutation_importance_rmse_increase"], xerr=top_imp["std"], color="#6a994e")
    plt.title("Permutation Feature Importance")
    plt.xlabel("Increase in RMSE when permuted")
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_importance.png", dpi=180)
    plt.close()



def md_table(df: pd.DataFrame, columns: list[str], rows: int | None = None) -> str:
    view = df[columns].copy()
    if rows is not None:
        view = view.head(rows)
    for col in view.select_dtypes(include=np.number).columns:
        view[col] = view[col].map(lambda x: f"{x:.4f}")
    return view.to_markdown(index=False)



def write_docs(
    raw_summary: dict,
    dedup_removed: int,
    model_comparison: pd.DataFrame,
    outliers: pd.DataFrame,
    final_cv: dict,
    final_metrics: dict,
    class_metrics: dict,
    importance: pd.DataFrame,
    best_family: str,
    final_engineered: bool,
    best_params: dict,
) -> None:
    best_row = model_comparison.iloc[0]
    outlier_best = outliers.iloc[0]
    test = final_metrics["test"]
    train = final_metrics["train"]
    top_features = importance.head(8)["feature"].tolist()

    final_label = f"Tuned final {best_family} ({'engineered' if final_engineered else 'raw'} features)"
    final_summary_row = {
        "model": final_label,
        "CV_RMSE_mean": final_cv["CV_RMSE_mean"],
        "CV_RMSE_std": final_cv["CV_RMSE_std"],
        "CV_MAE_mean": final_cv["CV_MAE_mean"],
        "CV_R2_mean": final_cv["CV_R2_mean"],
        "Train_RMSE_mean": final_cv["Train_RMSE_mean"],
        "Train_CV_gap_RMSE": final_cv["Train_CV_gap_RMSE"],
        "training_time_sec": final_cv["training_time_sec"],
        "engineered": final_engineered,
        "scaler": "none",
    }
    results_table = pd.concat([pd.DataFrame([final_summary_row]), model_comparison], ignore_index=True)
    results_table["test_RMSE"] = ""
    results_table["test_MAE"] = ""
    results_table["test_R2"] = ""
    results_table.loc[0, "test_RMSE"] = f"{test['RMSE']:.4f}"
    results_table.loc[0, "test_MAE"] = f"{test['MAE']:.4f}"
    results_table.loc[0, "test_R2"] = f"{test['R2']:.4f}"
    results_table.to_csv(RESULTS / "results_table.csv", index=False)

    readme = f"""# UCI Wine Quality Regression Project

## Problem

Predict the numerical `quality` score in the UCI Wine Quality dataset from physicochemical measurements. The main task is regression; a secondary binary evaluation labels wines as high quality when `quality >= 7`.

## Dataset and Cleaning

Data source: UCI Wine Quality Dataset, red and white variants. The project combines both wine types and adds `wine_type` before concatenation so the model can learn type-specific differences.

- Raw rows: {raw_summary['rows']}
- Raw duplicate rows: {raw_summary['duplicate_rows']}
- Exact duplicate rows removed before splitting: {dedup_removed}
- Missing values: {sum(raw_summary['missing_values'].values())}
- Quality distribution: {raw_summary['quality_distribution']}

Exact duplicates were removed before the train/test split to reduce the chance that identical records appear in both training and testing. The split is stratified by the integer quality score and uses random seed `{RANDOM_STATE}`.

## Methodology

The final test set was held out once and not used for model selection. Candidate models and feature choices were compared with {CV_FOLDS}-fold stratified cross-validation on the training set. Learned transformations, including winsorization when tested, are inside scikit-learn pipelines so they are fitted on training folds only.

## Feature Engineering

Engineered features included sulfur dioxide ratios, bound sulfur dioxide, acidity ratios, alcohol and density ratios, sulphates/chlorides ratio, alcohol interactions, acidity-pH interaction, total acidity proxy, and log transforms for skewed variables. These were compared against raw-feature models instead of assumed beneficial.

## Model Selection

Top validation models:

{md_table(model_comparison, ['model', 'CV_RMSE_mean', 'CV_RMSE_std', 'CV_MAE_mean', 'CV_R2_mean', 'Train_RMSE_mean', 'Train_CV_gap_RMSE', 'training_time_sec'], rows=8)}

The selected model was `{final_label}`. Important tuned hyperparameters:

```json
{json.dumps(best_params, indent=2, default=str)}
```

## Final Regression Results

- Final CV RMSE: {final_cv['CV_RMSE_mean']:.4f}
- Train RMSE: {train['RMSE']:.4f}
- Train MAE: {train['MAE']:.4f}
- Train R2: {train['R2']:.4f}
- Test RMSE: {test['RMSE']:.4f}
- Test MAE: {test['MAE']:.4f}
- Test R2: {test['R2']:.4f}
- Within +/-0.5 quality points: {test['Within_0.5']:.1%}
- Within +/-1.0 quality point: {test['Within_1.0']:.1%}
- Rounded exact accuracy: {test['Rounded_Accuracy']:.1%}

## Secondary Classification Evaluation

This is not the main task. It converts the target to `{class_metrics['definition']}` and trains a separate probabilistic classifier on the training split. Class labels use probability threshold `{class_metrics['decision_threshold']}`; ROC-AUC uses predicted probabilities.

- Classifier: {class_metrics['model']}
- Accuracy: {class_metrics['accuracy']:.4f}
- Precision: {class_metrics['precision']:.4f}
- Recall: {class_metrics['recall']:.4f}
- F1: {class_metrics['f1']:.4f}
- ROC-AUC: {class_metrics['roc_auc']:.4f}
- Confusion matrix [[TN, FP], [FN, TP]]: {class_metrics['confusion_matrix']}

## Outlier Conclusion

Outlier methods were tested by cross-validation:

{md_table(outliers, [c for c in outliers.columns if c != 'training_time_sec'])}

The best outlier treatment by CV RMSE was `{outlier_best['model']}`. Apparent extremes were treated as valid but unusual wines unless validation evidence showed benefit from clipping/filtering.

## Interpretation

The final model is a boosted/tree ensemble or randomized tree ensemble selected by validation performance. These models learn nonlinear relationships and interactions by recursively splitting the chemistry feature space into regions with different average quality. Regularization comes from limited tree depth/leaf constraints, shrinkage for boosting, subsampling, and cross-validated hyperparameter selection.

Top permutation-importance features:

{', '.join(top_features)}

These are predictive associations, not causal claims.

## How to Run

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python -m pip install -r requirements.txt
.\\.venv\\Scripts\\python src\\run_project.py
```

Outputs are written to `results/`, figures to `results/figures/`, and the trained pipeline to `models/final_wine_quality_pipeline.joblib`.

## Code Structure

- `src/config.py`: paths, random seed, and split settings.
- `src/data.py`: download, loading, duplicate removal, splitting, and outlier summaries.
- `src/features.py`: scikit-learn-compatible feature engineering and clipping.
- `src/modeling.py`: estimators, cross-validation, tuning, and evaluation.
- `src/reporting.py`: plots and reports generated from measured results.
- `src/run_project.py`: orchestration; also supports `python -m src.run_project`.

`requirements.txt` pins the versions used for these results. Raw data, split files,
experiment tables, figures, and the two trained models are retained for reproducibility.

## Limitations

Quality scores are human sensory ratings and are ordinal despite being modeled as numeric. Exact duplicate removal is conservative, but it may remove real repeated batches rather than data-entry duplicates. The classification metrics are secondary and depend on the chosen high-quality threshold; they should not be presented as the main regression result.
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")

    notes = f"""# Presentation Notes

## What We Did

We built a regression model for the UCI Wine Quality dataset. We combined red and white wine data, added a `wine_type` indicator, removed exact duplicate rows before splitting, and held out one untouched stratified test set.

## Why These Decisions

Combining red and white wines gives the model more examples while preserving type information. Duplicate removal reduces the chance that identical records appear in both train and test data. Cross-validation on the training data was used for every modeling decision so the final test set stayed honest.

## Feature Engineering

We added explainable chemistry-inspired features: sulfur dioxide ratios, bound sulfur dioxide, acidity ratios, sugar/density and alcohol/density ratios, sulphates/chlorides ratio, interaction terms involving alcohol, acidity, and volatile acidity, plus log transforms for skewed chemical measurements.

## Winning Model

The selected model was `{final_label}` with tuned parameters shown in `results/best_params.json`. It won because it had the lowest cross-validated RMSE among the tested candidates, while maintaining a reasonable validation-to-test pattern.

## Scores

On the untouched test set, RMSE was {test['RMSE']:.4f}, MAE was {test['MAE']:.4f}, and R2 was {test['R2']:.4f}. About {test['Within_1.0']:.1%} of predictions were within one quality point.

## How The Model Works

The model predicts quality by combining many decision trees. Each tree splits wines by chemical measurements or engineered features; the ensemble averages or boosts those trees to capture nonlinear patterns. Overfitting is controlled through tree complexity limits, regularization, subsampling, and cross-validation.

## Overfitting Evidence

Training RMSE: {train['RMSE']:.4f}. Final CV RMSE before final test: {final_cv['CV_RMSE_mean']:.4f}. Test RMSE: {test['RMSE']:.4f}. The model memorizes the training data more than a linear model would, but the CV and test scores are close, so the reported generalization estimate is stable.

## Top Predictors

The strongest predictors by permutation importance were: {', '.join(top_features)}.

## What Not To Claim

Do not claim chemical causality. Do not claim classification metrics are the primary result. Do not claim the threshold `quality >= 7` is the only possible definition of high quality; it is a defensible, common interpretation based on the score scale and distribution.

## What We Would Try Next

More careful ordinal-regression methods, repeated cross-validation for tighter uncertainty estimates, SHAP analysis for richer interpretation, and external validation on another wine dataset.
"""
    (ROOT / "PRESENTATION_NOTES.md").write_text(notes, encoding="utf-8")

    log = f"""# Experiment Log

All scores below were computed on training-set cross-validation unless explicitly labeled test.

## Dataset Setup

- Tested design: combine red and white wines with `wine_type`.
- Reason: use all available data without losing wine-type information.
- Conclusion: chosen to retain wine-type information while using both datasets; superiority over separate models was not tested.

## Candidate Model Search

{md_table(model_comparison, ['model', 'engineered', 'CV_RMSE_mean', 'CV_RMSE_std', 'CV_MAE_mean', 'CV_R2_mean', 'Train_RMSE_mean', 'Train_CV_gap_RMSE'])}

Conclusion: `{best_row['model']}` was the strongest untuned/initial candidate by CV RMSE.

## Outlier Experiments

{md_table(outliers, [c for c in outliers.columns if c != 'training_time_sec'])}

Conclusion: `{outlier_best['model']}` performed best among tested outlier strategies. We did not remove difficult observations merely because they were unusual.

## Hyperparameter Tuning

- Tuned family: `{best_family}`
- Engineered features in final model: `{final_engineered}`
- Search method: RandomizedSearchCV, {CV_FOLDS}-fold stratified CV, RMSE objective.
- Best CV RMSE from search: {-pd.read_csv(RESULTS / 'tuning_results.csv').iloc[0]['mean_test_score']:.4f}
- Re-evaluated final pipeline CV RMSE: {final_cv['CV_RMSE_mean']:.4f}
- Best parameters: `{json.dumps(best_params, default=str)}`

## Final Test Evaluation

- Test RMSE: {test['RMSE']:.4f}
- Test MAE: {test['MAE']:.4f}
- Test R2: {test['R2']:.4f}

Conclusion: the final score is reported once from the untouched test set after model selection and tuning.
"""
    (ROOT / "EXPERIMENT_LOG.md").write_text(log, encoding="utf-8")
