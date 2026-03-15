"""
Production-ready end-to-end pipeline for gold price prediction.

Usage
-----
    python -m src.pipeline --task regression --tune
    python -m src.pipeline --task classification

The pipeline:
1. Loads market + FRED macro data from the data/ directory.
2. Builds features (src/features.py).
3. Splits into train/test sets respecting temporal order.
4. (Optional) Runs hyperparameter tuning (src/tuning.py).
5. Compares all models on the held-out test set (src/models.py).
6. Evaluates the winner with time-series CV (src/validation.py).
7. Saves the best model to models/<task>_best_model.joblib.
8. Prints a final performance report.
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd

from src.features import build_features
from src.models import (
    compare_on_test,
    feature_importance_df,
    get_classification_models,
    get_regression_models,
    select_top_features,
)
from src.tuning import PARAM_GRIDS, randomized_search
from src.validation import compare_models_cv, time_series_cv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")
MODELS_DIR = os.path.join(_ROOT, "models")

YAHOO_CSV = os.path.join(DATA_DIR, "yahoo_market_data.csv")
FRED_CSV = os.path.join(DATA_DIR, "fred_macro_data.csv")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load and merge Yahoo Finance and (optionally) FRED macro data."""
    if not os.path.exists(YAHOO_CSV):
        raise FileNotFoundError(
            f"Market data not found at {YAHOO_CSV}. "
            "Run: python scripts/download_market_data.py"
        )

    df = pd.read_csv(YAHOO_CSV, parse_dates=["Date"], index_col="Date")
    df.sort_index(inplace=True)

    if os.path.exists(FRED_CSV):
        fred = pd.read_csv(FRED_CSV, parse_dates=["Date"], index_col="Date")
        fred.sort_index(inplace=True)
        # Align FRED to business-day index and forward-fill gaps
        fred = fred.reindex(df.index, method="ffill")
        df = pd.concat([df, fred], axis=1)
        print(f"Merged FRED macro features: {list(fred.columns)}")
    else:
        print(
            "FRED data not found — running without macro features. "
            "Run: python scripts/download_fred_data.py  (requires FRED_API_KEY)"
        )

    return df


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train/test preserving temporal order."""
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx], df.iloc[split_idx:]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    task: str = "regression",
    horizon: int = 1,
    test_ratio: float = 0.2,
    n_cv_splits: int = 5,
    tune: bool = False,
    n_tune_iter: int = 20,
    top_n_features: int | None = None,
) -> None:
    """
    Execute the full training pipeline.

    Parameters
    ----------
    task          : 'regression' or 'classification'
    horizon       : prediction horizon in trading days
    test_ratio    : fraction of data reserved for the test set
    n_cv_splits   : number of time-series CV folds
    tune          : whether to run hyperparameter tuning on the best model
    n_tune_iter   : iterations for RandomizedSearchCV
    top_n_features: if set, re-train using only the top N features
    """
    print("=" * 60)
    print(f"Gold Price ML Pipeline  |  task={task}  |  horizon={horizon}d")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading data ...")
    raw = load_data()
    print(f"  Raw data shape: {raw.shape}")

    # 2. Feature engineering
    print("\n[2/6] Building features ...")
    df = build_features(raw, task=task, horizon=horizon)
    feature_cols = [c for c in df.columns if c != "target"]
    print(f"  Feature matrix: {df.shape}  ({len(feature_cols)} features)")

    X = df[feature_cols]
    y = df["target"]

    # 3. Train / test split
    print("\n[3/6] Splitting data ...")
    train_df, test_df = temporal_split(df, test_ratio=test_ratio)
    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]
    print(f"  Train: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows")
    print(f"  Train period: {X_train.index[0].date()} → {X_train.index[-1].date()}")
    print(f"  Test  period: {X_test.index[0].date()}  → {X_test.index[-1].date()}")

    # 4. Model comparison (CV)
    print(f"\n[4/6] Comparing models with TimeSeriesSplit (n_splits={n_cv_splits}) ...")
    models = (
        get_regression_models() if task == "regression" else get_classification_models()
    )
    cv_comparison = compare_models_cv(
        models, X_train, y_train, n_splits=n_cv_splits, task=task
    )
    print("\n  Cross-validation results (train set):")
    print(cv_comparison.to_string())

    # Identify best model by CV primary metric
    if task == "regression":
        best_model_name = cv_comparison["MAE_mean"].idxmin()
    else:
        best_model_name = cv_comparison["Accuracy_mean"].idxmax()
    print(f"\n  Best model by CV: {best_model_name}")

    # 5. Hyperparameter tuning (optional)
    best_model = models[best_model_name]
    if tune:
        print(f"\n[5/6] Tuning {best_model_name} ...")
        class_name = type(best_model).__name__
        if class_name in PARAM_GRIDS:
            search = randomized_search(
                best_model,
                X_train,
                y_train,
                n_iter=n_tune_iter,
                n_splits=n_cv_splits,
                task=task,
            )
            best_model = search.best_estimator_
            print(f"  Best params: {search.best_params_}")
        else:
            print(f"  No default param grid for {class_name} — skipping tuning.")
    else:
        print("\n[5/6] Hyperparameter tuning skipped (pass --tune to enable).")
        best_model.fit(X_train, y_train)

    # Optional: re-fit on top-N features
    if top_n_features is not None:
        print(f"\n  Selecting top {top_n_features} features ...")
        try:
            top_feats = select_top_features(best_model, feature_cols, top_n=top_n_features)
            print(f"  Top features: {top_feats}")
            X_train = X_train[top_feats]
            X_test = X_test[top_feats]
            best_model.fit(X_train, y_train)
        except ValueError as exc:
            print(f"  Feature selection skipped: {exc}")

    # 6. Held-out test evaluation
    print("\n[6/6] Test-set evaluation ...")
    test_comparison = compare_on_test(
        {best_model_name: best_model},
        X_train,
        y_train,
        X_test,
        y_test,
        task=task,
    )
    print("\n  Test-set results:")
    print(test_comparison.to_string())

    # 7. Save best model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{task}_best_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"\n  Best model saved to {model_path}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gold price ML pipeline with hyperparameter tuning and CV."
    )
    parser.add_argument(
        "--task",
        choices=["regression", "classification"],
        default="regression",
        help="Prediction task (default: regression)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Prediction horizon in trading days (default: 1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of data for the test set (default: 0.2)",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of TimeSeriesSplit folds (default: 5)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run RandomizedSearchCV for the best model",
    )
    parser.add_argument(
        "--tune-iter",
        type=int,
        default=20,
        help="Number of iterations for RandomizedSearchCV (default: 20)",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=None,
        help="If set, re-train using only the top N features",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        task=args.task,
        horizon=args.horizon,
        test_ratio=args.test_ratio,
        n_cv_splits=args.cv_splits,
        tune=args.tune,
        n_tune_iter=args.tune_iter,
        top_n_features=args.top_features,
    )
