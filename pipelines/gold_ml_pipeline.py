"""
gold_ml_pipeline.py
-------------------
End-to-end, modular pipeline for gold price prediction.

Run directly for a full train → evaluate → save cycle:

    python pipelines/gold_ml_pipeline.py

Or import individual functions to use in notebooks or scripts.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def run_pipeline(
    yahoo_path: str | None = None,
    fred_path:  str | None = None,
    task:       str = "regression",
    model_out:  str = "models/gold_price_model.pkl",
    train_end:  str = "2021-12-31",
    n_splits:   int = 5,
) -> dict:
    """
    Execute the full gold price ML pipeline.

    Steps
    -----
    1. Load Yahoo Finance + FRED data.
    2. Engineer features (lags, rolling stats, RSI, ratios, correlations).
    3. Split into chronological train / test sets.
    4. Cross-validate and compare candidate models.
    5. Tune the best model with randomised hyperparameter search.
    6. Evaluate on the held-out test set.
    7. Save the final model to *model_out*.

    Parameters
    ----------
    yahoo_path : Path to Yahoo market CSV (default: data/yahoo_market_data.csv).
    fred_path  : Path to FRED macro CSV (default: data/fred_macro_data.csv).
    task       : ``'regression'`` or ``'classification'``.
    model_out  : Output path for the serialised model.
    train_end  : Last date (inclusive) of the training period (ISO format).
    n_splits   : Number of time-series CV folds.

    Returns
    -------
    Dictionary containing ``test_metrics``, ``best_params``, and ``feature_names``.
    """
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    from pipelines.data_loader import load_all_data
    df = load_all_data(yahoo_path, fred_path)

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    from pipelines.feature_engineer import build_features
    df_feat = build_features(df, task=task)

    feature_cols = [c for c in df_feat.columns if c != "target"]
    X = df_feat[feature_cols]
    y = df_feat["target"]

    # ------------------------------------------------------------------
    # 3. Train / test split (chronological — no shuffle)
    # ------------------------------------------------------------------
    train_mask = df_feat.index <= train_end
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    print(f"Train: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows")

    # ------------------------------------------------------------------
    # 4. Model comparison
    # ------------------------------------------------------------------
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor

    try:
        from xgboost import XGBRegressor as _XGB
        xgb_model = _XGB(n_estimators=200, random_state=42, verbosity=0)
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor as _GB
        xgb_model = _GB(n_estimators=200, random_state=42)

    candidates = {
        "Ridge":        Ridge(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "GradientBoost": xgb_model,
    }

    from pipelines.trainer import compare_models
    print("\n--- Model comparison (time-series CV) ---")
    comparison = compare_models(candidates, X_train, y_train, n_splits=n_splits)
    print(comparison)

    best_name = comparison.index[0]
    print(f"\nBest model: {best_name}")

    # ------------------------------------------------------------------
    # 5. Hyperparameter tuning for the best model
    # ------------------------------------------------------------------
    from pipelines.trainer import tune_model

    if best_name == "RandomForest":
        param_dist = {
            "n_estimators": [100, 200, 400],
            "max_depth":    [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        }
        base_estimator = RandomForestRegressor(random_state=42, n_jobs=-1)
    elif best_name == "GradientBoost":
        param_dist = {
            "n_estimators":     [100, 300, 500],
            "max_depth":        [3, 5, 7],
            "learning_rate":    [0.01, 0.05, 0.1],
            "subsample":        [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
        }
        base_estimator = xgb_model.__class__(random_state=42, verbosity=0)
    else:
        param_dist = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
        base_estimator = Ridge()

    print("\n--- Hyperparameter tuning ---")
    search = tune_model(
        base_estimator,
        param_dist,
        X_train,
        y_train,
        n_iter=20,
        n_splits=n_splits,
    )
    best_model  = search.best_estimator_
    best_params = search.best_params_

    # ------------------------------------------------------------------
    # 6. Test-set evaluation
    # ------------------------------------------------------------------
    from pipelines.trainer import evaluate_model
    print("\n--- Test-set evaluation ---")
    test_metrics = evaluate_model(best_model, X_test, y_test, task=task)

    # ------------------------------------------------------------------
    # 7. Save model
    # ------------------------------------------------------------------
    from pipelines.trainer import save_model
    save_model(best_model, model_out)

    return {
        "test_metrics":   test_metrics,
        "best_params":    best_params,
        "feature_names":  feature_cols,
        "model_comparison": comparison,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end gold price ML pipeline."
    )
    parser.add_argument("--yahoo-path", default=None, help="Path to Yahoo market CSV.")
    parser.add_argument("--fred-path",  default=None, help="Path to FRED macro CSV.")
    parser.add_argument(
        "--task",
        default="regression",
        choices=["regression", "classification"],
        help="Prediction task type.",
    )
    parser.add_argument(
        "--model-out",
        default="models/gold_price_model.pkl",
        help="Output path for serialised model.",
    )
    parser.add_argument(
        "--train-end",
        default="2021-12-31",
        help="Last date of training period (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--n-splits", type=int, default=5, help="Number of time-series CV folds."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = run_pipeline(
        yahoo_path=args.yahoo_path,
        fred_path=args.fred_path,
        task=args.task,
        model_out=args.model_out,
        train_end=args.train_end,
        n_splits=args.n_splits,
    )
    print("\nPipeline complete.")
    print("Test metrics:", results["test_metrics"])
