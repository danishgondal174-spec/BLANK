"""
trainer.py
----------
Training, cross-validation, hyperparameter optimisation, and evaluation
helpers for the gold price ML pipeline.
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    TimeSeriesSplit,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------------------------
# Cross-validation helpers
# ---------------------------------------------------------------------------

def timeseries_cv_score(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    scoring: str = "neg_mean_absolute_error",
) -> dict:
    """
    Evaluate *model* using time-series-aware K-fold cross-validation.

    Parameters
    ----------
    model   : Unfitted scikit-learn–compatible estimator.
    X       : Feature matrix (rows ordered chronologically).
    y       : Target series.
    n_splits: Number of CV folds.
    scoring : scikit-learn scoring string.

    Returns
    -------
    Dictionary with keys ``mean``, ``std``, and ``scores``.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring, n_jobs=-1)
    return {
        "mean":   float(-scores.mean()),
        "std":    float(scores.std()),
        "scores": (-scores).tolist(),
    }


def compare_models(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Cross-validate multiple models and return a summary DataFrame.

    Parameters
    ----------
    models  : Mapping of ``{name: unfitted_estimator}``.
    X_train : Training features.
    y_train : Training target.
    n_splits: CV folds.

    Returns
    -------
    DataFrame with columns ``MAE_mean`` and ``MAE_std``, sorted by MAE ascending.
    """
    results = {}
    for name, mdl in models.items():
        cv = timeseries_cv_score(mdl, X_train, y_train, n_splits=n_splits)
        results[name] = {"MAE_mean": cv["mean"], "MAE_std": cv["std"]}
        print(f"  {name:20s}  MAE = {cv['mean']:.4f} ± {cv['std']:.4f}")
    return pd.DataFrame(results).T.sort_values("MAE_mean")


# ---------------------------------------------------------------------------
# Hyperparameter optimisation
# ---------------------------------------------------------------------------

def tune_model(
    estimator,
    param_distributions: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 30,
    n_splits: int = 5,
    scoring: str = "neg_mean_absolute_error",
    random_state: int = 42,
    verbose: int = 1,
) -> RandomizedSearchCV:
    """
    Perform randomised hyperparameter search with time-series CV.

    Returns
    -------
    Fitted ``RandomizedSearchCV`` object.  Access ``search.best_estimator_``
    and ``search.best_params_`` for the tuned model and its parameters.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=tscv,
        scoring=scoring,
        n_jobs=-1,
        random_state=random_state,
        verbose=verbose,
        refit=True,
    )
    search.fit(X_train, y_train)
    best_mae = -search.best_score_
    print(f"Best CV MAE:  {best_mae:.4f}")
    print(f"Best params: {search.best_params_}")
    return search


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str = "regression",
) -> dict:
    """
    Compute evaluation metrics on the held-out test set.

    Parameters
    ----------
    model : Fitted estimator.
    X_test: Test features.
    y_test: True test targets.
    task  : ``'regression'`` or ``'classification'``.

    Returns
    -------
    Dictionary of metric names → values.
    """
    preds = model.predict(X_test)
    metrics: dict = {}

    if task == "regression":
        metrics["MAE"]  = float(mean_absolute_error(y_test, preds))
        metrics["RMSE"] = float(np.sqrt(mean_squared_error(y_test, preds)))
        # Directional accuracy: for each bar, did the model correctly predict
        # whether the price moved up or down relative to the *previous actual*
        # price?  Using the previous *actual* value (not the previous prediction)
        # avoids error accumulation and gives a fair per-bar measure.
        y_vals    = np.asarray(y_test, dtype=float)
        prev_vals = y_vals[:-1]                        # t-1 actual prices
        actual_dir = np.sign(y_vals[1:] - prev_vals)  # actual direction at t
        pred_dir   = np.sign(preds[1:]  - prev_vals)  # predicted direction at t
        metrics["DirectionalAccuracy"] = float(np.mean(actual_dir == pred_dir))
    elif task == "classification":
        from sklearn.metrics import accuracy_score, f1_score
        metrics["Accuracy"] = float(accuracy_score(y_test, preds))
        metrics["F1"]       = float(f1_score(y_test, preds, average="weighted"))

    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model, path: str = "models/gold_price_model.pkl") -> None:
    """Serialise *model* to *path* using joblib."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved → {os.path.abspath(path)}")


def load_model(path: str = "models/gold_price_model.pkl"):
    """Deserialise and return a model from *path*."""
    return joblib.load(path)
