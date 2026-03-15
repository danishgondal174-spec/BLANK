"""
Time-series cross-validation utilities for gold price ML.

Wraps scikit-learn's TimeSeriesSplit with helpers for reporting metrics
and comparing multiple models with proper temporal splits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class CVResult:
    """Container for per-fold and aggregate cross-validation results."""

    model_name: str
    task: str  # 'regression' or 'classification'
    fold_metrics: list[dict] = field(default_factory=list)

    @property
    def mean_metrics(self) -> dict:
        """Return the mean of each metric across folds."""
        if not self.fold_metrics:
            return {}
        keys = self.fold_metrics[0].keys()
        return {k: float(np.mean([f[k] for f in self.fold_metrics])) for k in keys}

    @property
    def std_metrics(self) -> dict:
        """Return the standard deviation of each metric across folds."""
        if not self.fold_metrics:
            return {}
        keys = self.fold_metrics[0].keys()
        return {k: float(np.std([f[k] for f in self.fold_metrics])) for k in keys}

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame with mean ± std for each metric."""
        means = self.mean_metrics
        stds = self.std_metrics
        rows = []
        for k in means:
            rows.append({"metric": k, "mean": means[k], "std": stds[k]})
        return pd.DataFrame(rows).set_index("metric")


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def time_series_cv(
    model,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    n_splits: int = 5,
    task: str = "regression",
    model_name: str = "model",
    gap: int = 0,
) -> CVResult:
    """
    Evaluate *model* using expanding-window time-series cross-validation.

    Parameters
    ----------
    model      : scikit-learn estimator (fit/predict interface)
    X          : feature matrix (rows must be time-ordered)
    y          : target vector
    n_splits   : number of CV folds
    task       : 'regression' or 'classification'
    model_name : label for reporting
    gap        : number of samples to skip between train and test sets
                 (use to avoid leakage when predicting *horizon* days ahead)

    Returns
    -------
    CVResult
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    X_arr = np.array(X)
    y_arr = np.array(y)

    result = CVResult(model_name=model_name, task=task)
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_arr)):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task == "regression":
            metrics = _regression_metrics(y_test, y_pred)
        else:
            metrics = _classification_metrics(y_test, y_pred)

        metrics["fold"] = fold_idx + 1
        result.fold_metrics.append(metrics)

    return result


def compare_models_cv(
    models: dict,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    n_splits: int = 5,
    task: str = "regression",
    gap: int = 0,
) -> pd.DataFrame:
    """
    Run time-series CV for every model in *models* and return a comparison
    DataFrame.

    Parameters
    ----------
    models : dict mapping model name → estimator instance

    Returns
    -------
    DataFrame with one row per model, mean and std of each metric.
    """
    rows = []
    for name, model in models.items():
        print(f"  Evaluating {name} ...")
        result = time_series_cv(
            model, X, y, n_splits=n_splits, task=task, model_name=name, gap=gap
        )
        row = {"model": name}
        for k, v in result.mean_metrics.items():
            row[f"{k}_mean"] = v
        for k, v in result.std_metrics.items():
            row[f"{k}_std"] = v
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    return df
