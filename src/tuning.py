"""
Hyperparameter tuning for gold price ML models.

Wraps scikit-learn's RandomizedSearchCV and GridSearchCV with
TimeSeriesSplit so that parameter search respects temporal order.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit


# ---------------------------------------------------------------------------
# Default parameter grids
# ---------------------------------------------------------------------------

PARAM_GRIDS: dict[str, dict[str, list]] = {
    "RandomForestRegressor": {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2", 0.5],
    },
    "RandomForestClassifier": {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2", 0.5],
    },
    "GradientBoostingRegressor": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 1.0],
    },
    "GradientBoostingClassifier": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 1.0],
    },
    "XGBRegressor": {
        "n_estimators": [100, 200, 400],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0],
    },
    "XGBClassifier": {
        "n_estimators": [100, 200, 400],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
    "Ridge": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "Lasso": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
    },
}


# ---------------------------------------------------------------------------
# Scoring shortcuts
# ---------------------------------------------------------------------------

SCORING = {
    "regression": "neg_mean_absolute_error",
    "classification": "f1",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def randomized_search(
    model,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    param_grid: dict[str, list] | None = None,
    n_iter: int = 30,
    n_splits: int = 5,
    task: str = "regression",
    random_state: int = 42,
    n_jobs: int = -1,
    gap: int = 0,
) -> RandomizedSearchCV:
    """
    Tune *model* with randomized search and time-series CV.

    Parameters
    ----------
    model      : unfitted scikit-learn estimator
    X          : feature matrix (time-ordered)
    y          : target
    param_grid : parameter distributions; if None, uses the built-in grid
    n_iter     : number of random combinations to try
    n_splits   : number of time-series CV folds
    task       : 'regression' or 'classification'
    random_state : seed for reproducibility
    n_jobs     : parallel workers (-1 = all)
    gap        : samples to skip between train and test folds

    Returns
    -------
    Fitted RandomizedSearchCV object
    """
    if param_grid is None:
        class_name = type(model).__name__
        param_grid = PARAM_GRIDS.get(class_name, {})
        if not param_grid:
            raise ValueError(
                f"No default param_grid for '{class_name}'. Pass one explicitly."
            )

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scoring = SCORING.get(task, "neg_mean_absolute_error")

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=tscv,
        n_jobs=n_jobs,
        random_state=random_state,
        refit=True,
        verbose=1,
    )
    search.fit(np.array(X), np.array(y))
    return search


def grid_search(
    model,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    param_grid: dict[str, list],
    n_splits: int = 5,
    task: str = "regression",
    n_jobs: int = -1,
    gap: int = 0,
) -> GridSearchCV:
    """
    Exhaustive grid search with time-series CV.

    Parameters
    ----------
    param_grid : explicit grid to search
    (other parameters same as randomized_search)

    Returns
    -------
    Fitted GridSearchCV object
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scoring = SCORING.get(task, "neg_mean_absolute_error")

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=tscv,
        n_jobs=n_jobs,
        refit=True,
        verbose=1,
    )
    search.fit(np.array(X), np.array(y))
    return search


def best_params_report(search) -> pd.DataFrame:
    """
    Summarise top-10 parameter combinations from a fitted search object.

    Returns a DataFrame sorted by mean test score (best first).
    """
    results = pd.DataFrame(search.cv_results_)
    cols = [c for c in results.columns if c.startswith("param_")]
    cols += ["mean_test_score", "std_test_score", "rank_test_score"]
    return (
        results[cols]
        .sort_values("rank_test_score")
        .head(10)
        .reset_index(drop=True)
    )
