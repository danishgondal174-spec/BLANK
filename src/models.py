"""
Model definitions and comparison utilities for gold price ML.

Provides a catalogue of models for both regression and classification,
plus helpers for feature selection and model evaluation on a held-out
test set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor

    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

def get_regression_models() -> dict:
    """Return a dict of name → estimator for regression tasks."""
    models = {
        "LinearRegression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "Ridge": Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]
        ),
        "Lasso": Pipeline(
            [("scaler", StandardScaler()), ("model", Lasso(alpha=0.01))]
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
        ),
    }
    if _XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    return models


def get_classification_models() -> dict:
    """Return a dict of name → estimator for classification tasks."""
    models = {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
        ),
    }
    if _XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    return models


# ---------------------------------------------------------------------------
# Held-out test evaluation
# ---------------------------------------------------------------------------

def evaluate_regression(
    model,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    model_name: str = "model",
) -> dict:
    """Train *model* and return test-set regression metrics."""
    model.fit(np.array(X_train), np.array(y_train))
    y_pred = model.predict(np.array(X_test))
    mse = mean_squared_error(y_test, y_pred)
    return {
        "model": model_name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_test, y_pred),
    }


def evaluate_classification(
    model,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    model_name: str = "model",
) -> dict:
    """Train *model* and return test-set classification metrics."""
    model.fit(np.array(X_train), np.array(y_train))
    y_pred = model.predict(np.array(X_test))
    return {
        "model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred, zero_division=0),
    }


def compare_on_test(
    models: dict,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    task: str = "regression",
) -> pd.DataFrame:
    """
    Fit and evaluate every model in *models* on a held-out test set.

    Returns a DataFrame sorted by the primary metric (RMSE or Accuracy).
    """
    rows = []
    for name, model in models.items():
        print(f"  Training {name} ...")
        if task == "regression":
            row = evaluate_regression(model, X_train, y_train, X_test, y_test, name)
        else:
            row = evaluate_classification(model, X_train, y_train, X_test, y_test, name)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    sort_col = "RMSE" if task == "regression" else "Accuracy"
    ascending = task == "regression"
    return df.sort_values(sort_col, ascending=ascending)


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def feature_importance_df(model, feature_names: list[str]) -> pd.DataFrame:
    """
    Extract feature importances from a tree-based model or a Pipeline
    containing one.

    Returns a DataFrame sorted by importance (descending).
    """
    # Unwrap Pipeline if needed
    estimator = model
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "feature_importances_"):
                estimator = step
                break
            if hasattr(step, "coef_"):
                estimator = step
                break

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_).flatten()
    else:
        raise ValueError(f"Cannot extract feature importances from {type(estimator).__name__}")

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def select_top_features(
    model,
    feature_names: list[str],
    top_n: int = 20,
) -> list[str]:
    """Return the names of the top *top_n* features by importance."""
    df = feature_importance_df(model, feature_names)
    return df["feature"].head(top_n).tolist()
