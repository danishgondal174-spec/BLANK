"""
Unit tests for the gold price ML pipeline modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(n: int = 300) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame that looks like Yahoo Finance output."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-01-02", periods=n, freq="B")
    base = 1300.0
    prices = base + np.cumsum(rng.normal(0, 5, n))
    sp500 = 2000.0 + np.cumsum(rng.normal(0, 10, n))
    nasdaq = 5000.0 + np.cumsum(rng.normal(0, 20, n))
    dxy = 100.0 + np.cumsum(rng.normal(0, 0.5, n))

    df = pd.DataFrame(
        {
            "gold_Open": prices * 0.999,
            "gold_High": prices * 1.005,
            "gold_Low": prices * 0.995,
            "gold_Close": prices,
            "gold_Volume": rng.integers(10_000, 100_000, n).astype(float),
            "sp500_Close": sp500,
            "nasdaq_Close": nasdaq,
            "dxy_Close": dxy,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

class TestFeatures:
    def test_build_features_regression(self):
        from src.features import build_features

        raw = _make_price_df(300)
        df = build_features(raw, task="regression", horizon=1)
        assert "target" in df.columns
        assert len(df) > 0
        assert not df.isnull().any().any(), "NaN values remain after build_features"

    def test_build_features_classification(self):
        from src.features import build_features

        raw = _make_price_df(300)
        df = build_features(raw, task="classification", horizon=1)
        assert set(df["target"].unique()).issubset({0, 1})

    def test_build_features_invalid_task(self):
        from src.features import build_features

        raw = _make_price_df(300)
        with pytest.raises(ValueError, match="Unknown task"):
            build_features(raw, task="unknown")

    def test_add_rsi_range(self):
        from src.features import add_rsi

        df = _make_price_df(100).copy()
        df = add_rsi(df, "gold_Close", window=14)
        col = "gold_Close_rsi14"
        assert col in df.columns
        valid = df[col].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_add_macd_columns(self):
        from src.features import add_macd

        df = _make_price_df(100).copy()
        df = add_macd(df, "gold_Close")
        for col in ["gold_Close_macd", "gold_Close_macd_signal", "gold_Close_macd_hist"]:
            assert col in df.columns

    def test_add_bollinger_bands(self):
        from src.features import add_bollinger_bands

        df = _make_price_df(100).copy()
        df = add_bollinger_bands(df, "gold_Close", window=20)
        assert "gold_Close_bb_upper20" in df.columns
        assert "gold_Close_bb_lower20" in df.columns

    def test_lag_features(self):
        from src.features import add_lag_features

        df = _make_price_df(50).copy()
        df = add_lag_features(df, "gold_Close", lags=[1, 5])
        assert "gold_Close_lag1" in df.columns
        assert "gold_Close_lag5" in df.columns
        # Verify at index 5 (after enough rows to skip the initial NaN lag values)
        check_row = 5
        assert df["gold_Close_lag1"].iloc[check_row] == pytest.approx(
            df["gold_Close"].iloc[check_row - 1]
        )

    def test_no_lookahead_in_target(self):
        from src.features import build_features

        raw = _make_price_df(300)
        df = build_features(raw, task="regression", horizon=1)
        # The last row of the original data cannot have a target (it's shifted)
        # build_features drops NaN rows so the last original date should be gone
        original_last_date = _make_price_df(300).index[-1]
        assert original_last_date not in df.index


# ---------------------------------------------------------------------------
# Time-series cross-validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def _get_data(self, task: str = "regression"):
        from src.features import build_features

        raw = _make_price_df(400)
        df = build_features(raw, task=task, horizon=1)
        feat_cols = [c for c in df.columns if c != "target"]
        return df[feat_cols], df["target"]

    def test_time_series_cv_regression(self):
        from src.validation import time_series_cv

        X, y = self._get_data("regression")
        model = LinearRegression()
        result = time_series_cv(model, X, y, n_splits=3, task="regression")
        assert len(result.fold_metrics) == 3
        assert "MAE" in result.mean_metrics
        assert result.mean_metrics["MAE"] > 0

    def test_time_series_cv_classification(self):
        from src.validation import time_series_cv

        X, y = self._get_data("classification")
        model = LogisticRegression(max_iter=500)
        result = time_series_cv(model, X, y, n_splits=3, task="classification")
        assert len(result.fold_metrics) == 3
        assert "Accuracy" in result.mean_metrics
        assert 0.0 <= result.mean_metrics["Accuracy"] <= 1.0

    def test_cv_result_summary(self):
        from src.validation import time_series_cv

        X, y = self._get_data("regression")
        result = time_series_cv(LinearRegression(), X, y, n_splits=3, task="regression")
        summary = result.summary()
        assert "MAE" in summary.index
        assert "mean" in summary.columns
        assert "std" in summary.columns

    def test_compare_models_cv(self):
        from src.validation import compare_models_cv

        X, y = self._get_data("regression")
        models = {
            "LR": LinearRegression(),
            "RF": RandomForestRegressor(n_estimators=10, random_state=42),
        }
        df = compare_models_cv(models, X, y, n_splits=3, task="regression")
        assert set(df.index) == {"LR", "RF"}
        assert "MAE_mean" in df.columns


# ---------------------------------------------------------------------------
# Hyperparameter tuning tests
# ---------------------------------------------------------------------------

class TestTuning:
    def test_randomized_search_regression(self):
        from src.tuning import randomized_search

        from src.features import build_features

        raw = _make_price_df(300)
        df = build_features(raw, task="regression", horizon=1)
        feat_cols = [c for c in df.columns if c != "target"]
        X, y = df[feat_cols], df["target"]

        model = RandomForestRegressor(random_state=42)
        small_grid = {
            "n_estimators": [10, 20],
            "max_depth": [3, 5],
        }
        search = randomized_search(
            model, X, y, param_grid=small_grid, n_iter=3, n_splits=3, task="regression"
        )
        assert hasattr(search, "best_estimator_")
        assert hasattr(search, "best_params_")

    def test_randomized_search_classification(self):
        from src.tuning import randomized_search

        from src.features import build_features

        raw = _make_price_df(300)
        df = build_features(raw, task="classification", horizon=1)
        feat_cols = [c for c in df.columns if c != "target"]
        X, y = df[feat_cols], df["target"]

        model = RandomForestClassifier(random_state=42)
        small_grid = {
            "n_estimators": [10, 20],
            "max_depth": [3, 5],
        }
        search = randomized_search(
            model, X, y, param_grid=small_grid, n_iter=3, n_splits=3, task="classification"
        )
        assert hasattr(search, "best_estimator_")

    def test_best_params_report(self):
        from src.tuning import best_params_report, randomized_search

        from src.features import build_features

        raw = _make_price_df(300)
        df = build_features(raw, task="regression", horizon=1)
        feat_cols = [c for c in df.columns if c != "target"]
        X, y = df[feat_cols], df["target"]

        model = RandomForestRegressor(random_state=42)
        small_grid = {"n_estimators": [10, 20], "max_depth": [3]}
        search = randomized_search(
            model, X, y, param_grid=small_grid, n_iter=2, n_splits=3, task="regression"
        )
        report = best_params_report(search)
        assert "mean_test_score" in report.columns


# ---------------------------------------------------------------------------
# Model catalogue tests
# ---------------------------------------------------------------------------

class TestModels:
    def _split_data(self, task: str = "regression"):
        from src.features import build_features

        raw = _make_price_df(400)
        df = build_features(raw, task=task, horizon=1)
        feat_cols = [c for c in df.columns if c != "target"]
        split = int(len(df) * 0.8)
        train, test = df.iloc[:split], df.iloc[split:]
        return train[feat_cols], train["target"], test[feat_cols], test["target"], feat_cols

    def test_compare_on_test_regression(self):
        from src.models import compare_on_test, get_regression_models

        X_train, y_train, X_test, y_test, _ = self._split_data("regression")
        # Use only fast models for the test
        models = {
            k: v
            for k, v in get_regression_models().items()
            if k in ("LinearRegression", "Ridge")
        }
        result = compare_on_test(models, X_train, y_train, X_test, y_test, task="regression")
        assert "MAE" in result.columns
        assert "RMSE" in result.columns

    def test_compare_on_test_classification(self):
        from src.models import compare_on_test, get_classification_models

        X_train, y_train, X_test, y_test, _ = self._split_data("classification")
        models = {
            k: v
            for k, v in get_classification_models().items()
            if k == "LogisticRegression"
        }
        result = compare_on_test(
            models, X_train, y_train, X_test, y_test, task="classification"
        )
        assert "Accuracy" in result.columns

    def test_feature_importance(self):
        from src.models import feature_importance_df

        X_train, y_train, _, _, feat_cols = self._split_data("regression")
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        fi = feature_importance_df(model, feat_cols)
        assert "feature" in fi.columns
        assert "importance" in fi.columns
        assert len(fi) == len(feat_cols)
        assert fi["importance"].sum() == pytest.approx(1.0, abs=1e-6)
