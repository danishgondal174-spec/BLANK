"""
test_feature_engineer.py
------------------------
Unit tests for pipelines/feature_engineer.py.

Run with:  pytest tests/test_feature_engineer.py -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

# Allow importing from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipelines.feature_engineer import (
    add_lag_features,
    add_return_features,
    add_rolling_features,
    add_rsi,
    add_ratio_features,
    add_rolling_correlations,
    add_target,
    build_features,
    compute_rsi,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Small synthetic daily price DataFrame (100 rows)."""
    np.random.seed(0)
    idx = pd.bdate_range("2020-01-01", periods=100)
    return pd.DataFrame(
        {
            "gold_Close":   1500 + np.cumsum(np.random.randn(100)),
            "sp500_Close":  3000 + np.cumsum(np.random.randn(100) * 5),
            "dxy_Close":    100  + np.cumsum(np.random.randn(100) * 0.1),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# add_lag_features
# ---------------------------------------------------------------------------

class TestAddLagFeatures:
    def test_columns_created(self, sample_df):
        out = add_lag_features(sample_df, columns=["gold_Close"], lags=[1, 5])
        assert "gold_Close_lag_1" in out.columns
        assert "gold_Close_lag_5" in out.columns

    def test_lag_values_correct(self, sample_df):
        out = add_lag_features(sample_df, columns=["gold_Close"], lags=[1])
        # Row at index 1 should equal original row 0
        assert out["gold_Close_lag_1"].iloc[1] == pytest.approx(
            sample_df["gold_Close"].iloc[0]
        )

    def test_first_row_is_nan(self, sample_df):
        out = add_lag_features(sample_df, columns=["gold_Close"], lags=[1])
        assert pd.isna(out["gold_Close_lag_1"].iloc[0])

    def test_missing_column_raises(self, sample_df):
        with pytest.raises(KeyError):
            add_lag_features(sample_df, columns=["nonexistent"])

    def test_input_not_modified(self, sample_df):
        original_cols = list(sample_df.columns)
        add_lag_features(sample_df, columns=["gold_Close"])
        assert list(sample_df.columns) == original_cols


# ---------------------------------------------------------------------------
# add_return_features
# ---------------------------------------------------------------------------

class TestAddReturnFeatures:
    def test_columns_created(self, sample_df):
        out = add_return_features(sample_df, columns=["gold_Close"], periods=[1, 5])
        assert "gold_Close_return_1d" in out.columns
        assert "gold_Close_return_5d" in out.columns

    def test_return_value_correct(self, sample_df):
        out = add_return_features(sample_df, columns=["gold_Close"], periods=[1])
        expected = (
            sample_df["gold_Close"].iloc[1] / sample_df["gold_Close"].iloc[0] - 1
        )
        assert out["gold_Close_return_1d"].iloc[1] == pytest.approx(expected)

    def test_missing_column_raises(self, sample_df):
        with pytest.raises(KeyError):
            add_return_features(sample_df, columns=["bad_col"])


# ---------------------------------------------------------------------------
# add_rolling_features
# ---------------------------------------------------------------------------

class TestAddRollingFeatures:
    def test_columns_created(self, sample_df):
        out = add_rolling_features(
            sample_df, columns=["gold_Close"], windows=[7], stats=["mean", "std"]
        )
        assert "gold_Close_rolling7_mean" in out.columns
        assert "gold_Close_rolling7_std" in out.columns

    def test_rolling_mean_value(self, sample_df):
        window = 7
        out = add_rolling_features(
            sample_df, columns=["gold_Close"], windows=[window], stats=["mean"]
        )
        expected = sample_df["gold_Close"].iloc[:window].mean()
        assert out["gold_Close_rolling7_mean"].iloc[window - 1] == pytest.approx(expected)

    def test_missing_column_raises(self, sample_df):
        with pytest.raises(KeyError):
            add_rolling_features(sample_df, columns=["no_such_col"])


# ---------------------------------------------------------------------------
# compute_rsi / add_rsi
# ---------------------------------------------------------------------------

class TestRSI:
    def test_rsi_range(self, sample_df):
        rsi = compute_rsi(sample_df["gold_Close"])
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_add_rsi_column_created(self, sample_df):
        out = add_rsi(sample_df, columns=["gold_Close"])
        assert "gold_Close_rsi14" in out.columns

    def test_add_rsi_missing_column_raises(self, sample_df):
        with pytest.raises(KeyError):
            add_rsi(sample_df, columns=["bad"])


# ---------------------------------------------------------------------------
# add_ratio_features
# ---------------------------------------------------------------------------

class TestAddRatioFeatures:
    def test_ratio_column_created(self, sample_df):
        out = add_ratio_features(sample_df, ratios=[("gold_Close", "sp500_Close")])
        assert "gold_Close_to_sp500_Close_ratio" in out.columns

    def test_ratio_value_correct(self, sample_df):
        out = add_ratio_features(sample_df, ratios=[("gold_Close", "sp500_Close")])
        expected = (
            sample_df["gold_Close"].iloc[0] / sample_df["sp500_Close"].iloc[0]
        )
        assert out["gold_Close_to_sp500_Close_ratio"].iloc[0] == pytest.approx(expected)

    def test_missing_numerator_raises(self, sample_df):
        with pytest.raises(KeyError):
            add_ratio_features(sample_df, ratios=[("bad_col", "sp500_Close")])

    def test_missing_denominator_raises(self, sample_df):
        with pytest.raises(KeyError):
            add_ratio_features(sample_df, ratios=[("gold_Close", "bad_col")])


# ---------------------------------------------------------------------------
# add_rolling_correlations
# ---------------------------------------------------------------------------

class TestAddRollingCorrelations:
    def test_correlation_column_created(self, sample_df):
        df = add_return_features(sample_df, columns=["gold_Close", "dxy_Close"])
        out = add_rolling_correlations(
            df,
            pairs=[("gold_Close_return_1d", "dxy_Close_return_1d")],
            windows=[30],
        )
        assert "gold_Close_return_1d_x_dxy_Close_return_1d_corr30" in out.columns

    def test_correlation_range(self, sample_df):
        df = add_return_features(sample_df, columns=["gold_Close", "dxy_Close"])
        out = add_rolling_correlations(
            df,
            pairs=[("gold_Close_return_1d", "dxy_Close_return_1d")],
            windows=[30],
        )
        valid = out["gold_Close_return_1d_x_dxy_Close_return_1d_corr30"].dropna()
        assert (valid >= -1.0 - 1e-9).all() and (valid <= 1.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# add_target
# ---------------------------------------------------------------------------

class TestAddTarget:
    def test_regression_target_is_future_price(self, sample_df):
        out = add_target(sample_df, price_col="gold_Close", forward_period=1, task="regression")
        assert out["target"].iloc[0] == pytest.approx(sample_df["gold_Close"].iloc[1])

    def test_classification_target_binary(self, sample_df):
        out = add_target(sample_df, price_col="gold_Close", task="classification")
        valid = out["target"].dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})

    def test_last_rows_are_nan(self, sample_df):
        out = add_target(sample_df, price_col="gold_Close", forward_period=1)
        assert pd.isna(out["target"].iloc[-1])

    def test_invalid_task_raises(self, sample_df):
        with pytest.raises(ValueError):
            add_target(sample_df, task="invalid")


# ---------------------------------------------------------------------------
# build_features (integration)
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def test_output_has_target(self, sample_df):
        out = build_features(sample_df)
        assert "target" in out.columns

    def test_no_nans_after_build(self, sample_df):
        out = build_features(sample_df)
        assert not out.isnull().any().any()

    def test_row_count_reduced_by_lags(self, sample_df):
        # The longest lag/window is 30 rows, so we expect fewer rows than input
        out = build_features(sample_df)
        assert len(out) < len(sample_df)

    def test_invalid_columns_raises(self, sample_df):
        with pytest.raises(ValueError):
            build_features(sample_df, price_cols=["nonexistent"])

    def test_regression_target_is_float(self, sample_df):
        out = build_features(sample_df, task="regression")
        assert out["target"].dtype == float

    def test_classification_target_binary(self, sample_df):
        out = build_features(sample_df, task="classification")
        assert set(out["target"].unique()).issubset({0.0, 1.0})
