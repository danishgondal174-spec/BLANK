"""
Feature engineering for gold price prediction.

This module provides functions to build a rich feature matrix from raw
OHLCV and macro data.  All rolling statistics are computed strictly on
past observations (no lookahead bias).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Lag / return features
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame, column: str, lags: list[int]) -> pd.DataFrame:
    """Add lagged values of *column* as new columns."""
    for lag in lags:
        df[f"{column}_lag{lag}"] = df[column].shift(lag)
    return df


def add_return_features(df: pd.DataFrame, column: str, periods: list[int]) -> pd.DataFrame:
    """Add percentage-return features over *periods* look-back windows."""
    for p in periods:
        df[f"{column}_ret{p}"] = df[column].pct_change(p)
    return df


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: pd.DataFrame,
    column: str,
    windows: list[int],
) -> pd.DataFrame:
    """Add rolling mean, std, min, and max for *column* over *windows*."""
    for w in windows:
        df[f"{column}_ma{w}"] = df[column].rolling(w).mean()
        df[f"{column}_std{w}"] = df[column].rolling(w).std()
        df[f"{column}_min{w}"] = df[column].rolling(w).min()
        df[f"{column}_max{w}"] = df[column].rolling(w).max()
    return df


# ---------------------------------------------------------------------------
# Technical indicators (manual implementations — no external dependency)
# ---------------------------------------------------------------------------

def add_rsi(df: pd.DataFrame, column: str, window: int = 14) -> pd.DataFrame:
    """Relative Strength Index."""
    delta = df[column].diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    df[f"{column}_rsi{window}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(
    df: pd.DataFrame,
    column: str,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    df[f"{column}_macd"] = macd_line
    df[f"{column}_macd_signal"] = signal_line
    df[f"{column}_macd_hist"] = macd_line - signal_line
    return df


def add_bollinger_bands(
    df: pd.DataFrame,
    column: str,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Upper band, lower band, and %B position."""
    ma = df[column].rolling(window).mean()
    std = df[column].rolling(window).std()
    df[f"{column}_bb_upper{window}"] = ma + num_std * std
    df[f"{column}_bb_lower{window}"] = ma - num_std * std
    df[f"{column}_bb_pct{window}"] = (df[column] - (ma - num_std * std)) / (
        2 * num_std * std
    )
    return df


# ---------------------------------------------------------------------------
# Cross-asset ratio features
# ---------------------------------------------------------------------------

def add_ratio_features(df: pd.DataFrame, numerator_col: str, denominator_col: str) -> pd.DataFrame:
    """Add ratio numerator_col / denominator_col as a feature."""
    safe_denom = df[denominator_col].replace(0, np.nan)
    df[f"{numerator_col}_to_{denominator_col}"] = df[numerator_col] / safe_denom
    return df


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-week and month features."""
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    return df


# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------

def add_target(
    df: pd.DataFrame,
    price_column: str = "gold_Close",
    horizon: int = 1,
    task: str = "regression",
) -> pd.DataFrame:
    """
    Add the target variable.

    Parameters
    ----------
    price_column : column containing the gold closing price
    horizon      : number of trading days ahead to predict
    task         : 'regression' → next close price
                   'classification' → 1 if price goes up, 0 otherwise
    """
    future_price = df[price_column].shift(-horizon)
    if task == "regression":
        df["target"] = future_price
    elif task == "classification":
        df["target"] = (future_price > df[price_column]).astype(int)
    else:
        raise ValueError(f"Unknown task '{task}'. Choose 'regression' or 'classification'.")
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    task: str = "regression",
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Apply the complete feature engineering pipeline to a raw market DataFrame.

    Expected columns (from download_market_data.py):
        gold_Close, sp500_Close, nasdaq_Close, dxy_Close
    Optional FRED columns (from download_fred_data.py):
        cpi, fed_funds_rate, treasury_10y, unemployment, breakeven_inflation_10y

    Returns a DataFrame with features + 'target' column; rows with NaN
    values (from rolling windows at the start) are dropped.
    """
    df = df.copy()

    # -- Gold features --
    df = add_lag_features(df, "gold_Close", lags=[1, 2, 3, 5, 10, 20])
    df = add_return_features(df, "gold_Close", periods=[1, 5, 10, 20])
    df = add_rolling_features(df, "gold_Close", windows=[5, 10, 20, 50, 200])
    df = add_rsi(df, "gold_Close", window=14)
    df = add_macd(df, "gold_Close")
    df = add_bollinger_bands(df, "gold_Close", window=20)

    # -- Other asset features --
    for col in ["sp500_Close", "nasdaq_Close", "dxy_Close"]:
        if col in df.columns:
            df = add_return_features(df, col, periods=[1, 5, 20])
            df = add_rolling_features(df, col, windows=[10, 20])

    # -- Cross-asset ratios --
    if "sp500_Close" in df.columns:
        df = add_ratio_features(df, "gold_Close", "sp500_Close")
    if "dxy_Close" in df.columns:
        df = add_ratio_features(df, "gold_Close", "dxy_Close")

    # -- FRED macro features (forward-fill already applied during download) --
    fred_cols = [
        "cpi", "fed_funds_rate", "treasury_10y", "unemployment", "breakeven_inflation_10y"
    ]
    for col in fred_cols:
        if col in df.columns:
            df = add_return_features(df, col, periods=[1, 3])

    # -- Calendar --
    df = add_calendar_features(df)

    # -- Target --
    df = add_target(df, price_column="gold_Close", horizon=horizon, task=task)

    # Drop rows with any NaN (from rolling windows or target shift)
    df.dropna(inplace=True)

    return df
