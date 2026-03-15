"""
feature_engineer.py
-------------------
Reusable feature-engineering functions for the gold price ML pipeline.

All functions take a DataFrame indexed by date (business-day frequency) and
return a *new* DataFrame with the added features.  They never modify the input
in-place so they are safe to chain.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------

def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int] = (1, 5, 21),
) -> pd.DataFrame:
    """
    Add lagged versions of *columns* at each lag in *lags* (trading days).

    Parameters
    ----------
    df      : DataFrame indexed by date, sorted ascending.
    columns : Column names to lag.
    lags    : Lag offsets in rows (1 = previous day, 5 ≈ 1 week, 21 ≈ 1 month).

    Returns
    -------
    DataFrame with new columns named ``{col}_lag_{lag}``.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
        for lag in lags:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)
    return out


# ---------------------------------------------------------------------------
# Return features
# ---------------------------------------------------------------------------

def add_return_features(
    df: pd.DataFrame,
    columns: list[str],
    periods: list[int] = (1, 5),
) -> pd.DataFrame:
    """
    Add percentage-return features for *columns* over each period.

    Returns
    -------
    DataFrame with new columns named ``{col}_return_{period}d``.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
        for period in periods:
            out[f"{col}_return_{period}d"] = out[col].pct_change(period)
    return out


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int] = (7, 30),
    stats: list[str] = ("mean", "std"),
) -> pd.DataFrame:
    """
    Add rolling mean and/or standard deviation features.

    Parameters
    ----------
    df      : Input DataFrame.
    columns : Columns to compute rolling statistics for.
    windows : Rolling window sizes (in rows).
    stats   : Statistics to compute; supported values: ``'mean'``, ``'std'``.

    Returns
    -------
    DataFrame with new columns named ``{col}_rolling{window}_{stat}``.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
        for window in windows:
            r = out[col].rolling(window)
            if "mean" in stats:
                out[f"{col}_rolling{window}_mean"] = r.mean()
            if "std" in stats:
                out[f"{col}_rolling{window}_std"] = r.std()
    return out


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) for a price series.

    Parameters
    ----------
    series : Closing price series.
    period : Look-back period (default 14 days).

    Returns
    -------
    pd.Series of RSI values in the range [0, 100].
    """
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0.0, float("nan"))
    return 100.0 - (100.0 / (1.0 + rs))


def add_rsi(
    df: pd.DataFrame,
    columns: list[str],
    period: int = 14,
) -> pd.DataFrame:
    """
    Add RSI columns for each column in *columns*.

    Returns
    -------
    DataFrame with new columns named ``{col}_rsi{period}``.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
        out[f"{col}_rsi{period}"] = compute_rsi(out[col], period=period)
    return out


# ---------------------------------------------------------------------------
# Cross-asset ratio & correlation
# ---------------------------------------------------------------------------

def add_ratio_features(
    df: pd.DataFrame,
    ratios: list[tuple[str, str]],
) -> pd.DataFrame:
    """
    Add price-ratio features.

    Parameters
    ----------
    df     : Input DataFrame.
    ratios : List of ``(numerator_col, denominator_col)`` tuples.

    Returns
    -------
    DataFrame with new columns named ``{num}_to_{den}_ratio``.
    """
    out = df.copy()
    for num, den in ratios:
        if num not in out.columns:
            raise KeyError(f"Column '{num}' not found.")
        if den not in out.columns:
            raise KeyError(f"Column '{den}' not found.")
        out[f"{num}_to_{den}_ratio"] = out[num] / out[den].replace(0.0, float("nan"))
    return out


def add_rolling_correlations(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    windows: list[int] = (30,),
) -> pd.DataFrame:
    """
    Add rolling Pearson correlation between pairs of columns.

    Parameters
    ----------
    df      : Input DataFrame (should already contain 1-day return columns).
    pairs   : List of ``(col_a, col_b)`` column-name pairs.
    windows : Rolling window sizes.

    Returns
    -------
    DataFrame with new columns named ``{col_a}_x_{col_b}_corr{window}``.
    """
    out = df.copy()
    for col_a, col_b in pairs:
        if col_a not in out.columns:
            raise KeyError(f"Column '{col_a}' not found.")
        if col_b not in out.columns:
            raise KeyError(f"Column '{col_b}' not found.")
        for window in windows:
            out[f"{col_a}_x_{col_b}_corr{window}"] = (
                out[col_a].rolling(window).corr(out[col_b])
            )
    return out


# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------

def add_target(
    df: pd.DataFrame,
    price_col: str = "gold_Close",
    forward_period: int = 1,
    task: str = "regression",
) -> pd.DataFrame:
    """
    Add a prediction target column.

    Parameters
    ----------
    df             : Input DataFrame.
    price_col      : Column containing the asset price.
    forward_period : How many days ahead to predict (default 1 = next day).
    task           : ``'regression'`` → next-day price; ``'classification'`` → 1 if
                     price goes up, 0 otherwise.

    Returns
    -------
    DataFrame with a new ``target`` column.  The last *forward_period* rows will
    have NaN targets (they have no known future) and should be removed before
    training.
    """
    out = df.copy()
    future_price = out[price_col].shift(-forward_period)
    if task == "regression":
        out["target"] = future_price
    elif task == "classification":
        out["target"] = (future_price > out[price_col]).astype(float)
    else:
        raise ValueError(f"Unsupported task '{task}'. Use 'regression' or 'classification'.")
    return out


# ---------------------------------------------------------------------------
# Full feature pipeline (convenience wrapper)
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    price_cols: list[str] | None = None,
    task: str = "regression",
) -> pd.DataFrame:
    """
    Apply all feature-engineering steps in the recommended order.

    Parameters
    ----------
    df         : Raw market DataFrame (business-day frequency, no future data).
    price_cols : Asset close-price columns to engineer features from.
                 Defaults to ``['gold_Close', 'sp500_Close', 'dxy_Close']``.
    task       : ``'regression'`` or ``'classification'``.

    Returns
    -------
    Feature-engineered DataFrame with ``target`` column added and NaN rows
    (from lags / rolling windows) dropped.
    """
    if price_cols is None:
        price_cols = ["gold_Close", "sp500_Close", "dxy_Close"]

    present = [c for c in price_cols if c in df.columns]
    if not present:
        raise ValueError(
            "None of the specified price columns were found in the DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()

    out = add_lag_features(out, columns=present)
    out = add_return_features(out, columns=present)
    out = add_rolling_features(out, columns=present)

    if "gold_Close" in present:
        out = add_rsi(out, columns=["gold_Close"])

    if "gold_Close" in present and "sp500_Close" in present:
        out = add_ratio_features(out, ratios=[("gold_Close", "sp500_Close")])

    # Rolling correlation on 1-day return columns (added by add_return_features)
    ret_gold   = "gold_Close_return_1d"
    ret_dxy    = "dxy_Close_return_1d"
    if ret_gold in out.columns and ret_dxy in out.columns:
        out = add_rolling_correlations(out, pairs=[(ret_gold, ret_dxy)])

    out = add_target(out, price_col="gold_Close", task=task)

    # Drop rows that contain NaN (from lags/rolling windows or target shift)
    out = out.dropna()

    return out
