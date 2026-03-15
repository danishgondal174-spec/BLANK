"""
data_loader.py
--------------
Functions to load and merge market data for the gold price ML pipeline.
"""

from __future__ import annotations

import os
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_yahoo_data(path: str | None = None) -> pd.DataFrame:
    """
    Load the Yahoo Finance market data CSV produced by download_market_data.py.

    Parameters
    ----------
    path : Optional explicit file path.  Defaults to ``data/yahoo_market_data.csv``
           relative to the project root.

    Returns
    -------
    DataFrame indexed by ``Date`` (DatetimeIndex), sorted ascending.
    """
    if path is None:
        path = os.path.join(DATA_DIR, "yahoo_market_data.csv")
    path = os.path.abspath(path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Yahoo market data not found at '{path}'.\n"
            "Run 'python scripts/download_market_data.py' first."
        )

    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df.sort_index(inplace=True)
    return df


def load_fred_data(path: str | None = None) -> pd.DataFrame:
    """
    Load the FRED macroeconomic data CSV produced by fetch_fred_data.py.

    Returns
    -------
    DataFrame indexed by ``Date`` (DatetimeIndex), sorted ascending.
    Returns an empty DataFrame if the file does not exist (FRED data is optional).
    """
    if path is None:
        path = os.path.join(DATA_DIR, "fred_macro_data.csv")
    path = os.path.abspath(path)

    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df.sort_index(inplace=True)
    return df


def merge_datasets(
    yahoo: pd.DataFrame,
    fred: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge Yahoo Finance and FRED datasets on their date indexes.

    FRED series are forward-filled to align monthly/weekly releases to the
    business-day frequency of Yahoo data.

    Parameters
    ----------
    yahoo : Yahoo Finance DataFrame (business-day index).
    fred  : Optional FRED macro DataFrame.  Pass ``None`` or an empty DataFrame
            to skip merging.

    Returns
    -------
    Merged DataFrame on the intersection of dates present in Yahoo data.
    """
    if fred is None or fred.empty:
        return yahoo.copy()

    merged = yahoo.join(fred, how="left")
    # Forward-fill monthly FRED series (e.g. CPI, FEDFUNDS) to daily
    fred_cols = list(fred.columns)
    merged[fred_cols] = merged[fred_cols].ffill()
    return merged


def load_all_data(
    yahoo_path: str | None = None,
    fred_path:  str | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: load Yahoo + FRED data and return merged DataFrame.
    """
    yahoo  = load_yahoo_data(yahoo_path)
    fred   = load_fred_data(fred_path)
    merged = merge_datasets(yahoo, fred if not fred.empty else None)
    print(
        f"Loaded data: {merged.shape[0]} rows × {merged.shape[1]} columns "
        f"({merged.index.min().date()} → {merged.index.max().date()})"
    )
    return merged
