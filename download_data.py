"""
download_data.py
================
Downloads historical market data from Yahoo Finance for multiple tickers
and saves a unified, merged CSV file to the data/ directory.

Tickers included:
    GC=F       - Gold Futures
    ^GSPC      - S&P 500 Index
    ^IXIC      - Nasdaq Composite Index
    DX-Y.NYB   - US Dollar Index
    CL=F       - Crude Oil Futures (WTI)
    ^TNX       - US 10-Year Treasury Yield
    UUP        - Invesco DB US Dollar Index Bullish Fund (proxy if DX-Y.NYB fails)
    ^VIX       - CBOE Volatility Index

Usage:
    pip install -r requirements.txt
    python download_data.py

Output:
    data/market_data.csv  – one row per trading day, columns for each ticker/metric
"""

import os
import sys

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Date range for historical data
START_DATE = "2010-01-01"
END_DATE = "2026-03-15"

# Tickers to download: friendly_name -> Yahoo Finance ticker symbol
TICKERS = {
    "gold": "GC=F",         # Gold Futures
    "sp500": "^GSPC",       # S&P 500 Index
    "nasdaq": "^IXIC",      # Nasdaq Composite
    "dxy": "DX-Y.NYB",      # US Dollar Index
    "crude_oil": "CL=F",    # Crude Oil (WTI) Futures
    "us10y": "^TNX",        # US 10-Year Treasury Yield
    "uup": "UUP",           # Dollar ETF proxy (backup for DX-Y.NYB)
    "vix": "^VIX",          # Volatility Index
}

# Which price columns to keep from Yahoo Finance (all in lower-case)
KEEP_COLUMNS = ["open", "high", "low", "close", "volume"]

# Output path
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "market_data.csv")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def download_ticker(name: str, symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Download daily OHLCV data for a single ticker from Yahoo Finance.

    Parameters
    ----------
    name   : friendly name used to prefix column headers (e.g. "gold")
    symbol : Yahoo Finance ticker symbol (e.g. "GC=F")
    start  : start date string "YYYY-MM-DD"
    end    : end date string "YYYY-MM-DD"

    Returns
    -------
    pd.DataFrame with a DatetimeIndex and prefixed columns, or None on failure.
    """
    print(f"  Downloading {name} ({symbol}) ...", end=" ", flush=True)
    try:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as exc:
        print(f"ERROR – {exc}")
        return None

    if df.empty:
        print("⚠  No data returned (ticker may be unavailable).")
        return None

    # Flatten MultiIndex columns produced by newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Normalise column names to lower-case
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    # Keep only the desired columns that actually exist in the data
    available = [c for c in KEEP_COLUMNS if c in df.columns]
    if not available:
        print(f"⚠  No expected columns found. Got: {list(df.columns)}")
        return None

    df = df[available].copy()

    # Add the ticker prefix so merged columns stay unambiguous
    df.columns = [f"{name}_{col}" for col in df.columns]

    # Ensure the index is a proper DatetimeIndex (date part only)
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"

    print(f"✓  {len(df)} rows, {len(df.columns)} columns")
    return df


def merge_dataframes(dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Outer-join multiple DataFrames on their date index.

    Parameters
    ----------
    dataframes : dict mapping name -> DataFrame

    Returns
    -------
    Single merged DataFrame sorted by date.
    """
    merged = None
    for name, df in dataframes.items():
        if merged is None:
            merged = df
        else:
            merged = merged.join(df, how="outer")
    if merged is not None:
        merged = merged.sort_index()
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Yahoo Finance Historical Market Data Downloader")
    print("=" * 60)
    print(f"  Date range : {START_DATE}  →  {END_DATE}")
    print(f"  Tickers    : {len(TICKERS)}")
    print()

    # ------------------------------------------------------------------
    # 1. Download each ticker
    # ------------------------------------------------------------------
    all_data: dict[str, pd.DataFrame] = {}

    for name, symbol in TICKERS.items():
        df = download_ticker(name, symbol, START_DATE, END_DATE)
        if df is not None:
            all_data[name] = df

    if not all_data:
        print("\n✗  No data could be downloaded. Check your internet connection.")
        sys.exit(1)

    print(f"\n  Successfully downloaded {len(all_data)}/{len(TICKERS)} tickers.\n")

    # ------------------------------------------------------------------
    # 2. Merge all DataFrames on the date index
    # ------------------------------------------------------------------
    print("  Merging datasets on date index ...", end=" ", flush=True)
    merged = merge_dataframes(all_data)
    print(f"✓  {len(merged)} rows × {len(merged.columns)} columns")

    # ------------------------------------------------------------------
    # 3. Save to CSV
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged.to_csv(OUTPUT_FILE)
    print(f"\n✅ Saved: {OUTPUT_FILE}")
    print(f"   Shape : {merged.shape[0]} rows × {merged.shape[1]} columns")
    print(f"   Date range in file : {merged.index.min().date()} → {merged.index.max().date()}")
    print()
    print("  Preview (last 5 rows):")
    print(merged.tail().to_string())
    print()


if __name__ == "__main__":
    main()
