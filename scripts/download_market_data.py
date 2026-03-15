"""
download_market_data.py
-----------------------
Downloads historical market data from Yahoo Finance for:
  - Gold futures          (GC=F)
  - S&P 500 index         (^GSPC)
  - US Dollar Index       (DX-Y.NYB)
  - Crude Oil futures     (CL=F)
  - 10-Year Treasury Yield (^TNX)

All series are merged on the Date column and saved to:
  data/yahoo_market_data.csv

Usage:
  python scripts/download_market_data.py
  python scripts/download_market_data.py --start 2010-01-01 --end 2024-12-31

Requirements:
  pip install -r requirements.txt
"""

import argparse
import os
import sys

import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TICKERS = {
    "gold":  "GC=F",       # Gold Futures
    "sp500": "^GSPC",      # S&P 500
    "dxy":   "DX-Y.NYB",   # US Dollar Index
    "oil":   "CL=F",       # Crude Oil Futures
    "tnx":   "^TNX",       # 10-Year Treasury Yield
}

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "yahoo_market_data.csv")

DEFAULT_START = "2010-01-01"
DEFAULT_END   = "2024-12-31"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def download_ticker(name: str, symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data for a single ticker and prefix column names."""
    print(f"  Downloading {name} ({symbol}) …")
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        print(f"  WARNING: No data returned for {name} ({symbol}). Skipping.")
        return pd.DataFrame()

    # Keep only OHLCV columns
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in cols if c in df.columns]].copy()

    # Prefix column names with ticker name
    df.columns = [f"{name}_{col}" for col in df.columns]
    df.index.name = "Date"

    return df


def merge_all(tickers: dict, start: str, end: str) -> pd.DataFrame:
    """Download and merge all tickers into one DataFrame."""
    frames = []
    for name, symbol in tickers.items():
        df = download_ticker(name, symbol, start, end)
        if not df.empty:
            frames.append(df)

    if not frames:
        sys.exit("ERROR: No data was downloaded. Check your internet connection.")

    merged = pd.concat(frames, axis=1, join="outer")
    merged.sort_index(inplace=True)
    # Forward-fill weekends / holidays, then drop rows still missing gold Close
    merged.ffill(inplace=True)
    merged.dropna(subset=["gold_Close"], inplace=True)
    return merged


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download and merge Yahoo Finance market data.")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",   default=DEFAULT_END,   help="End date   (YYYY-MM-DD)")
    args = parser.parse_args()

    print(f"\n=== Yahoo Finance Market Data Downloader ===")
    print(f"  Date range : {args.start} → {args.end}")
    print(f"  Output file: {os.path.abspath(OUTPUT_FILE)}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    merged = merge_all(TICKERS, args.start, args.end)

    merged.to_csv(OUTPUT_FILE)

    print(f"\n✓  Saved {len(merged):,} rows × {len(merged.columns)} columns to:")
    print(f"   {os.path.abspath(OUTPUT_FILE)}")
    print(f"\nColumn list:\n  " + "\n  ".join(merged.columns.tolist()))
    print("\nSample (first 3 rows):")
    print(merged.head(3).to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
