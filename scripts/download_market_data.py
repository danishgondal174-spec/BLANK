"""
download_market_data.py
-----------------------
Downloads historical market data from Yahoo Finance using the yfinance library
and merges it into a single CSV file at data/yahoo_market_data.csv.

Usage:
    python scripts/download_market_data.py
    python scripts/download_market_data.py --start 2015-01-01 --end 2026-01-01
    python scripts/download_market_data.py --interval 1wk

Supported intervals: 1d (daily, default), 1wk (weekly), 1mo (monthly)
"""

import argparse
import os
import sys

import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Ticker configuration
# ---------------------------------------------------------------------------
# Each entry: (prefix_for_columns, yahoo_ticker, human_readable_name, required)
# required=True  -> script exits if this ticker fails completely
# required=False -> ticker is skipped gracefully when unavailable
TICKERS = [
    ("gold",   "GC=F",      "Gold Futures",        True),
    ("sp500",  "^GSPC",     "S&P 500",             True),
    ("nasdaq", "^IXIC",     "Nasdaq Composite",    True),
    # Dollar Index — sometimes unavailable; UUP ETF is used as a proxy fallback
    ("dxy",    "DX-Y.NYB",  "Dollar Index",        False),
    ("uup",    "UUP",       "Dollar Index ETF (proxy)", False),
    # Optional extras (comment out if you don't need them)
    ("wti",    "CL=F",      "WTI Crude Oil",       False),
    ("tnx",    "^TNX",      "US 10Y Treasury Yield", False),
]

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "yahoo_market_data.csv",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download historical market data from Yahoo Finance."
    )
    parser.add_argument(
        "--start",
        default="2010-01-01",
        help="Start date in YYYY-MM-DD format (default: 2010-01-01)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        choices=["1d", "1wk", "1mo"],
        help="Data interval (default: 1d for daily)",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
        help=f"Output CSV path (default: {OUTPUT_PATH})",
    )
    return parser.parse_args()


def download_ticker(prefix, ticker, name, start, end, interval):
    """Download data for a single ticker and return a renamed DataFrame or None."""
    print(f"  Downloading {name} ({ticker}) ...", end=" ", flush=True)
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR — {exc}")
        return None

    if df is None or df.empty:
        print("No data returned.")
        return None

    # Flatten MultiIndex columns produced by newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Rename columns to <prefix>_<column> (e.g. gold_close)
    df.columns = [f"{prefix}_{col.lower().replace(' ', '_')}" for col in df.columns]

    print(f"OK ({len(df)} rows)")
    return df


def main():
    args = parse_args()

    print("=" * 60)
    print("Gold ML — Yahoo Finance Data Downloader")
    print("=" * 60)
    print(f"Date range : {args.start}  →  {args.end or 'today'}")
    print(f"Interval   : {args.interval}")
    print(f"Output     : {args.output}")
    print()

    frames = {}
    for prefix, ticker, name, required in TICKERS:
        df = download_ticker(prefix, ticker, name, args.start, args.end, args.interval)
        if df is not None:
            frames[prefix] = df
        elif required:
            print(
                f"\nFATAL: Required ticker '{ticker}' ({name}) returned no data. "
                "Check your internet connection and try again.",
                file=sys.stderr,
            )
            sys.exit(1)

    if not frames:
        print("No data downloaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Dollar-index fallback: if primary DXY failed but proxy UUP succeeded, keep UUP only
    if "dxy" not in frames and "uup" in frames:
        print(
            "\nNote: Dollar Index (DX-Y.NYB) was unavailable. "
            "Using UUP ETF as a proxy for DXY.\n"
        )
    elif "dxy" in frames and "uup" in frames:
        # Both succeeded — drop the proxy to avoid duplicate dollar-index signals
        del frames["uup"]

    # Outer-join all series on the shared date index
    merged = None
    for df in frames.values():
        merged = df if merged is None else merged.join(df, how="outer")

    merged = merged.sort_index()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    merged.to_csv(args.output)

    print()
    print("=" * 60)
    print("Download summary")
    print("=" * 60)
    for prefix, df in frames.items():
        print(f"  {prefix:10s}  {len(df):>6d} rows")
    print()
    print(f"Merged dataset shape : {merged.shape[0]} rows × {merged.shape[1]} columns")
    print(f"Saved to             : {args.output}")
    print()
    print(merged.tail(3).to_string())


if __name__ == "__main__":
    main()
