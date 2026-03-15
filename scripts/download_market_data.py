"""
Download historical market data from Yahoo Finance.

Fetches OHLCV data for gold (GC=F), S&P 500 (^GSPC), Nasdaq (^IXIC),
and the DXY US Dollar Index (DX-Y.NYB) and saves a merged CSV to
data/yahoo_market_data.csv.
"""

import os
import pandas as pd
import yfinance as yf

TICKERS = {
    "gold": "GC=F",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "dxy": "DX-Y.NYB",
}

START_DATE = "2010-01-01"
END_DATE = None  # None means today
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "yahoo_market_data.csv")


def download_ticker(name: str, symbol: str, start: str, end) -> pd.DataFrame:
    """Download OHLCV data for a single ticker and prefix column names."""
    print(f"Downloading {name} ({symbol}) ...")
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    df = df.copy()
    df.columns = [f"{name}_{col}" for col in df.columns]
    return df


def main() -> None:
    frames = []
    for name, symbol in TICKERS.items():
        df = download_ticker(name, symbol, START_DATE, END_DATE)
        frames.append(df)

    merged = pd.concat(frames, axis=1)
    merged.index.name = "Date"
    merged.dropna(how="all", inplace=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    merged.to_csv(OUTPUT_PATH)
    print(f"\nSaved {len(merged)} rows to {OUTPUT_PATH}")
    print(merged.tail())


if __name__ == "__main__":
    main()
