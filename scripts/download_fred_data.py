"""
Download macroeconomic indicators from FRED (Federal Reserve Economic Data).

Series downloaded:
  - CPIAUCSL  : Consumer Price Index (inflation)
  - FEDFUNDS  : Federal Funds Effective Rate
  - DGS10     : 10-Year Treasury Constant Maturity Rate
  - DTWEXBGS  : Trade-Weighted US Dollar Index
  - UNRATE    : Unemployment Rate
  - T10YIE    : 10-Year Breakeven Inflation Rate (inflation expectations)

The data is saved to data/fred_macro_data.csv.

Usage
-----
1. Obtain a free FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html
2. Set it as an environment variable:
       export FRED_API_KEY=<your_key>
   or pass it via the --api-key CLI argument.
3. Run:
       python scripts/download_fred_data.py
"""

import argparse
import os
import pandas as pd

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "fred_macro_data.csv"
)

FRED_SERIES = {
    "cpi": "CPIAUCSL",
    "fed_funds_rate": "FEDFUNDS",
    "treasury_10y": "DGS10",
    "usd_index_fred": "DTWEXBGS",
    "unemployment": "UNRATE",
    "breakeven_inflation_10y": "T10YIE",
}

START_DATE = "2010-01-01"


def get_api_key(cli_key: str | None = None) -> str:
    """Return FRED API key from CLI argument or environment variable."""
    key = cli_key or os.environ.get("FRED_API_KEY", "")
    if not key:
        raise ValueError(
            "FRED API key not found. Set the FRED_API_KEY environment variable "
            "or pass --api-key <key>."
        )
    return key


def download_fred_series(api_key: str) -> pd.DataFrame:
    """Download all configured FRED series and return a merged DataFrame."""
    try:
        from fredapi import Fred
    except ImportError as exc:
        raise ImportError(
            "fredapi is not installed. Run: pip install fredapi"
        ) from exc

    fred = Fred(api_key=api_key)
    frames = {}
    for name, series_id in FRED_SERIES.items():
        print(f"Downloading FRED series {series_id} ({name}) ...")
        try:
            s = fred.get_series(series_id, observation_start=START_DATE)
            frames[name] = s
        except Exception as exc:
            print(f"  Warning: could not download {series_id}: {exc}")

    df = pd.DataFrame(frames)
    df.index.name = "Date"
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FRED macro data.")
    parser.add_argument("--api-key", default=None, help="FRED API key")
    args = parser.parse_args()

    api_key = get_api_key(args.api_key)
    df = download_fred_series(api_key)

    # Forward-fill monthly/weekly series to align with daily market data
    df = df.asfreq("B").ffill()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH)
    print(f"\nSaved {len(df)} rows to {OUTPUT_PATH}")
    print(df.tail())


if __name__ == "__main__":
    main()
