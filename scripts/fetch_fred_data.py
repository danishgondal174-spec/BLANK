"""
fetch_fred_data.py
------------------
Download macroeconomic time series from the Federal Reserve Economic Data (FRED)
API and save them to data/fred_macro_data.csv.

Usage
-----
    export FRED_API_KEY="your_key_here"
    python scripts/fetch_fred_data.py

Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html
"""

import os
import sys
import pandas as pd


SERIES_IDS = {
    "treasury_10y": "DGS10",       # 10-Year Treasury Constant Maturity Rate
    "cpi":          "CPIAUCSL",    # CPI All Urban Consumers (monthly, SA)
    "usd_eur":      "DEXUSEU",     # USD/EUR spot exchange rate (daily)
    "fed_funds":    "FEDFUNDS",    # Effective Federal Funds Rate (monthly)
    "wti_oil":      "DCOILWTICO",  # WTI Crude Oil Spot Price (daily)
}

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "fred_macro_data.csv"
)


def fetch_fred_data(api_key: str, start_date: str = "2010-01-01") -> pd.DataFrame:
    """
    Fetch each series in SERIES_IDS from FRED, align to business-day frequency,
    forward-fill gaps (e.g. weekends, monthly releases), and return a combined
    DataFrame indexed by date.
    """
    try:
        from fredapi import Fred
    except ImportError:
        print("fredapi is not installed. Run: pip install fredapi")
        sys.exit(1)

    fred = Fred(api_key=api_key)
    frames = {}

    for name, sid in SERIES_IDS.items():
        print(f"  Downloading {name} ({sid})…")
        series = fred.get_series(sid, observation_start=start_date)
        frames[name] = series.rename(name)

    df = pd.concat(frames.values(), axis=1)
    df.index = pd.to_datetime(df.index)

    # Resample to business-day frequency; forward-fill monthly/weekly series
    df = df.resample("B").last().ffill()

    return df


def main() -> None:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print(
            "ERROR: FRED_API_KEY environment variable is not set.\n"
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "Then run:  export FRED_API_KEY='your_key_here'"
        )
        sys.exit(1)

    print("Fetching FRED macroeconomic data…")
    df = fetch_fred_data(api_key)

    output_path = os.path.abspath(OUTPUT_PATH)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index_label="Date")

    print(f"Saved {df.shape[0]} rows × {df.shape[1]} columns → {output_path}")
    print(df.tail())


if __name__ == "__main__":
    main()
