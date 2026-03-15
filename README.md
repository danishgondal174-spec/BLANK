# Gold ML Project

Historical market data pipeline using Python and the Yahoo Finance API.

---

## What it downloads

| Name | Ticker | Description |
|------|--------|-------------|
| gold | `GC=F` | Gold Futures |
| sp500 | `^GSPC` | S&P 500 Index |
| nasdaq | `^IXIC` | Nasdaq Composite |
| dxy | `DX-Y.NYB` | US Dollar Index |
| crude_oil | `CL=F` | Crude Oil (WTI) Futures |
| us10y | `^TNX` | US 10-Year Treasury Yield |
| uup | `UUP` | Dollar ETF (DXY proxy) |
| vix | `^VIX` | CBOE Volatility Index |

All series are merged into a single CSV file at `data/market_data.csv`.

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the download script

```bash
python download_data.py
```

### 3. Check the output

```
data/market_data.csv
```

Each row is one trading day. Columns follow the pattern `<name>_<metric>`,
for example `gold_close`, `sp500_volume`, `dxy_open`, etc.

---

## Requirements

- Python 3.10+
- `yfinance >= 0.2.0`
- `pandas >= 1.3.0`

---

## Notes

- Tickers that are unavailable or return no data are skipped gracefully;
  the script still saves whatever data it could retrieve.
- The date range defaults to **2010-01-01 → 2026-03-15**. Edit `START_DATE`
  and `END_DATE` at the top of `download_data.py` to change it.
- The `data/` directory is created automatically if it does not exist.
