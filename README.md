# Gold ML Project

Historical market data pipeline + Machine-Learning model using Python and the Yahoo Finance API.

---

## What it does

| Step | Script | Output |
|------|--------|--------|
| 1 – Download data | `download_data.py` | `data/market_data.csv` |
| 2 – Train ML model | `train_model.py` | `data/models/` |

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

### 2. Download historical market data

```bash
python download_data.py
```

Output: `data/market_data.csv` — one row per trading day with OHLCV columns
for each ticker (e.g. `gold_close`, `sp500_volume`).

### 3. Train the ML model

```bash
python train_model.py
```

The model predicts the **next trading day's Gold closing price**.

Output files in `data/models/`:

| File | Description |
|------|-------------|
| `gold_price_model.joblib` | Trained sklearn pipeline (scaler + Random Forest) |
| `metrics.json` | Test-set evaluation metrics (MAE, RMSE, MAPE, R²) |
| `actual_vs_predicted.png` | Actual vs predicted price chart |
| `feature_importance.png` | Top-20 most important features |

---

## ML model details

**Task**: Regression – predict next-day Gold close price  
**Algorithm**: Random Forest Regressor (300 trees)  
**Train/test split**: Chronological (first 80 % train, last 20 % test)

**Features engineered** from the raw data:

- Gold technical indicators: 5/10/20/50-day moving averages, Bollinger Bands, RSI-14, rolling volatility
- Gold lag features: close price of the last 1–5 days
- Gold multi-period returns: 1-day, 5-day, 10-day, 20-day
- Cross-asset closing prices and 1-day / 5-day returns: S&P 500, Nasdaq, DXY, Crude Oil, US 10Y Yield, VIX

---

## Requirements

- Python 3.10+
- `yfinance >= 0.2.0`
- `pandas >= 1.3.0`
- `scikit-learn >= 1.2.0`
- `matplotlib >= 3.5.0`
- `joblib >= 1.2.0`
- `numpy >= 1.23.0`

---

## Notes

- Tickers that are unavailable or return no data are skipped gracefully;
  the script still saves whatever data it could retrieve.
- The date range defaults to **2010-01-01 → 2026-03-15**. Edit `START_DATE`
  and `END_DATE` at the top of `download_data.py` to change it.
- The `data/` directory is created automatically if it does not exist.
- Generated CSV and model files are excluded from version control via `.gitignore`.

