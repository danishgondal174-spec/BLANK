# Gold Price ML Pipeline

End-to-end machine-learning pipeline for predicting gold (XAU/USD) prices using
Yahoo Finance market data and optional FRED macroeconomic indicators.

## Features

| Area | What's included |
|---|---|
| **Data** | Yahoo Finance (gold, S&P 500, Nasdaq, DXY) + FRED macro data (CPI, Fed Funds Rate, 10-Y Treasury, unemployment, breakeven inflation) |
| **Feature engineering** | Lags, percentage returns, rolling stats (mean/std/min/max), RSI, MACD, Bollinger Bands, cross-asset ratios, calendar features |
| **Models** | Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost |
| **Validation** | `TimeSeriesSplit` cross-validation — no data leakage, no random shuffling |
| **Tuning** | `RandomizedSearchCV` / `GridSearchCV` with time-series CV |
| **Feature selection** | Feature-importance ranking; re-train on top-N features |
| **Production** | Modular `src/` package; CLI pipeline; `joblib` model serialisation |

## Project structure

```
.
├── data/                        # Downloaded CSV files (git-ignored)
├── models/                      # Saved model files (git-ignored)
├── notebooks/
│   └── gold_ml_advanced.ipynb   # Interactive walkthrough notebook
├── scripts/
│   ├── download_market_data.py  # Fetch Yahoo Finance data
│   └── download_fred_data.py    # Fetch FRED macro data (needs API key)
├── src/
│   ├── features.py              # Feature engineering
│   ├── models.py                # Model catalogue + evaluation helpers
│   ├── tuning.py                # Hyperparameter tuning wrappers
│   ├── validation.py            # TimeSeriesSplit CV utilities
│   └── pipeline.py              # End-to-end CLI pipeline
├── tests/
│   └── test_pipeline.py         # Unit tests
└── requirements.txt
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download market data

```bash
python scripts/download_market_data.py
```

### 3. (Optional) Download FRED macro data

Get a free API key at <https://fred.stlouisfed.org/docs/api/api_key.html>, then:

```bash
export FRED_API_KEY=<your_key>
python scripts/download_fred_data.py
```

### 4. Run the end-to-end pipeline

```bash
# Regression (predict next-day gold close price)
python -m src.pipeline --task regression

# Classification (predict up/down direction)
python -m src.pipeline --task classification

# With hyperparameter tuning
python -m src.pipeline --task regression --tune --tune-iter 30

# Using top-30 features only
python -m src.pipeline --task regression --top-features 30
```

### 5. Explore the notebook

```bash
pip install notebook
jupyter notebook notebooks/gold_ml_advanced.ipynb
```

## CLI reference

```
usage: python -m src.pipeline [-h] [--task {regression,classification}]
                               [--horizon N] [--test-ratio R]
                               [--cv-splits N] [--tune] [--tune-iter N]
                               [--top-features N]

Options:
  --task            regression (default) or classification
  --horizon         Prediction horizon in trading days (default: 1)
  --test-ratio      Fraction of data for the test set (default: 0.2)
  --cv-splits       TimeSeriesSplit folds (default: 5)
  --tune            Run RandomizedSearchCV for the best model
  --tune-iter       RandomizedSearchCV iterations (default: 20)
  --top-features    Re-train using only the top N features
```

## Running tests

```bash
python -m pytest tests/ -v
```

## Next steps / avenues for improvement

- **More FRED series** — oil prices (DCOILWTICO), VIX, M2 money supply
- **Longer horizons** — change `--horizon` to predict 5- or 21-day returns
- **LSTM / Transformer** — sequence models for capturing temporal patterns
- **Backtesting** — simulate a trading strategy driven by model signals
- **Periodic retraining** — add a GitHub Actions workflow to retrain weekly

