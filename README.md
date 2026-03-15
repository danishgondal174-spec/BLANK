# Gold ML Project

A beginner-to-research-grade machine learning project for predicting gold prices using historical market data (gold, S&P 500, DXY, and more).

## Contents

| File / Folder | Description |
|---------------|-------------|
| `data/yahoo_market_data.csv` | Merged historical market data downloaded from Yahoo Finance |
| `scripts/download_market_data.py` | Script to fetch and merge all market data |
| `notebooks/` | Jupyter notebooks for EDA, feature engineering, and modelling |
| [`GUIDE.md`](GUIDE.md) | **Step-by-step ML guide** (start here) |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download market data
python scripts/download_market_data.py

# 3. Open the ML guide
# See GUIDE.md for the full step-by-step walkthrough
```

## Guide Overview

The [`GUIDE.md`](GUIDE.md) covers:

- ✅ Defining your target variable (price regression vs direction classification)
- ✅ Choosing features: gold lags, S&P 500, DXY, technical indicators
- ✅ Feature engineering: moving averages, returns, rolling correlations
- ✅ Time-series train/test split (no data leakage)
- ✅ Model selection: baseline → linear → tree-based → boosting
- ✅ Training code examples (Logistic Regression, Random Forest, XGBoost)
- ✅ Evaluation: accuracy, F1, confusion matrix, cumulative return plots
- ✅ Tips for avoiding lookahead bias and data leakage
- ✅ Next steps: hyperparameter tuning, time-series cross-validation, backtesting
