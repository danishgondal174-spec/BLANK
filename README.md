# Gold ML Project

A machine learning pipeline for gold price prediction using Yahoo Finance market data and macroeconomic indicators from FRED.

## Project Structure

```
BLANK/
├── data/                        # Raw and processed CSV data files
├── models/                      # Serialised trained models
├── notebooks/                   # Jupyter notebooks for exploration and modelling
├── pipelines/                   # Reusable Python modules
│   ├── data_loader.py           # Load and merge Yahoo + FRED data
│   ├── feature_engineer.py      # Lags, rolling stats, RSI, ratios, correlations
│   ├── trainer.py               # CV, hyperparameter tuning, evaluation, model save/load
│   └── gold_ml_pipeline.py      # End-to-end orchestration script
├── reports/                     # Saved plots (feature importance, SHAP, etc.)
├── scripts/
│   ├── download_market_data.py  # Fetch Yahoo Finance data
│   ├── fetch_fred_data.py       # Fetch FRED macroeconomic data
│   └── predict.py               # Generate predictions with the saved model
├── tests/
│   └── test_feature_engineer.py # Unit tests for feature engineering
├── NEXT_STEPS.md                # Actionable advanced ML checklist
├── README.md
└── requirements.txt
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download market data
python scripts/download_market_data.py

# 3. (Optional) Download FRED macro data
export FRED_API_KEY="your_key_here"
python scripts/fetch_fred_data.py

# 4. Run the full ML pipeline
python pipelines/gold_ml_pipeline.py

# 5. Generate a prediction from the saved model
python scripts/predict.py

# 6. Run tests
pytest tests/ -v
```

## What's Next?

See **[NEXT_STEPS.md](NEXT_STEPS.md)** for the full advanced checklist covering:

1. Time series cross-validation
2. Feature engineering (lags, rolling stats, RSI, correlations)
3. Hyperparameter optimisation
4. Model comparison (Ridge, Random Forest, XGBoost, LightGBM)
5. Model explainability (SHAP, permutation importance)
6. Macroeconomic data from FRED
7. Modular pipeline and productionisation
