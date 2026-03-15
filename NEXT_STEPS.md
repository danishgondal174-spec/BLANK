# Advanced Next Steps: Gold Price ML Pipeline

This checklist covers the recommended advanced engineering and research improvements after establishing the baseline gold price ML data pipeline and initial model. Work through steps in order — each builds on the previous.

---

## Step 1 — Time Series Cross-Validation

> **Why:** Random cross-validation leaks future data into training. Time-aware splits give honest estimates of out-of-sample performance.

- [ ] Replace any random `train_test_split` with scikit-learn's `TimeSeriesSplit`
- [ ] Choose a number of folds (5–10) that gives meaningful hold-out periods
- [ ] Report mean ± std of chosen metric (RMSE / MAE / accuracy) across folds
- [ ] Use the same split strategy in *every* subsequent tuning step

```python
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import numpy as np

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(
    model, X, y,
    cv=tscv,
    scoring="neg_mean_absolute_error",
)
print(f"MAE: {-scores.mean():.4f} ± {scores.std():.4f}")
```

---

## Step 2 — Feature Engineering

> **Why:** Raw OHLCV data rarely contains enough signal. Derived features often provide the biggest accuracy lift.

### 2a — Lag Features
- [ ] Add 1-day, 5-day, and 21-day lags for gold close price and key correlated assets (S&P 500, DXY)
- [ ] Add 1-day percentage return for each asset

```python
for lag in [1, 5, 21]:
    df[f"gold_lag_{lag}"] = df["gold_Close"].shift(lag)
    df[f"sp500_lag_{lag}"] = df["sp500_Close"].shift(lag)
    df[f"dxy_lag_{lag}"]   = df["dxy_Close"].shift(lag)

df["gold_return_1d"]  = df["gold_Close"].pct_change(1)
df["sp500_return_1d"] = df["sp500_Close"].pct_change(1)
```

### 2b — Rolling Statistics
- [ ] Compute 7-day and 30-day rolling mean and rolling standard deviation for gold and S&P 500
- [ ] Add a 14-day rolling RSI for gold

```python
for window in [7, 30]:
    df[f"gold_ma{window}"]  = df["gold_Close"].rolling(window).mean()
    df[f"gold_std{window}"] = df["gold_Close"].rolling(window).std()

# RSI helper
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)

df["gold_rsi14"] = compute_rsi(df["gold_Close"])
```

### 2c — Cross-Asset Correlations / Ratios
- [ ] Add the gold-to-S&P 500 price ratio
- [ ] Compute 30-day rolling correlation between daily gold returns and DXY returns

```python
df["gold_sp500_ratio"] = df["gold_Close"] / df["sp500_Close"]

df["gold_dxy_corr30"] = (
    df["gold_return_1d"]
    .rolling(30)
    .corr(df["dxy_Close"].pct_change())
)
```

- [ ] Drop rows with NaN values introduced by lags/rolling windows before training

---

## Step 3 — Hyperparameter Optimization

> **Why:** Default hyperparameters are rarely optimal. Systematic search can meaningfully reduce error.

- [ ] Choose a primary model (e.g. XGBoost, Random Forest, or Ridge Regression)
- [ ] Define a parameter grid / search space appropriate for that model
- [ ] Run `RandomizedSearchCV` (faster) or `GridSearchCV` (exhaustive) with `TimeSeriesSplit` as the CV strategy
- [ ] Log best parameters and best cross-validated score
- [ ] Refit the best model on the full training set; evaluate on the held-out test period

```python
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

param_dist = {
    "n_estimators":     [100, 300, 500],
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}

search = RandomizedSearchCV(
    XGBRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=30,
    cv=TimeSeriesSplit(n_splits=5),
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    random_state=42,
    verbose=1,
)
search.fit(X_train, y_train)
print("Best params:", search.best_params_)
print("Best CV MAE:", -search.best_score_)
```

---

## Step 4 — Model Comparison

> **Why:** No single algorithm dominates all tasks. A structured comparison surfaces the best approach for your data.

- [ ] Benchmark at least four algorithm families:
  - **Linear baseline**: Ridge Regression
  - **Tree ensemble**: Random Forest Regressor
  - **Gradient boosting**: XGBoost or LightGBM
  - **Linear time-aware**: Lasso with lagged features
- [ ] Use the same `TimeSeriesSplit` and feature set for every model
- [ ] Record MAE, RMSE, and directional accuracy (% correct up/down predictions) in a comparison table
- [ ] Select the top-1 or top-2 models for further tuning

```python
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pandas as pd

models = {
    "Ridge":         Ridge(),
    "RandomForest":  RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost":       XGBRegressor(n_estimators=200, random_state=42),
}

results = {}
for name, mdl in models.items():
    scores = cross_val_score(
        mdl, X_train, y_train,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_mean_absolute_error",
    )
    results[name] = {"MAE_mean": -scores.mean(), "MAE_std": scores.std()}

pd.DataFrame(results).T.sort_values("MAE_mean")
```

---

## Step 5 — Model Explainability

> **Why:** Understanding *why* a model makes predictions builds trust, catches data-leakage bugs, and guides further feature work.

### 5a — Permutation Importance (model-agnostic)
- [ ] Run `sklearn.inspection.permutation_importance` on the test set
- [ ] Plot the top-20 most important features; remove or investigate any surprises

```python
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

result = permutation_importance(
    best_model, X_test, y_test,
    n_repeats=10, random_state=42, n_jobs=-1
)
feat_imp = pd.Series(result.importances_mean, index=X_test.columns)
feat_imp.nlargest(20).sort_values().plot(kind="barh", figsize=(8, 8))
plt.title("Permutation Importance (test set)")
plt.tight_layout()
plt.savefig("reports/permutation_importance.png", dpi=150)
plt.show()
```

### 5b — SHAP Values (tree models)
- [ ] Install `shap`: `pip install shap`
- [ ] Compute SHAP values for the best tree model
- [ ] Generate a beeswarm summary plot and a waterfall plot for a single prediction

```python
import shap

explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Summary (global)
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("reports/shap_summary.png", dpi=150, bbox_inches="tight")
plt.show()

# Single prediction waterfall
shap.plots.waterfall(explainer(X_test)[0])
```

- [ ] Document any surprising feature drivers as potential bugs or new research leads

---

## Step 6 — Macroeconomic Data from FRED

> **Why:** Gold prices are influenced by interest rates, inflation, and the dollar. FRED data is free and comprehensive.

- [ ] Install `fredapi`: `pip install fredapi`
- [ ] Obtain a free API key at <https://fred.stlouisfed.org/docs/api/api_key.html>
- [ ] Store the key safely: use an environment variable `FRED_API_KEY` (never commit it to git)
- [ ] Download and merge the following series into your main DataFrame:

| Series ID | Description |
|-----------|-------------|
| `DGS10`   | 10-Year Treasury Yield |
| `CPIAUCSL`| CPI (inflation proxy) |
| `DEXUSEU` | USD/EUR Exchange Rate |
| `FEDFUNDS`| Federal Funds Rate |
| `DCOILWTICO` | WTI Crude Oil Price |

```python
# scripts/fetch_fred_data.py
import os
import pandas as pd
from fredapi import Fred

fred = Fred(api_key=os.environ["FRED_API_KEY"])

series_ids = {
    "treasury_10y": "DGS10",
    "cpi":          "CPIAUCSL",
    "usd_eur":      "DEXUSEU",
    "fed_funds":    "FEDFUNDS",
    "wti_oil":      "DCOILWTICO",
}

frames = {}
for name, sid in series_ids.items():
    s = fred.get_series(sid)
    frames[name] = s.rename(name)

fred_df = pd.concat(frames.values(), axis=1)
fred_df.index = pd.to_datetime(fred_df.index)
fred_df = fred_df.resample("B").last().ffill()   # align to business days
fred_df.to_csv("data/fred_macro_data.csv")
print("Saved FRED data:", fred_df.shape)
```

- [ ] Merge FRED data with `yahoo_market_data.csv` on the `Date` index
- [ ] Add lag features for macro variables (e.g. 1-month lag for CPI which is released monthly)
- [ ] Re-run feature importance to see which macro variables matter most

---

## Step 7 — Modular Pipeline & Productionization

> **Why:** Notebooks are great for exploration but are hard to schedule, test, or deploy. A modular codebase enables automation and reuse.

### 7a — Refactor into Functions / Classes
- [ ] Extract data-loading logic into `pipelines/data_loader.py`
- [ ] Extract feature-engineering logic into `pipelines/feature_engineer.py`
- [ ] Extract training/evaluation logic into `pipelines/trainer.py`
- [ ] Add a top-level `pipelines/gold_ml_pipeline.py` that chains all steps end-to-end

```
pipelines/
├── data_loader.py        # download / load Yahoo + FRED data
├── feature_engineer.py   # lags, rolling, RSI, ratios
├── trainer.py            # train, CV, evaluate, save model
└── gold_ml_pipeline.py   # orchestrates the full pipeline
```

### 7b — Model Serialization
- [ ] Save the best trained model with `joblib.dump(model, "models/gold_price_model.pkl")`
- [ ] Write a `scripts/predict.py` that loads the saved model and outputs a prediction for the latest available data

```python
import joblib, pandas as pd

model = joblib.load("models/gold_price_model.pkl")
latest = pd.read_csv("data/yahoo_market_data.csv", index_col="Date", parse_dates=True)
# ... apply same feature engineering ...
prediction = model.predict(latest.tail(1)[features])
print(f"Predicted next-day gold close: {prediction[0]:.2f}")
```

### 7c — Scheduled Retraining (optional)
- [ ] Create a GitHub Actions workflow (`.github/workflows/retrain.yml`) that runs `gold_ml_pipeline.py` on a schedule (e.g. weekly)
- [ ] Store updated model artifacts as GitHub Actions artifacts or in an S3 bucket

### 7d — Testing
- [ ] Add `pytest` unit tests for feature-engineering functions (no data leakage, correct shapes)
- [ ] Add a smoke test that runs the full pipeline on a small synthetic dataset

---

## Quick Reference: Priority Order

| Priority | Step | Expected Effort |
|----------|------|-----------------|
| 🔴 High  | 1. Time series CV | 1–2 hours |
| 🔴 High  | 2. Feature engineering | 2–4 hours |
| 🟠 Medium | 3. Hyperparameter tuning | 2–3 hours |
| 🟠 Medium | 4. Model comparison | 2–3 hours |
| 🟡 Medium | 5. Explainability (SHAP) | 1–2 hours |
| 🟡 Medium | 6. FRED macro data | 1–2 hours |
| 🟢 Lower | 7. Productionization | 4–8 hours |

---

## Suggested Directory Structure (target state)

```
BLANK/
├── data/
│   ├── yahoo_market_data.csv
│   └── fred_macro_data.csv
├── models/
│   └── gold_price_model.pkl
├── notebooks/
│   ├── 01_eda_and_baseline.ipynb
│   ├── 02_hyperparameter_tuning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_comparison.ipynb
│   └── 05_model_explainability.ipynb
├── pipelines/
│   ├── data_loader.py
│   ├── feature_engineer.py
│   ├── trainer.py
│   └── gold_ml_pipeline.py
├── reports/
│   ├── permutation_importance.png
│   └── shap_summary.png
├── scripts/
│   ├── download_market_data.py
│   ├── fetch_fred_data.py
│   └── predict.py
├── tests/
│   └── test_feature_engineer.py
├── NEXT_STEPS.md
├── README.md
└── requirements.txt
```

---

## Dependencies to Add

Add these to `requirements.txt` as you complete each step:

```
# Step 3-4 (already likely installed)
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0

# Step 5
shap>=0.44

# Step 6
fredapi>=0.5

# Step 7
joblib>=1.3
pytest>=8.0
```
