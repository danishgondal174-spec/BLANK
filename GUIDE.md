# Gold Price ML Model — Step-by-Step Guide

This guide walks you from raw market data to a research-grade machine learning workflow for gold price prediction. Every section includes actionable code you can paste directly into a Jupyter notebook.

---

## Checklist

- [ ] Step 1 — Define your target variable
- [ ] Step 2 — Choose your features
- [ ] Step 3 — Feature engineering
- [ ] Step 4 — Train/test split (time-series safe)
- [ ] Step 5 — Select a model
- [ ] Step 6 — Train the model
- [ ] Step 7 — Evaluate and visualise results
- [ ] Step 8 — Avoid lookahead bias / data leakage
- [ ] Step 9 — Next steps (tuning, cross-validation, backtesting)

---

## Prerequisites

Make sure you have run the data download script so that `data/yahoo_market_data.csv` exists.

```bash
pip install -r requirements.txt
```

Load the dataset at the top of every notebook section:

```python
import pandas as pd
import numpy as np

df = pd.read_csv("data/yahoo_market_data.csv", parse_dates=["Date"], index_col="Date")
df.sort_index(inplace=True)   # always sort chronologically
print(df.shape)
df.head()
```

---

## Step 1 — Define Your Target Variable

Decide **what you want to predict** before touching any features.

### Option A — Regression (predict tomorrow's price)

```python
# Tomorrow's closing gold price
df["target_price"] = df["gold_Close"].shift(-1)
```

### Option B — Classification (predict direction: up or down)

```python
# 1 if gold closes higher tomorrow, 0 otherwise
df["target_direction"] = (df["gold_Close"].shift(-1) > df["gold_Close"]).astype(int)
```

### Option C — Regression on returns (less scale-sensitive)

```python
# Tomorrow's percentage return
df["target_return"] = df["gold_Close"].pct_change().shift(-1)
```

> **Tip:** Start with classification (Option B). It is easier to evaluate, and
> accuracy/F1 metrics give clear intuition on whether the model adds value.

**Important — drop the last row after shifting**, because it will have a `NaN`
target (there is no "tomorrow" for the final date):

```python
df.dropna(subset=["target_direction"], inplace=True)
```

---

## Step 2 — Choose Your Features

Good features for gold price prediction fall into three groups.

### 2.1 Gold's own price history

```python
# Lagged closing prices
df["gold_lag1"] = df["gold_Close"].shift(1)
df["gold_lag5"] = df["gold_Close"].shift(5)

# Daily percentage return
df["gold_return_1d"] = df["gold_Close"].pct_change()

# 5-day cumulative return
df["gold_return_5d"] = df["gold_Close"].pct_change(5)
```

### 2.2 Macro / correlated assets

```python
df["sp500_return_1d"] = df["sp500_Close"].pct_change()
df["dxy_return_1d"]   = df["dxy_Close"].pct_change()   # US Dollar Index
```

### 2.3 Technical indicators (requires `ta` library)

```bash
pip install ta
```

```python
from ta.momentum import RSIIndicator
from ta.trend import MACD

# RSI (14-period)
df["gold_rsi14"] = RSIIndicator(close=df["gold_Close"], window=14).rsi()

# MACD line
macd = MACD(close=df["gold_Close"])
df["gold_macd"]        = macd.macd()
df["gold_macd_signal"] = macd.macd_signal()
```

### 2.4 Select your feature list

```python
features = [
    "gold_lag1", "gold_lag5",
    "gold_return_1d", "gold_return_5d",
    "sp500_return_1d", "dxy_return_1d",
    "gold_rsi14", "gold_macd", "gold_macd_signal",
]
target = "target_direction"   # or "target_price" / "target_return"
```

---

## Step 3 — Feature Engineering

### 3.1 Rolling moving averages

```python
df["gold_ma7"]  = df["gold_Close"].rolling(7).mean()
df["gold_ma21"] = df["gold_Close"].rolling(21).mean()
df["gold_ma50"] = df["gold_Close"].rolling(50).mean()

# Price relative to its own moving average (mean-reversion signal)
df["gold_ma7_ratio"]  = df["gold_Close"] / df["gold_ma7"]
df["gold_ma21_ratio"] = df["gold_Close"] / df["gold_ma21"]
```

### 3.2 Rolling volatility

```python
df["gold_vol7"]  = df["gold_return_1d"].rolling(7).std()
df["gold_vol21"] = df["gold_return_1d"].rolling(21).std()
```

### 3.3 Rolling correlation between gold and the S&P 500

```python
df["gold_sp500_corr21"] = (
    df["gold_return_1d"]
    .rolling(21)
    .corr(df["sp500_return_1d"])
)
```

### 3.4 Drop rows with NaN values created by rolling calculations

```python
df.dropna(inplace=True)
print(f"Dataset size after dropping NaNs: {df.shape}")
```

---

## Step 4 — Train/Test Split (Time-Series Safe)

**Never shuffle a time series.** Future data must not appear in training.

```python
train = df.loc[:"2021-12-31"]
test  = df.loc["2022-01-01":]

X_train = train[features]
y_train = train[target]

X_test  = test[features]
y_test  = test[target]

print(f"Train rows: {len(train)}, Test rows: {len(test)}")
```

> **Rule of thumb:** use ~80 % of the timeline for training and the most recent
> ~20 % for testing. Adjust the split date based on how much data you have.

---

## Step 5 — Select a Model

Start simple, then increase complexity only if needed.

| Level | Model | Use case |
|-------|-------|----------|
| Baseline | Majority-class / last-value carry | Sanity check |
| Linear | `LogisticRegression` / `LinearRegression` | Interpretable, fast |
| Tree-based | `RandomForestClassifier` / `GradientBoostingClassifier` | Non-linear patterns |
| Boosting | `XGBClassifier` / `LGBMClassifier` | State-of-the-art tabular |

---

## Step 6 — Train the Model

### 6.1 Baseline (always beat this first)

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
print("Baseline accuracy:", accuracy_score(y_test, baseline.predict(X_test)))
```

### 6.2 Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
])
lr_pipe.fit(X_train, y_train)
lr_preds = lr_pipe.predict(X_test)
print("Logistic Regression accuracy:", accuracy_score(y_test, lr_preds))
```

### 6.3 Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("Random Forest accuracy:", accuracy_score(y_test, rf_preds))
```

### 6.4 Gradient Boosting (XGBoost)

```bash
pip install xgboost
```

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
print("XGBoost accuracy:", accuracy_score(y_test, xgb_preds))
```

---

## Step 7 — Evaluate and Visualise Results

### 7.1 Classification metrics

```python
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print(classification_report(y_test, xgb_preds, target_names=["Down", "Up"]))

cm = confusion_matrix(y_test, xgb_preds)
ConfusionMatrixDisplay(cm, display_labels=["Down", "Up"]).plot()
plt.title("Confusion Matrix — XGBoost")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
```

### 7.2 Regression metrics (if you chose Option A or C)

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae  = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds, squared=False)
print(f"MAE: {mae:.4f}   RMSE: {rmse:.4f}")
```

### 7.3 Feature importance plot (Random Forest or XGBoost)

```python
import pandas as pd
import matplotlib.pyplot as plt

feat_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
feat_imp.plot(kind="bar", figsize=(10, 4), title="Feature Importances — Random Forest")
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=150)
plt.show()
```

### 7.4 Cumulative returns backtest plot

```python
test_copy = test.copy()
test_copy["pred_direction"] = rf_preds              # 1=long, 0=no position
test_copy["gold_daily_ret"] = test_copy["gold_Close"].pct_change()

# Strategy: hold gold only when model predicts "Up"
test_copy["strategy_ret"]   = test_copy["pred_direction"].shift(1) * test_copy["gold_daily_ret"]

cum_gold     = (1 + test_copy["gold_daily_ret"]).cumprod()
cum_strategy = (1 + test_copy["strategy_ret"]).cumprod()

plt.figure(figsize=(12, 5))
plt.plot(cum_gold,     label="Buy & Hold Gold")
plt.plot(cum_strategy, label="ML Strategy")
plt.title("Cumulative Returns — Test Period")
plt.legend()
plt.tight_layout()
plt.savefig("cumulative_returns.png", dpi=150)
plt.show()
```

---

## Step 8 — Avoid Lookahead Bias and Data Leakage

These are the most common mistakes in financial ML. Follow each rule carefully.

| Rule | Why it matters | How to enforce it |
|------|---------------|-------------------|
| Shift target by −1 | The label must be in the **future** relative to features | Always use `.shift(-1)` when creating targets |
| No future data in rolling stats | `.rolling(n).mean()` uses only past rows by default — confirm this | Ensure rolling calculations on test data do not include any rows from the future; compute them on the full sorted DataFrame before splitting |
| Fit scalers on train only | `StandardScaler` fitted on test leaks test statistics | Use `sklearn.Pipeline` or call `.fit()` only on `X_train` |
| No random shuffle of rows | Shuffling mixes past and future | Use chronological splits; never pass `shuffle=True` to a time-series splitter |
| Recompute indicators carefully | Some libraries look ahead internally | Verify indicator formulas, especially for multi-period ones |

---

## Step 9 — Next Steps

### 9.1 Time-series cross-validation

Use `TimeSeriesSplit` instead of random k-fold:

```python
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

tscv   = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(rf, df[features], df[target], cv=tscv, scoring="accuracy")
print("CV accuracy scores:", scores)
print(f"Mean ± Std: {scores.mean():.4f} ± {scores.std():.4f}")
```

### 9.2 Hyperparameter tuning

```python
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth":    [3, 5, 7, None],
    "min_samples_leaf": [1, 5, 10],
}
search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    n_iter=20,
    cv=TimeSeriesSplit(n_splits=5),
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
)
search.fit(df[features], df[target])
print("Best params:", search.best_params_)
print(f"Best CV accuracy: {search.best_score_:.4f}")
```

### 9.3 Backtesting with realistic assumptions

For a research-grade backtest, account for:
- **Transaction costs** (e.g. 0.05 % per trade)
- **Slippage** (fill at next open, not same-day close)
- **Position sizing** (equal weight vs. Kelly criterion)

Example with transaction costs:

```python
COST_PER_TRADE = 0.0005   # 5 basis points

position_change = test_copy["pred_direction"].diff().abs()
test_copy["net_strategy_ret"] = (
    test_copy["strategy_ret"] - position_change * COST_PER_TRADE
)

cum_net = (1 + test_copy["net_strategy_ret"]).cumprod()
```

### 9.4 Ideas for further improvement

- Add **sentiment features** (news, Google Trends for "gold")
- Add **macro variables** (inflation expectations, Fed Funds rate)
- Try **LSTM / GRU** recurrent networks for sequence modelling
- Ensemble multiple models (blending, stacking)
- Use **SHAP values** for model interpretability

---

## Quick Reference

```
data/
  yahoo_market_data.csv     ← raw market data

notebooks/
  01_eda.ipynb              ← starter EDA notebook
  02_feature_engineering.ipynb
  03_model_training.ipynb

scripts/
  download_market_data.py   ← data download pipeline
```

---

*Happy modelling! If you run into issues, check that all feature columns exist in your DataFrame before fitting the model, and that no `NaN` values remain.*
