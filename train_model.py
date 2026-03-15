"""
train_model.py
==============
Builds and evaluates a machine-learning model that predicts the **next
trading day's Gold futures closing price** using the merged market dataset
produced by download_data.py.

Pipeline overview
-----------------
1. Load   data/market_data.csv
2. Engineer features (lags, moving averages, daily returns, RSI,
   Bollinger Bands, and cross-asset closing prices)
3. Create target: gold_close shifted one day forward (next-day price)
4. Split chronologically into train / test (no data leakage)
5. Train a Random Forest Regressor wrapped in an sklearn Pipeline
6. Evaluate on the test set (MAE, RMSE, MAPE, R²)
7. Save the trained model, metrics JSON, and two diagnostic charts

Output files
------------
    data/models/gold_price_model.joblib  – trained sklearn Pipeline
    data/models/metrics.json            – evaluation metrics
    data/models/actual_vs_predicted.png – actual vs predicted plot
    data/models/feature_importance.png  – top-20 feature importances

Usage
-----
    # Step 1 – download the market data
    python download_data.py

    # Step 2 – train the model
    python train_model.py
"""

import json
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")  # non-interactive backend – works on servers and CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_FILE = os.path.join("data", "market_data.csv")
MODELS_DIR = os.path.join("data", "models")

TARGET_COL = "gold_close"   # we predict *next day's* gold close
TEST_RATIO = 0.20           # last 20 % of rows form the test set
RANDOM_SEED = 42

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothed RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all engineered features to the DataFrame in-place.

    Features added
    --------------
    Gold technical indicators (moving averages, Bollinger Bands, RSI, returns):
        gold_ma5, gold_ma10, gold_ma20, gold_ma50
        gold_bb_upper20, gold_bb_lower20, gold_bb_width20
        gold_rsi14
        gold_ret_1d, gold_ret_5d, gold_ret_10d, gold_ret_20d
        gold_vol_5d   (rolling 5-day return volatility)
        gold_high_low_range

    Lag features for gold close (1-5 days):
        gold_lag_1, gold_lag_2, gold_lag_3, gold_lag_4, gold_lag_5

    Daily returns for cross-asset series:
        sp500_ret_1d, nasdaq_ret_1d, dxy_ret_1d / uup_ret_1d,
        crude_oil_ret_1d, us10y_ret_1d, vix_ret_1d
    """
    df = df.copy()
    gc = df[TARGET_COL]

    # --- Gold moving averages ---
    for w in (5, 10, 20, 50):
        df[f"gold_ma{w}"] = gc.rolling(w).mean()
        df[f"gold_ma{w}_ratio"] = gc / df[f"gold_ma{w}"]  # price relative to MA

    # --- Bollinger Bands (20-day) ---
    ma20 = gc.rolling(20).mean()
    std20 = gc.rolling(20).std()
    df["gold_bb_upper20"] = ma20 + 2 * std20
    df["gold_bb_lower20"] = ma20 - 2 * std20
    df["gold_bb_width20"] = (df["gold_bb_upper20"] - df["gold_bb_lower20"]) / ma20
    df["gold_bb_pct20"] = (gc - df["gold_bb_lower20"]) / (
        df["gold_bb_upper20"] - df["gold_bb_lower20"] + 1e-9
    )

    # --- RSI ---
    df["gold_rsi14"] = compute_rsi(gc, 14)

    # --- Gold returns ---
    for lag in (1, 5, 10, 20):
        df[f"gold_ret_{lag}d"] = gc.pct_change(lag)

    # --- Rolling volatility of daily returns ---
    df["gold_vol_5d"] = gc.pct_change().rolling(5).std()
    df["gold_vol_20d"] = gc.pct_change().rolling(20).std()

    # --- High-Low range (if columns exist) ---
    if "gold_high" in df.columns and "gold_low" in df.columns:
        df["gold_high_low_range"] = (df["gold_high"] - df["gold_low"]) / gc

    # --- Lag features for gold close ---
    for lag in range(1, 6):
        df[f"gold_lag_{lag}"] = gc.shift(lag)

    # --- Cross-asset daily returns ---
    cross_asset_cols = [
        "sp500_close", "nasdaq_close", "dxy_close", "uup_close",
        "crude_oil_close", "us10y_close", "vix_close",
    ]
    for col in cross_asset_cols:
        if col in df.columns:
            df[f"{col.replace('_close', '')}_ret_1d"] = df[col].pct_change()
            df[f"{col.replace('_close', '')}_ret_5d"] = df[col].pct_change(5)

    return df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """Return an sklearn Pipeline: StandardScaler → RandomForestRegressor."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(**RF_PARAMS)),
        ]
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # MAPE – guard against zero actuals
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE_%": round(mape, 4), "R2": round(r2, 4)}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict,
    out_path: str,
) -> None:
    """Save a chart of actual vs predicted gold price on the test set."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, y_true, label="Actual Gold Close", color="goldenrod", linewidth=1.5)
    ax.plot(dates, y_pred, label="Predicted Gold Close", color="steelblue", linewidth=1.5, linestyle="--")
    ax.set_title(
        f"Gold Close – Actual vs Predicted (test set)\n"
        f"MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  "
        f"MAPE={metrics['MAPE_%']:.2f}%  R²={metrics['R2']:.4f}",
        fontsize=11,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Chart saved: {out_path}")


def plot_feature_importance(
    feature_names: list,
    importances: np.ndarray,
    out_path: str,
    top_n: int = 20,
) -> None:
    """Save a horizontal bar chart of the top-N most important features."""
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), top_vals[::-1], align="center", color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances – Random Forest")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Chart saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Gold Price – ML Model Training Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if not os.path.exists(DATA_FILE):
        print(
            f"\n✗  Data file not found: {DATA_FILE}\n"
            "   Please run 'python download_data.py' first to generate it."
        )
        sys.exit(1)

    print(f"\n[1/5] Loading data from {DATA_FILE} ...", end=" ", flush=True)
    raw = pd.read_csv(DATA_FILE, index_col="date", parse_dates=True)
    print(f"✓  {len(raw)} rows × {len(raw.columns)} columns")
    print(f"   Date range: {raw.index.min().date()} → {raw.index.max().date()}")

    if TARGET_COL not in raw.columns:
        print(f"\n✗  Target column '{TARGET_COL}' not found in the dataset.")
        print(f"   Available columns: {list(raw.columns)}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    print("\n[2/5] Engineering features ...", end=" ", flush=True)
    df = add_features(raw)

    # Create the target: next-day gold close
    df["target"] = df[TARGET_COL].shift(-1)

    # Drop rows with NaN in target or features
    df = df.dropna(subset=["target"])
    df = df.dropna(axis=0)

    # Select feature columns: everything except the raw OHLCV and the target
    exclude = {"target"}
    # Keep all engineered + raw cols as potential features; let the model decide
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].values
    y = df["target"].values
    dates = df.index

    print(f"✓  {len(df)} usable rows, {len(feature_cols)} features")

    # ------------------------------------------------------------------
    # 3. Train / test split (chronological)
    # ------------------------------------------------------------------
    print("\n[3/5] Splitting train / test ...", end=" ", flush=True)
    split_idx = int(len(df) * (1 - TEST_RATIO))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates[split_idx:]

    print(
        f"✓  train={len(X_train)} rows  "
        f"({dates[0].date()} → {dates[split_idx - 1].date()}), "
        f"test={len(X_test)} rows  "
        f"({dates[split_idx].date()} → {dates[-1].date()})"
    )

    # ------------------------------------------------------------------
    # 4. Train model
    # ------------------------------------------------------------------
    print(f"\n[4/5] Training Random Forest ({RF_PARAMS['n_estimators']} trees) ...", end=" ", flush=True)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("✓  Done")

    # ------------------------------------------------------------------
    # 5. Evaluate and save
    # ------------------------------------------------------------------
    print("\n[5/5] Evaluating and saving outputs ...")

    y_pred = pipeline.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    print(f"\n  Test-set metrics:")
    for k, v in metrics.items():
        print(f"    {k:<10} {v}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save model
    model_path = os.path.join(MODELS_DIR, "gold_price_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"\n  Model saved : {model_path}")

    # Save metrics
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"  Metrics saved: {metrics_path}")

    # Save feature importances (RF is inside the pipeline's second step)
    rf_model = pipeline.named_steps["rf"]
    plot_actual_vs_predicted(
        dates_test, y_test, y_pred, metrics,
        os.path.join(MODELS_DIR, "actual_vs_predicted.png"),
    )
    plot_feature_importance(
        feature_cols,
        rf_model.feature_importances_,
        os.path.join(MODELS_DIR, "feature_importance.png"),
    )

    print("\n✅ All done!")
    print(f"   Model  → {model_path}")
    print(f"   Charts → {MODELS_DIR}/")


if __name__ == "__main__":
    main()
