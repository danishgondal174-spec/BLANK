"""
predict.py
----------
Load the saved model and generate a gold price prediction for the latest
available market data.

Usage
-----
    python scripts/predict.py [--yahoo-path data/yahoo_market_data.csv]
                               [--fred-path  data/fred_macro_data.csv]
                               [--model-path models/gold_price_model.pkl]
"""

from __future__ import annotations

import argparse
import sys
import os

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def predict_latest(
    yahoo_path: str | None = None,
    fred_path:  str | None = None,
    model_path: str = "models/gold_price_model.pkl",
) -> float:
    """
    Load the trained model and return its prediction for the most recent row.

    The function applies the same feature-engineering pipeline used during
    training so that the feature vector is consistent.

    Returns
    -------
    Predicted gold close price (regression) or label (classification).
    """
    from pipelines.data_loader import load_all_data
    from pipelines.feature_engineer import build_features
    from pipelines.trainer import load_model

    model = load_model(model_path)

    df = load_all_data(yahoo_path, fred_path)
    df_feat = build_features(df)

    feature_cols = [c for c in df_feat.columns if c != "target"]
    latest_row   = df_feat[feature_cols].iloc[[-1]]

    prediction = model.predict(latest_row)[0]

    latest_date = df_feat.index[-1].date()
    print(f"Latest data date : {latest_date}")
    print(f"Prediction       : {prediction:.4f}")
    return float(prediction)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a gold price prediction.")
    parser.add_argument("--yahoo-path", default=None)
    parser.add_argument("--fred-path",  default=None)
    parser.add_argument("--model-path", default="models/gold_price_model.pkl")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    predict_latest(
        yahoo_path=args.yahoo_path,
        fred_path=args.fred_path,
        model_path=args.model_path,
    )
