#!/usr/bin/env python3
"""
Prediction Script
Make predictions using a trained model
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import ModelTrainer


def predict(model_path: str, features: list):
    """
    Make predictions using a trained model
    
    Args:
        model_path: Path to the saved model
        features: List of feature values
    """
    trainer = ModelTrainer()
    model = trainer.load_model(model_path)
    
    # Convert to numpy array and reshape
    X = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(X)
    
    print(f"\nInput features: {features}")
    print(f"Predicted class: {prediction[0]}")
    
    # Try to get probabilities
    try:
        probabilities = model.predict_proba(X)
        print(f"Class probabilities: {probabilities[0]}")
    except NotImplementedError:
        pass
    
    return prediction[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions with trained model")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to the trained model file")
    parser.add_argument("--features", type=float, nargs='+', required=True,
                       help="Feature values for prediction")
    
    args = parser.parse_args()
    
    predict(args.model, args.features)
