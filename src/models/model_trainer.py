"""
Model Trainer Module
Handles model training, evaluation, and saving
"""

import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, Any, Optional

from .classifier import Classifier


class ModelTrainer:
    """
    Handles the complete ML training pipeline
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the trainer
        
        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = model_dir
        self.model = None
        self.training_history = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              model_type: str = "random_forest", **model_params) -> Classifier:
        """
        Train a model
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type: Type of classifier to use
            **model_params: Additional model parameters
            
        Returns:
            Trained classifier
        """
        print(f"Training {model_type} model...")
        
        self.model = Classifier(model_type=model_type, **model_params)
        self.model.fit(X_train, y_train)
        
        # Store training info
        self.training_history = {
            "model_type": model_type,
            "n_samples": len(X_train),
            "n_features": X_train.shape[1],
            "trained_at": datetime.now().isoformat()
        }
        
        print("Training completed!")
        return self.model
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 average: str = "weighted") -> Dict[str, Any]:
        """
        Evaluate the trained model
        
        Args:
            X_test: Test features
            y_test: Test labels
            average: Averaging method for multi-class metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model trained. Call train() first.")
        
        y_pred = self.model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average=average),
            "recall": recall_score(y_test, y_pred, average=average),
            "f1_score": f1_score(y_test, y_pred, average=average),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred)
        }
        
        return metrics
    
    def print_evaluation_report(self, metrics: Dict[str, Any]) -> None:
        """
        Print a formatted evaluation report
        
        Args:
            metrics: Dictionary of metrics from evaluate()
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION REPORT")
        print("="*50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("\nClassification Report:")
        print(metrics['classification_report'])
        print("="*50)
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """
        Save the trained model to disk
        
        Args:
            filename: Name of the file (auto-generated if not provided)
            
        Returns:
            Path to the saved model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_{self.training_history.get('model_type', 'unknown')}_{timestamp}.joblib"
        
        filepath = os.path.join(self.model_dir, filename)
        
        # Save model and training history
        save_data = {
            "model": self.model,
            "training_history": self.training_history
        }
        joblib.dump(save_data, filepath)
        
        print(f"Model saved to: {filepath}")
        return filepath
    
    def load_model(self, filepath: str) -> Classifier:
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded classifier
        """
        save_data = joblib.load(filepath)
        self.model = save_data["model"]
        self.training_history = save_data.get("training_history", {})
        
        print(f"Model loaded from: {filepath}")
        return self.model
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                       model_type: str = "random_forest",
                       cv: int = 5, **model_params) -> Dict[str, Any]:
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Labels
            model_type: Type of classifier
            cv: Number of cross-validation folds
            **model_params: Additional model parameters
            
        Returns:
            Cross-validation results
        """
        from sklearn.model_selection import cross_val_score
        
        model = Classifier(model_type=model_type, **model_params)
        
        scores = cross_val_score(model.model, X, y, cv=cv, scoring='accuracy')
        
        return {
            "cv_scores": scores,
            "mean_score": scores.mean(),
            "std_score": scores.std()
        }
