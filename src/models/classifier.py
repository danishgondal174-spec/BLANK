"""
Classifier Module
Contains various classification model implementations
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from typing import Optional, Dict, Any


class Classifier:
    """
    Unified classifier interface supporting multiple algorithms
    """
    
    AVAILABLE_MODELS = [
        "random_forest",
        "logistic_regression", 
        "svm",
        "knn",
        "decision_tree",
        "gradient_boosting",
        "naive_bayes"
    ]
    
    def __init__(self, model_type: str = "random_forest", **kwargs):
        """
        Initialize the classifier
        
        Args:
            model_type: Type of classifier to use
            **kwargs: Additional parameters for the model
        """
        self.model_type = model_type
        self.model = self._create_model(model_type, **kwargs)
        self.is_fitted = False
    
    def _create_model(self, model_type: str, **kwargs) -> Any:
        """Create the underlying sklearn model"""
        default_params = {
            "random_forest": {"n_estimators": 100, "random_state": 42},
            "logistic_regression": {"max_iter": 1000, "random_state": 42},
            "svm": {"kernel": "rbf", "random_state": 42},
            "knn": {"n_neighbors": 5},
            "decision_tree": {"random_state": 42},
            "gradient_boosting": {"n_estimators": 100, "random_state": 42},
            "naive_bayes": {}
        }
        
        model_classes = {
            "random_forest": RandomForestClassifier,
            "logistic_regression": LogisticRegression,
            "svm": SVC,
            "knn": KNeighborsClassifier,
            "decision_tree": DecisionTreeClassifier,
            "gradient_boosting": GradientBoostingClassifier,
            "naive_bayes": GaussianNB
        }
        
        if model_type not in model_classes:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available models: {self.AVAILABLE_MODELS}")
        
        # Merge default params with user-provided params
        params = {**default_params.get(model_type, {}), **kwargs}
        return model_classes[model_type](**params)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Classifier":
        """
        Train the classifier
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            self
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Features to predict
            
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError(f"{self.model_type} does not support predict_proba")
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return self.model.get_params()
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importances if available
        
        Returns:
            Feature importance array or None
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_).mean(axis=0)
        return None
