"""
Data Preprocessor Module
Handles data preprocessing and feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Union


class Preprocessor:
    """Class for data preprocessing operations"""
    
    def __init__(self):
        self.scaler = None
        self.label_encoder = None
        self.is_fitted = False
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, 
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets
        
        Args:
            X: Feature array
            y: Target array
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, 
                               random_state=random_state, stratify=y)
    
    def scale_features(self, X: np.ndarray, method: str = "standard", 
                       fit: bool = True) -> np.ndarray:
        """
        Scale features using specified method
        
        Args:
            X: Feature array
            method: Scaling method ('standard' or 'minmax')
            fit: Whether to fit the scaler (True for training, False for testing)
            
        Returns:
            Scaled feature array
        """
        if self.scaler is None:
            if method == "standard":
                self.scaler = StandardScaler()
            elif method == "minmax":
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def encode_labels(self, y: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Encode categorical labels to numeric values
        
        Args:
            y: Target array with categorical labels
            fit: Whether to fit the encoder
            
        Returns:
            Encoded target array
        """
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        
        if fit:
            return self.label_encoder.fit_transform(y)
        else:
            return self.label_encoder.transform(y)
    
    def decode_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Decode numeric labels back to original categories
        
        Args:
            y_encoded: Encoded target array
            
        Returns:
            Original categorical labels
        """
        if self.label_encoder is None:
            raise ValueError("Label encoder not fitted.")
        return self.label_encoder.inverse_transform(y_encoded)
    
    def handle_missing_values(self, X: np.ndarray, 
                              strategy: str = "mean") -> np.ndarray:
        """
        Handle missing values in the data
        
        Args:
            X: Feature array
            strategy: Strategy for handling missing values ('mean', 'median', 'zero')
            
        Returns:
            Array with missing values handled
        """
        X_df = pd.DataFrame(X)
        
        if strategy == "mean":
            return X_df.fillna(X_df.mean()).values
        elif strategy == "median":
            return X_df.fillna(X_df.median()).values
        elif strategy == "zero":
            return X_df.fillna(0).values
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
