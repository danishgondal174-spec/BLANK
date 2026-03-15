"""
Data Loader Module
Handles loading data from various sources
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from typing import Tuple, Optional


class DataLoader:
    """Class to load datasets for ML training"""
    
    AVAILABLE_DATASETS = ["iris", "wine", "breast_cancer"]
    
    def __init__(self):
        self.data = None
        self.target = None
        self.feature_names = None
        self.target_names = None
    
    def load_builtin_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a built-in sklearn dataset
        
        Args:
            dataset_name: Name of the dataset ('iris', 'wine', 'breast_cancer')
            
        Returns:
            Tuple of (features, target)
        """
        dataset_loaders = {
            "iris": load_iris,
            "wine": load_wine,
            "breast_cancer": load_breast_cancer
        }
        
        if dataset_name not in dataset_loaders:
            raise ValueError(f"Dataset '{dataset_name}' not found. "
                           f"Available datasets: {self.AVAILABLE_DATASETS}")
        
        dataset = dataset_loaders[dataset_name]()
        self.data = dataset.data
        self.target = dataset.target
        self.feature_names = dataset.feature_names
        self.target_names = dataset.target_names
        
        return self.data, self.target
    
    def load_csv(self, filepath: str, target_column: str, 
                 feature_columns: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from a CSV file
        
        Args:
            filepath: Path to the CSV file
            target_column: Name of the target column
            feature_columns: List of feature column names (if None, uses all except target)
            
        Returns:
            Tuple of (features, target)
        """
        df = pd.read_csv(filepath)
        
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        self.data = df[feature_columns].values
        self.target = df[target_column].values
        self.feature_names = feature_columns
        
        return self.data, self.target
    
    def get_dataframe(self) -> pd.DataFrame:
        """Convert loaded data to pandas DataFrame"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_builtin_dataset or load_csv first.")
        
        df = pd.DataFrame(self.data, columns=self.feature_names)
        df['target'] = self.target
        return df
