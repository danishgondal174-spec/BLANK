"""
Tests for Data Module
"""

import pytest
import numpy as np
from src.data import DataLoader, Preprocessor


class TestDataLoader:
    """Tests for DataLoader class"""
    
    def test_load_iris_dataset(self):
        """Test loading the iris dataset"""
        loader = DataLoader()
        X, y = loader.load_builtin_dataset("iris")
        
        assert X.shape == (150, 4)
        assert y.shape == (150,)
        assert loader.feature_names is not None
        assert loader.target_names is not None
    
    def test_load_wine_dataset(self):
        """Test loading the wine dataset"""
        loader = DataLoader()
        X, y = loader.load_builtin_dataset("wine")
        
        assert X.shape[0] == 178
        assert len(np.unique(y)) == 3
    
    def test_load_invalid_dataset(self):
        """Test that invalid dataset name raises error"""
        loader = DataLoader()
        
        with pytest.raises(ValueError):
            loader.load_builtin_dataset("invalid_dataset")
    
    def test_get_dataframe(self):
        """Test converting data to DataFrame"""
        loader = DataLoader()
        loader.load_builtin_dataset("iris")
        df = loader.get_dataframe()
        
        assert 'target' in df.columns
        assert len(df) == 150


class TestPreprocessor:
    """Tests for Preprocessor class"""
    
    def test_split_data(self):
        """Test data splitting"""
        preprocessor = Preprocessor()
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 3, 100)
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2)
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
    
    def test_scale_features_standard(self):
        """Test standard scaling"""
        preprocessor = Preprocessor()
        X = np.random.rand(100, 4) * 100
        
        X_scaled = preprocessor.scale_features(X, method="standard")
        
        # Check that mean is approximately 0 and std is approximately 1
        assert np.abs(X_scaled.mean()) < 0.1
        assert np.abs(X_scaled.std() - 1) < 0.1
    
    def test_scale_features_minmax(self):
        """Test min-max scaling"""
        preprocessor = Preprocessor()
        X = np.random.rand(100, 4) * 100
        
        X_scaled = preprocessor.scale_features(X, method="minmax")
        
        # Check that values are between 0 and 1
        assert X_scaled.min() >= 0
        assert X_scaled.max() <= 1
    
    def test_encode_decode_labels(self):
        """Test label encoding and decoding"""
        preprocessor = Preprocessor()
        y = np.array(['cat', 'dog', 'cat', 'bird', 'dog'])
        
        y_encoded = preprocessor.encode_labels(y)
        y_decoded = preprocessor.decode_labels(y_encoded)
        
        assert np.array_equal(y, y_decoded)
    
    def test_handle_missing_values(self):
        """Test handling missing values"""
        preprocessor = Preprocessor()
        X = np.array([[1, 2], [np.nan, 4], [5, np.nan]])
        
        X_filled = preprocessor.handle_missing_values(X, strategy="mean")
        
        assert not np.isnan(X_filled).any()
