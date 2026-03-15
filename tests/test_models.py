"""
Tests for Models Module
"""

import pytest
import numpy as np
from sklearn.datasets import load_iris
from src.models import Classifier, ModelTrainer


class TestClassifier:
    """Tests for Classifier class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        iris = load_iris()
        return iris.data, iris.target
    
    def test_random_forest_classifier(self, sample_data):
        """Test Random Forest classifier"""
        X, y = sample_data
        
        classifier = Classifier(model_type="random_forest")
        classifier.fit(X, y)
        
        assert classifier.is_fitted
        predictions = classifier.predict(X)
        assert len(predictions) == len(y)
    
    def test_logistic_regression_classifier(self, sample_data):
        """Test Logistic Regression classifier"""
        X, y = sample_data
        
        classifier = Classifier(model_type="logistic_regression")
        classifier.fit(X, y)
        
        assert classifier.is_fitted
        accuracy = np.mean(classifier.predict(X) == y)
        assert accuracy > 0.8  # Should have reasonable accuracy
    
    def test_predict_before_fit(self, sample_data):
        """Test that predicting before fitting raises error"""
        X, y = sample_data
        
        classifier = Classifier(model_type="random_forest")
        
        with pytest.raises(ValueError):
            classifier.predict(X)
    
    def test_invalid_model_type(self):
        """Test that invalid model type raises error"""
        with pytest.raises(ValueError):
            Classifier(model_type="invalid_model")
    
    def test_feature_importance(self, sample_data):
        """Test getting feature importance"""
        X, y = sample_data
        
        classifier = Classifier(model_type="random_forest")
        classifier.fit(X, y)
        
        importance = classifier.get_feature_importance()
        assert importance is not None
        assert len(importance) == X.shape[1]
    
    def test_predict_proba(self, sample_data):
        """Test prediction probabilities"""
        X, y = sample_data
        
        classifier = Classifier(model_type="random_forest")
        classifier.fit(X, y)
        
        proba = classifier.predict_proba(X)
        assert proba.shape[0] == len(X)
        assert proba.shape[1] == len(np.unique(y))
        # Probabilities should sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestModelTrainer:
    """Tests for ModelTrainer class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        iris = load_iris()
        return iris.data, iris.target
    
    def test_train_model(self, sample_data):
        """Test model training"""
        X, y = sample_data
        
        trainer = ModelTrainer(model_dir="/tmp/test_models")
        model = trainer.train(X, y, model_type="random_forest")
        
        assert model is not None
        assert trainer.model is not None
    
    def test_evaluate_model(self, sample_data):
        """Test model evaluation"""
        X, y = sample_data
        
        trainer = ModelTrainer(model_dir="/tmp/test_models")
        trainer.train(X, y, model_type="random_forest")
        
        metrics = trainer.evaluate(X, y)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'confusion_matrix' in metrics
    
    def test_save_and_load_model(self, sample_data):
        """Test saving and loading model"""
        import os
        X, y = sample_data
        
        trainer = ModelTrainer(model_dir="/tmp/test_models")
        trainer.train(X, y, model_type="random_forest")
        
        model_path = trainer.save_model("test_model.joblib")
        assert os.path.exists(model_path)
        
        # Load in new trainer
        new_trainer = ModelTrainer()
        loaded_model = new_trainer.load_model(model_path)
        
        # Verify loaded model works
        predictions = loaded_model.predict(X)
        assert len(predictions) == len(y)
        
        # Cleanup
        os.remove(model_path)
    
    def test_cross_validate(self, sample_data):
        """Test cross-validation"""
        X, y = sample_data
        
        trainer = ModelTrainer(model_dir="/tmp/test_models")
        cv_results = trainer.cross_validate(X, y, model_type="random_forest", cv=5)
        
        assert 'cv_scores' in cv_results
        assert 'mean_score' in cv_results
        assert 'std_score' in cv_results
        assert len(cv_results['cv_scores']) == 5
