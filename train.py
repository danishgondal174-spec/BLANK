#!/usr/bin/env python3
"""
Main Training Script
Complete ML pipeline for training and evaluating models
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import DataLoader, Preprocessor
from src.models import Classifier, ModelTrainer
from src.utils import Visualizer


def main(dataset_name: str = "iris", model_type: str = "random_forest",
         test_size: float = 0.2, save_model: bool = True):
    """
    Main training pipeline
    
    Args:
        dataset_name: Name of the dataset to use
        model_type: Type of classifier
        test_size: Test set proportion
        save_model: Whether to save the trained model
    """
    print("\n" + "="*60)
    print("GOLD ML PROJECT - Model Training Pipeline")
    print("="*60)
    
    # 1. Load Data
    print(f"\n[1/5] Loading {dataset_name} dataset...")
    data_loader = DataLoader()
    X, y = data_loader.load_builtin_dataset(dataset_name)
    print(f"      Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"      Target classes: {list(data_loader.target_names)}")
    
    # 2. Preprocess Data
    print("\n[2/5] Preprocessing data...")
    preprocessor = Preprocessor()
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        X, y, test_size=test_size
    )
    X_train_scaled = preprocessor.scale_features(X_train, method="standard")
    X_test_scaled = preprocessor.scale_features(X_test, fit=False)
    print(f"      Training set: {X_train.shape[0]} samples")
    print(f"      Test set: {X_test.shape[0]} samples")
    
    # 3. Train Model
    print(f"\n[3/5] Training {model_type} model...")
    trainer = ModelTrainer(model_dir="models")
    model = trainer.train(X_train_scaled, y_train, model_type=model_type)
    
    # 4. Evaluate Model
    print("\n[4/5] Evaluating model...")
    metrics = trainer.evaluate(X_test_scaled, y_test)
    trainer.print_evaluation_report(metrics)
    
    # 5. Visualizations and Save
    print("\n[5/5] Creating visualizations...")
    visualizer = Visualizer(save_dir="plots")
    
    # Plot confusion matrix
    visualizer.plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names=list(data_loader.target_names),
        save_path="plots/confusion_matrix.png"
    )
    
    # Plot metrics comparison
    visualizer.plot_metrics_comparison(
        metrics,
        save_path="plots/metrics_comparison.png"
    )
    
    # Plot feature importance if available
    importance = model.get_feature_importance()
    if importance is not None:
        visualizer.plot_feature_importance(
            importance,
            feature_names=list(data_loader.feature_names),
            save_path="plots/feature_importance.png"
        )
    
    print("      Plots saved to 'plots/' directory")
    
    # Save model
    if save_model:
        model_path = trainer.save_model()
        print(f"      Model saved to: {model_path}")
    
    print("\n" + "="*60)
    print("Training pipeline completed successfully!")
    print("="*60 + "\n")
    
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML Model")
    parser.add_argument("--dataset", type=str, default="iris",
                       choices=DataLoader.AVAILABLE_DATASETS,
                       help="Dataset to use for training")
    parser.add_argument("--model", type=str, default="random_forest",
                       choices=Classifier.AVAILABLE_MODELS,
                       help="Type of classifier to use")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Proportion of data for testing")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save the trained model")
    
    args = parser.parse_args()
    
    main(
        dataset_name=args.dataset,
        model_type=args.model,
        test_size=args.test_size,
        save_model=not args.no_save
    )
