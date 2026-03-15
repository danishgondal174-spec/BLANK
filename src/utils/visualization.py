"""
Visualization Module
Handles plotting and visualization of ML results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
import os


class Visualizer:
    """Class for creating ML visualizations"""
    
    def __init__(self, save_dir: str = "plots"):
        """
        Initialize visualizer
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def plot_confusion_matrix(self, confusion_mat: np.ndarray,
                              class_names: Optional[List[str]] = None,
                              title: str = "Confusion Matrix",
                              save_path: Optional[str] = None) -> None:
        """
        Plot a confusion matrix
        
        Args:
            confusion_mat: Confusion matrix array
            class_names: Names of the classes
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(confusion_mat))]
        
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, importance: np.ndarray,
                                feature_names: List[str],
                                title: str = "Feature Importance",
                                top_n: int = 10,
                                save_path: Optional[str] = None) -> None:
        """
        Plot feature importances
        
        Args:
            importance: Feature importance values
            feature_names: Names of features
            title: Plot title
            top_n: Number of top features to show
            save_path: Path to save the plot
        """
        # Sort by importance
        indices = np.argsort(importance)[-top_n:]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importance[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_learning_curve(self, train_scores: np.ndarray,
                           test_scores: np.ndarray,
                           train_sizes: np.ndarray,
                           title: str = "Learning Curve",
                           save_path: Optional[str] = None) -> None:
        """
        Plot learning curve
        
        Args:
            train_scores: Training scores
            test_scores: Test scores
            train_sizes: Training set sizes
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='r',
                label='Training score')
        plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', color='g',
                label='Cross-validation score')
        
        plt.fill_between(train_sizes, 
                        train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1), 
                        alpha=0.1, color='r')
        plt.fill_between(train_sizes,
                        test_scores.mean(axis=1) - test_scores.std(axis=1),
                        test_scores.mean(axis=1) + test_scores.std(axis=1),
                        alpha=0.1, color='g')
        
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.title(title)
        plt.legend(loc='best')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_metrics_comparison(self, metrics_dict: dict,
                                title: str = "Model Metrics",
                                save_path: Optional[str] = None) -> None:
        """
        Plot comparison of different metrics
        
        Args:
            metrics_dict: Dictionary with metric names as keys and values
            title: Plot title
            save_path: Path to save the plot
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [metrics_dict.get(m, 0) for m in metrics]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(metrics, values, color=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'])
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.ylim(0, 1.1)
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
