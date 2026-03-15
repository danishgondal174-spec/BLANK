# Gold ML Project

A complete machine learning project template with modular design for training, evaluating, and deploying classification models.

## Features

- **Multiple Classifiers**: Support for Random Forest, Logistic Regression, SVM, KNN, Decision Tree, Gradient Boosting, and Naive Bayes
- **Built-in Datasets**: Easy access to Iris, Wine, and Breast Cancer datasets
- **Data Preprocessing**: Scaling, encoding, train-test splitting, and missing value handling
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, and confusion matrix
- **Visualization**: Automatic generation of confusion matrices, feature importance plots, and metrics comparisons
- **Model Persistence**: Save and load trained models with joblib

## Project Structure

```
gold-ml-project/
├── src/
│   ├── data/
│   │   ├── data_loader.py      # Data loading utilities
│   │   └── preprocessor.py     # Data preprocessing
│   ├── models/
│   │   ├── classifier.py       # Unified classifier interface
│   │   └── model_trainer.py    # Training and evaluation
│   └── utils/
│       └── visualization.py    # Plotting utilities
├── tests/
│   ├── test_data.py           # Data module tests
│   └── test_models.py         # Model module tests
├── data/                      # Data directory
├── models/                    # Saved models directory
├── plots/                     # Generated plots
├── train.py                   # Main training script
├── predict.py                 # Prediction script
└── requirements.txt           # Dependencies
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd gold-ml-project

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training a Model

```bash
# Train with default settings (Iris dataset, Random Forest)
python train.py

# Train with specific dataset and model
python train.py --dataset wine --model logistic_regression

# Train without saving the model
python train.py --dataset breast_cancer --model svm --no-save
```

### Making Predictions

```bash
# Predict using a trained model
python predict.py --model models/model_random_forest_20240101_120000.joblib --features 5.1 3.5 1.4 0.2
```

### Available Options

**Datasets:**
- `iris` - Iris flower classification (150 samples, 4 features, 3 classes)
- `wine` - Wine classification (178 samples, 13 features, 3 classes)
- `breast_cancer` - Breast cancer classification (569 samples, 30 features, 2 classes)

**Models:**
- `random_forest` - Random Forest Classifier
- `logistic_regression` - Logistic Regression
- `svm` - Support Vector Machine
- `knn` - K-Nearest Neighbors
- `decision_tree` - Decision Tree
- `gradient_boosting` - Gradient Boosting
- `naive_bayes` - Gaussian Naive Bayes

## Usage in Code

```python
from src.data import DataLoader, Preprocessor
from src.models import Classifier, ModelTrainer

# Load data
loader = DataLoader()
X, y = loader.load_builtin_dataset("iris")

# Preprocess
preprocessor = Preprocessor()
X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
X_train = preprocessor.scale_features(X_train)
X_test = preprocessor.scale_features(X_test, fit=False)

# Train
trainer = ModelTrainer()
model = trainer.train(X_train, y_train, model_type="random_forest")

# Evaluate
metrics = trainer.evaluate(X_test, y_test)
trainer.print_evaluation_report(metrics)

# Save
trainer.save_model("my_model.joblib")
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v
```

## Output

After training, the following will be generated:
- **Model file**: Saved in `models/` directory
- **Confusion Matrix**: `plots/confusion_matrix.png`
- **Metrics Comparison**: `plots/metrics_comparison.png`
- **Feature Importance**: `plots/feature_importance.png` (for applicable models)

## License

MIT License
