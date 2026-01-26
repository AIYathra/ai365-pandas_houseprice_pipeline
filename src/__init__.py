"""
House Prices Regression Pipeline

A modular pipeline for the Kaggle House Prices dataset using Pandas, NumPy, and scikit-learn.
"""

from .load_data import load_train_data, load_test_data, load_data
from .clean_data import clean_data, handle_missing_values, get_data_quality_report
from .feature_engineering import engineer_features, select_features
from .train_model import (
    train_linear_regression,
    train_ridge_regression,
    train_random_forest,
    evaluate_model,
    save_model,
    load_model
)

__version__ = '1.0.0'

__all__ = [
    'load_train_data',
    'load_test_data', 
    'load_data',
    'clean_data',
    'handle_missing_values',
    'get_data_quality_report',
    'engineer_features',
    'select_features',
    'train_linear_regression',
    'train_ridge_regression',
    'train_random_forest',
    'evaluate_model',
    'save_model',
    'load_model',
]
