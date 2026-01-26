"""
Model training module for Kaggle House Prices dataset.

This module provides functions to train and evaluate a baseline regression model
for predicting house prices.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
from pathlib import Path


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into training and validation sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    return X_train, X_val, y_train, y_val


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Train a basic linear regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained LinearRegression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression model trained")
    return model


def train_ridge_regression(X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0) -> Ridge:
    """
    Train a Ridge regression model (L2 regularization).
    
    Args:
        X_train: Training features
        y_train: Training target
        alpha: Regularization strength
        
    Returns:
        Trained Ridge model
    """
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    print(f"Ridge Regression model trained (alpha={alpha})")
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, 
                       n_estimators: int = 100, random_state: int = 42) -> RandomForestRegressor:
    """
    Train a Random Forest regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees in the forest
        random_state: Random seed for reproducibility
        
    Returns:
        Trained RandomForestRegressor model
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    print(f"Random Forest model trained ({n_estimators} trees)")
    return model


def evaluate_model(model, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """
    Evaluate model performance on validation set.
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation target
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    # Root Mean Squared Logarithmic Error (common metric for this competition)
    rmsle = np.sqrt(mean_squared_error(np.log1p(y_val), np.log1p(y_pred)))
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'RMSLE': rmsle
    }
    
    print("\nModel Evaluation Metrics:")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSLE: {rmsle:.4f}")
    
    return metrics


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict:
    """
    Perform cross-validation on the model.
    
    Args:
        model: Model to cross-validate
        X: Feature matrix
        y: Target vector
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary with cross-validation scores
    """
    # Negative MSE for cross-validation (sklearn convention)
    cv_scores = cross_val_score(model, X, y, cv=cv, 
                                scoring='neg_mean_squared_error', n_jobs=-1)
    
    # Convert to RMSE
    cv_rmse = np.sqrt(-cv_scores)
    
    print(f"\n{cv}-Fold Cross-Validation Results:")
    print(f"Mean RMSE: ${cv_rmse.mean():,.2f}")
    print(f"Std RMSE: ${cv_rmse.std():,.2f}")
    
    return {
        'cv_rmse_mean': cv_rmse.mean(),
        'cv_rmse_std': cv_rmse.std(),
        'cv_scores': cv_rmse
    }


def save_model(model, filepath: str = None):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        filepath: Path to save the model (default: data/processed/model.pkl)
    """
    if filepath is None:
        project_root = Path(__file__).parent.parent
        filepath = project_root / 'data' / 'processed' / 'model.pkl'
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved to {filepath}")


def load_model(filepath: str = None):
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model
    """
    if filepath is None:
        project_root = Path(__file__).parent.parent
        filepath = project_root / 'data' / 'processed' / 'model.pkl'
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from {filepath}")
    return model


if __name__ == "__main__":
    # Example usage
    from load_data import load_train_data
    from clean_data import clean_data
    from feature_engineering import engineer_features, select_features
    
    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        train_df = load_train_data()
        train_df = clean_data(train_df, is_training=True)
        train_df = engineer_features(train_df, encode=True)
        X, y = select_features(train_df, 'SalePrice')
        
        # Split data
        X_train, X_val, y_train, y_val = split_data(X, y)
        
        # Train models
        print("\n" + "="*50)
        print("Training Linear Regression...")
        lr_model = train_linear_regression(X_train, y_train)
        lr_metrics = evaluate_model(lr_model, X_val, y_val)
        
        print("\n" + "="*50)
        print("Training Ridge Regression...")
        ridge_model = train_ridge_regression(X_train, y_train, alpha=10.0)
        ridge_metrics = evaluate_model(ridge_model, X_val, y_val)
        
        print("\n" + "="*50)
        print("Training Random Forest...")
        rf_model = train_random_forest(X_train, y_train, n_estimators=100)
        rf_metrics = evaluate_model(rf_model, X_val, y_val)
        
        # Cross-validation on best model
        print("\n" + "="*50)
        print("Performing cross-validation on Random Forest...")
        cv_results = cross_validate_model(rf_model, X, y, cv=5)
        
        # Save the best model
        save_model(rf_model)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
