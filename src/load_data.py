"""
Data loading module for Kaggle House Prices dataset.

This module provides functions to load the raw House Prices dataset from CSV files.
"""

import pandas as pd
import os
from pathlib import Path


def get_data_path(filename: str) -> Path:
    """
    Get the full path to a data file.
    
    Args:
        filename: Name of the data file (e.g., 'train.csv', 'test.csv')
        
    Returns:
        Path object to the data file
    """
    # Get the project root directory (2 levels up from this file)
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'raw' / filename
    return data_path


def load_train_data(filepath: str = None) -> pd.DataFrame:
    """
    Load the training dataset.
    
    Args:
        filepath: Optional custom path to the training CSV file.
                 If None, uses default path: data/raw/train.csv
                 
    Returns:
        DataFrame containing the training data
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
    """
    if filepath is None:
        filepath = get_data_path('train.csv')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Training data not found at {filepath}. "
            "Please download the Kaggle House Prices dataset and place train.csv in data/raw/"
        )
    
    df = pd.read_csv(filepath)
    print(f"Loaded training data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_test_data(filepath: str = None) -> pd.DataFrame:
    """
    Load the test dataset.
    
    Args:
        filepath: Optional custom path to the test CSV file.
                 If None, uses default path: data/raw/test.csv
                 
    Returns:
        DataFrame containing the test data
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
    """
    if filepath is None:
        filepath = get_data_path('test.csv')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Test data not found at {filepath}. "
            "Please download the Kaggle House Prices dataset and place test.csv in data/raw/"
        )
    
    df = pd.read_csv(filepath)
    print(f"Loaded test data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_data(train_path: str = None, test_path: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both training and test datasets.
    
    Args:
        train_path: Optional custom path to the training CSV file
        test_path: Optional custom path to the test CSV file
        
    Returns:
        Tuple of (train_df, test_df)
    """
    train_df = load_train_data(train_path)
    test_df = load_test_data(test_path)
    return train_df, test_df


if __name__ == "__main__":
    # Example usage
    try:
        train_df = load_train_data()
        print("\nTraining data preview:")
        print(train_df.head())
        print("\nTraining data info:")
        print(train_df.info())
    except FileNotFoundError as e:
        print(f"Error: {e}")
