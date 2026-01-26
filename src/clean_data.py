"""
Data cleaning module for Kaggle House Prices dataset.

This module provides functions to clean and preprocess the dataset, 
handling missing values, outliers, and data quality issues.
"""

import pandas as pd
import numpy as np


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Strategy:
    - Numerical features: Fill with median
    - Categorical features: Fill with mode or 'None' for meaningful missingness
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    # Numerical columns - fill with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # Categorical columns - fill with mode or 'None'
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            # For some columns, NA might mean "None" (e.g., no garage, no basement)
            if col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                      'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                      'PoolQC', 'Fence', 'MiscFeature', 'Alley', 'FireplaceQu']:
                df[col].fillna('None', inplace=True)
            else:
                # Fill with mode
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df[col].fillna(mode_value[0], inplace=True)
    
    return df


def remove_outliers(df: pd.DataFrame, target_col: str = 'SalePrice', threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers from the dataset using z-score method.
    
    Args:
        df: Input DataFrame
        target_col: Target column name (only relevant for training data)
        threshold: Z-score threshold for outlier detection
        
    Returns:
        DataFrame with outliers removed
    """
    df = df.copy()
    
    # Only remove outliers if target column exists (training data)
    if target_col in df.columns:
        z_scores = np.abs((df[target_col] - df[target_col].mean()) / df[target_col].std())
        df = df[z_scores < threshold]
        print(f"Removed {len(z_scores) - len(df)} outliers based on {target_col}")
    
    return df


def clean_data(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and outliers.
    
    Args:
        df: Input DataFrame
        is_training: Whether this is training data (for outlier removal)
        
    Returns:
        Cleaned DataFrame
    """
    print(f"Cleaning data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Handle missing values
    df = handle_missing_values(df)
    print(f"After handling missing values: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Remove outliers (only for training data)
    if is_training and 'SalePrice' in df.columns:
        df = remove_outliers(df, 'SalePrice')
        print(f"After removing outliers: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


def get_data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a data quality report showing missing values and data types.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with data quality metrics
    """
    report = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percent': (df.isnull().sum() / len(df) * 100).values,
        'Data_Type': df.dtypes.values,
        'Unique_Values': [df[col].nunique() for col in df.columns]
    })
    
    return report.sort_values('Missing_Percent', ascending=False)


if __name__ == "__main__":
    # Example usage
    from load_data import load_train_data
    
    try:
        # Load data
        train_df = load_train_data()
        
        # Get data quality report
        print("\nData Quality Report (before cleaning):")
        quality_report = get_data_quality_report(train_df)
        print(quality_report.head(10))
        
        # Clean data
        cleaned_df = clean_data(train_df, is_training=True)
        
        print("\nData Quality Report (after cleaning):")
        quality_report_after = get_data_quality_report(cleaned_df)
        print(quality_report_after.head(10))
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
