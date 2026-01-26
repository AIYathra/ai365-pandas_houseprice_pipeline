"""
Feature engineering module for Kaggle House Prices dataset.

This module provides functions to create new features and transform existing ones
to improve model performance.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def create_total_area_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features representing total areas of the house.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with new area features
    """
    df = df.copy()
    
    # Total square footage
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
    # Total bathrooms
    df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    
    # Total porch area
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    
    return df


def create_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features related to the age of the house.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with age-related features
    """
    df = df.copy()
    
    # House age
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    
    # Years since remodel
    df['YearsSinceRemodel'] = df['YrSold'] - df['YearRemodAdd']
    
    # Was the house recently built?
    df['IsNew'] = (df['YrSold'] == df['YearBuilt']).astype(int)
    
    return df


def create_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features combining quality ratings.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with quality features
    """
    df = df.copy()
    
    # Overall quality score
    df['OverallScore'] = df['OverallQual'] * df['OverallCond']
    
    # External quality score
    if 'ExterQual' in df.columns and 'ExterCond' in df.columns:
        quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
        df['ExterQualNum'] = df['ExterQual'].map(quality_map).fillna(0)
        df['ExterCondNum'] = df['ExterCond'].map(quality_map).fillna(0)
        df['ExterScore'] = df['ExterQualNum'] * df['ExterCondNum']
    
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features using label encoding.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with encoded categorical features
    """
    df = df.copy()
    
    # Get categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Apply label encoding
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    return df


def engineer_features(df: pd.DataFrame, encode: bool = True) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.
    
    Args:
        df: Input DataFrame
        encode: Whether to encode categorical features
        
    Returns:
        DataFrame with engineered features
    """
    print(f"Engineering features: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Create area features
    df = create_total_area_features(df)
    
    # Create age features
    df = create_age_features(df)
    
    # Create quality features
    df = create_quality_features(df)
    
    # Encode categorical features
    if encode:
        df = encode_categorical_features(df)
    
    print(f"After feature engineering: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


def select_features(df: pd.DataFrame, target_col: str = 'SalePrice') -> tuple[pd.DataFrame, pd.Series]:
    """
    Select features and target for modeling.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        
    Returns:
        Tuple of (features_df, target_series)
    """
    # Remove ID column if present
    feature_cols = [col for col in df.columns if col not in ['Id', target_col]]
    
    X = df[feature_cols]
    y = df[target_col] if target_col in df.columns else None
    
    return X, y


if __name__ == "__main__":
    # Example usage
    from load_data import load_train_data
    from clean_data import clean_data
    
    try:
        # Load and clean data
        train_df = load_train_data()
        train_df = clean_data(train_df, is_training=True)
        
        # Engineer features
        train_df = engineer_features(train_df, encode=True)
        
        # Select features
        X, y = select_features(train_df, 'SalePrice')
        
        print("\nFeature matrix shape:", X.shape)
        print("Target shape:", y.shape)
        print("\nSample features:")
        print(X.head())
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
