"""
Data loading and parsing utilities for Wind Farm Dataset.
Handles the special CSV structure with coordinates and time series data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
from datetime import datetime


def load_wind_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load wind farm dataset with special structure.
    
    Args:
        csv_path: Path to Wind Spatio-Temporal Dataset2.csv
        
    Returns:
        coords_df: DataFrame with coordinates (rows: header, lat, lon)
        timeseries_df: DataFrame with time series data
    """
    # Read first 3 rows for coordinates (rows 0-2: header, lat, lon)
    coords_df = pd.read_csv(csv_path, nrows=3, header=None)
    
    # Read time series data (skip first 4 rows: turbine names, lat, lon, empty row)
    # Row 4 has the actual column headers (Time, Turbine1_Speed, etc.)
    timeseries_df = pd.read_csv(csv_path, skiprows=4, low_memory=False)
    
    # Convert Time column to datetime
    timeseries_df['Time'] = pd.to_datetime(timeseries_df['Time'], format='%m/%d/%Y %H:%M')
    timeseries_df = timeseries_df.sort_values('Time').reset_index(drop=True)
    
    return coords_df, timeseries_df


def extract_node_metadata(coords_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract node metadata from coordinate rows.
    
    Args:
        coords_df: DataFrame with coordinates (3 rows: headers, lat, lon)
        
    Returns:
        metadata: DataFrame with columns [node_id, node_name, node_type, latitude, longitude]
    """
    # First row is headers, second is latitudes, third is longitudes
    headers = coords_df.iloc[0, 1:].values  # Skip first column
    latitudes = coords_df.iloc[1, 1:].values
    longitudes = coords_df.iloc[2, 1:].values
    
    # Create metadata dataframe
    metadata = pd.DataFrame({
        'node_name': headers,
        'latitude': pd.to_numeric(latitudes, errors='coerce'),
        'longitude': pd.to_numeric(longitudes, errors='coerce')
    })
    
    # Determine node type
    metadata['node_type'] = metadata['node_name'].apply(
        lambda x: 'mast' if 'Mast' in str(x) else 'turbine'
    )
    
    # Assign node IDs
    metadata['node_id'] = range(len(metadata))
    
    # Remove rows with NaN coordinates
    metadata = metadata.dropna(subset=['latitude', 'longitude'])
    
    # Reorder columns
    metadata = metadata[['node_id', 'node_name', 'node_type', 'latitude', 'longitude']]
    
    return metadata


def get_column_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify and group columns by type (Speed, Power, Direction).
    
    Args:
        df: Time series dataframe
        
    Returns:
        Dictionary with keys ['speed', 'power', 'direction'] and column name lists
    """
    speed_cols = [col for col in df.columns if 'Speed' in col and col != 'Time']
    power_cols = [col for col in df.columns if 'Power' in col]
    direction_cols = [col for col in df.columns if 'Direction' in col]
    
    # Separate turbine and mast columns
    turbine_speed = [col for col in speed_cols if 'Turbine' in col]
    turbine_power = [col for col in power_cols if 'Turbine' in col]
    mast_speed = [col for col in speed_cols if 'Mast' in col]
    mast_direction = direction_cols
    
    return {
        'turbine_speed': turbine_speed,
        'turbine_power': turbine_power,
        'mast_speed': mast_speed,
        'mast_direction': mast_direction,
        'all_speed': speed_cols,
        'all_power': power_cols,
        'all_direction': direction_cols
    }


def handle_missing_values(df: pd.DataFrame, 
                         power_cols: List[str],
                         method: str = 'forward_fill') -> pd.DataFrame:
    """
    Handle missing values in power columns.
    
    Args:
        df: Time series dataframe
        power_cols: List of power column names
        method: 'forward_fill', 'interpolate', or 'zero'
        
    Returns:
        DataFrame with handled missing values
    """
    df_clean = df.copy()
    
    if method == 'forward_fill':
        # Forward fill then backward fill for any remaining NaNs
        df_clean[power_cols] = df_clean[power_cols].ffill().bfill()
    elif method == 'interpolate':
        # Linear interpolation
        df_clean[power_cols] = df_clean[power_cols].interpolate(method='linear', limit_direction='both')
    elif method == 'zero':
        # Fill with zeros (assumes turbine is off)
        df_clean[power_cols] = df_clean[power_cols].fillna(0)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return df_clean


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from Time column.
    
    Args:
        df: DataFrame with 'Time' column
        
    Returns:
        DataFrame with added temporal features
    """
    df = df.copy()
    
    # Basic temporal features
    df['hour'] = df['Time'].dt.hour
    df['day_of_week'] = df['Time'].dt.dayofweek
    df['day_of_year'] = df['Time'].dt.dayofyear
    df['month'] = df['Time'].dt.month
    df['week_of_year'] = df['Time'].dt.isocalendar().week
    
    # Cyclical encoding for hour (24-hour cycle)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Cyclical encoding for day of week (7-day cycle)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Cyclical encoding for month (12-month cycle)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Binary feature: weekend or not
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    return df


def create_lagged_features(df: pd.DataFrame, 
                          columns: List[str],
                          lags: List[int] = [1, 6, 24]) -> pd.DataFrame:
    """
    Create lagged features for specified columns.
    
    Args:
        df: Time series dataframe
        columns: Column names to create lags for
        lags: List of lag values (in hours)
        
    Returns:
        DataFrame with added lagged features
    """
    df = df.copy()
    
    for col in columns:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    # Drop rows with NaN from lagging
    max_lag = max(lags)
    df = df.iloc[max_lag:].reset_index(drop=True)
    
    return df


def create_rolling_features(df: pd.DataFrame,
                           columns: List[str],
                           windows: List[int] = [6, 24]) -> pd.DataFrame:
    """
    Create rolling statistics features.
    
    Args:
        df: Time series dataframe
        columns: Column names to create rolling stats for
        windows: List of window sizes (in hours)
        
    Returns:
        DataFrame with added rolling features
    """
    df = df.copy()
    
    for col in columns:
        for window in windows:
            # Rolling mean
            df[f'{col}_rolling_mean_{window}h'] = df[col].rolling(window=window, min_periods=1).mean()
            # Rolling std
            df[f'{col}_rolling_std_{window}h'] = df[col].rolling(window=window, min_periods=1).std()
            # Rolling min
            df[f'{col}_rolling_min_{window}h'] = df[col].rolling(window=window, min_periods=1).min()
            # Rolling max
            df[f'{col}_rolling_max_{window}h'] = df[col].rolling(window=window, min_periods=1).max()
    
    return df


def encode_wind_direction(df: pd.DataFrame, direction_cols: List[str]) -> pd.DataFrame:
    """
    Encode wind direction as sin/cos for circular feature.
    
    Args:
        df: Time series dataframe
        direction_cols: List of direction column names
        
    Returns:
        DataFrame with sin/cos encoded directions
    """
    df = df.copy()
    
    for col in direction_cols:
        # Convert degrees to radians
        radians = np.deg2rad(df[col])
        
        # Create sin/cos features
        df[f'{col}_sin'] = np.sin(radians)
        df[f'{col}_cos'] = np.cos(radians)
    
    return df


def split_data_chronological(df: pd.DataFrame,
                            train_ratio: float = 0.6,
                            val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into train/val/test sets.
    
    Args:
        df: Time series dataframe
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        
    Returns:
        train_df, val_df, test_df
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Example usage
    data_path = Path("data/raw/Wind Spatio-Temporal Dataset2.csv")
    
    # Load data
    coords_df, timeseries_df = load_wind_dataset(data_path)
    print(f"Loaded {len(timeseries_df)} timestamps")
    
    # Extract metadata
    metadata = extract_node_metadata(coords_df)
    print(f"Extracted metadata for {len(metadata)} nodes")
    
    # Get column groups
    col_groups = get_column_groups(timeseries_df)
    print(f"Found {len(col_groups['turbine_power'])} turbines")
    
    # Handle missing values
    timeseries_df = handle_missing_values(timeseries_df, col_groups['turbine_power'])
    
    # Create features
    timeseries_df = create_temporal_features(timeseries_df)
    timeseries_df = encode_wind_direction(timeseries_df, col_groups['mast_direction'])
    
    print("Data loading and preprocessing complete!")
