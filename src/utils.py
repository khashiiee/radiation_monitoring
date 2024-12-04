import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict, List

def calculate_background_radiation(data: pd.DataFrame, 
                                 window_size: str = '1H') -> float:
    """
    Calculate background radiation level using historical data.
    
    Args:
        data: DataFrame with radiation readings
        window_size: Time window for calculating background radiation
        
    Returns:
        float: Estimated background radiation level
    """
    # Calculate the median radiation level as background
    background = data['Value'].rolling(window_size).median()
    return background

def calculate_statistics(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate basic statistics for radiation readings.
    
    Args:
        data: DataFrame with radiation readings
        
    Returns:
        Dictionary containing various statistical measures
    """
    stats = {
        'mean': data['Value'].mean(),
        'median': data['Value'].median(),
        'std': data['Value'].std(),
        'min': data['Value'].min(),
        'max': data['Value'].max(),
        'q25': data['Value'].quantile(0.25),
        'q75': data['Value'].quantile(0.75)
    }
    return stats

def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate if coordinates are within St. Himark boundaries.
    
    Args:
        lat: Latitude value
        lon: Longitude value
        
    Returns:
        bool: True if coordinates are valid
    """
    # St. Himark boundaries (from provided data)
    LAT_BOUNDS = (0.04, 0.19)  # Approximate values from the data
    LON_BOUNDS = (-119.96, -119.74)  # Approximate values from the data
    
    return (LAT_BOUNDS[0] <= lat <= LAT_BOUNDS[1] and 
            LON_BOUNDS[0] <= lon <= LON_BOUNDS[1])

def detect_anomalies(data: pd.DataFrame, 
                    threshold: float = 3.0) -> pd.Series:
    """
    Detect anomalous radiation readings using Z-score method.
    
    Args:
        data: DataFrame with radiation readings
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        Boolean series indicating anomalous readings
    """
    z_scores = np.abs((data['Value'] - data['Value'].mean()) / data['Value'].std())
    return z_scores > threshold

def interpolate_missing_values(data: pd.DataFrame, 
                             method: str = 'linear') -> pd.DataFrame:
    """
    Interpolate missing values in radiation readings.
    
    Args:
        data: DataFrame with radiation readings
        method: Interpolation method ('linear', 'cubic', etc.)
        
    Returns:
        DataFrame with interpolated values
    """
    return data.interpolate(method=method)

def calculate_exposure_risk(radiation_level: float, 
                          duration: float) -> Tuple[str, float]:
    """
    Calculate exposure risk based on radiation level and duration.
    
    Args:
        radiation_level: Radiation level in cpm
        duration: Exposure duration in hours
        
    Returns:
        Tuple of (risk_level, cumulative_exposure)
    """
    # Convert CPM to mSv/h (approximate conversion)
    # Note: This is a simplified conversion and should be adjusted based on actual sensor calibration
    msv_per_hour = radiation_level * 0.001
    
    cumulative_exposure = msv_per_hour * duration
    
    if cumulative_exposure < 0.1:
        risk_level = 'LOW'
    elif cumulative_exposure < 1:
        risk_level = 'MODERATE'
    else:
        risk_level = 'HIGH'
        
    return risk_level, cumulative_exposure

def get_sensor_reliability(sensor_data: pd.DataFrame) -> float:
    """
    Calculate sensor reliability score based on data consistency.
    
    Args:
        sensor_data: DataFrame with readings from a single sensor
        
    Returns:
        float: Reliability score between 0 and 1
    """
    # Calculate reliability based on:
    # 1. Missing data percentage
    # 2. Variance in readings
    # 3. Number of anomalies
    
    total_readings = len(sensor_data)
    missing_data = sensor_data['Value'].isna().sum()
    missing_ratio = 1 - (missing_data / total_readings)
    
    # Check for anomalies
    anomalies = detect_anomalies(sensor_data)
    anomaly_ratio = 1 - (anomalies.sum() / total_readings)
    
    # Calculate variance score (lower variance = higher reliability)
    variance = sensor_data['Value'].var()
    variance_score = 1 / (1 + np.log1p(variance))
    
    # Combine scores (you can adjust weights as needed)
    reliability_score = (missing_ratio * 0.4 + 
                        anomaly_ratio * 0.4 + 
                        variance_score * 0.2)
    
    return np.clip(reliability_score, 0, 1)

def calculate_contamination_spread(
    data: pd.DataFrame,
    center_point: Tuple[float, float],
    time_window: str = '1H'
) -> Dict[str, Union[float, List[Tuple[float, float]]]]:
    """
    Calculate the spread of contamination from a central point.
    
    Args:
        data: DataFrame with radiation readings
        center_point: (latitude, longitude) of contamination source
        time_window: Time window for analysis
        
    Returns:
        Dictionary with spread metrics
    """
    # Calculate distances from center point
    data['distance'] = data.apply(
        lambda row: np.sqrt(
            (row['Lat'] - center_point[0])**2 + 
            (row['Long'] - center_point[1])**2
        ), 
        axis=1
    )
    
    # Group by time window and calculate spread metrics
    spread_data = data.groupby(pd.Grouper(key='Timestamp', freq=time_window)).agg({
        'distance': ['max', 'mean'],
        'Value': ['mean', 'max']
    })
    
    return {
        'max_spread_distance': spread_data['distance']['max'].max(),
        'avg_spread_distance': spread_data['distance']['mean'].mean(),
        'peak_radiation': spread_data['Value']['max'].max(),
        'avg_radiation': spread_data['Value']['mean'].mean()
    }