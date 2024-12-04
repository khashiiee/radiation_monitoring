# Version of the package
__version__ = '0.1.0'

# Import main classes for easier access
from .data_processing import RadiationDataProcessor
from .visualization import RadiationVisualizer
from .utils import (
    calculate_background_radiation,
    calculate_statistics,
    validate_coordinates,
    detect_anomalies
)

# Define what should be available when using "from src import *"
__all__ = [
    'RadiationDataProcessor',
    'RadiationVisualizer',
    'calculate_background_radiation',
    'calculate_statistics',
    'validate_coordinates',
    'detect_anomalies'
]