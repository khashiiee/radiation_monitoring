# notebooks/analyze_radiation.py
# notebooks/analyze_radiation.py

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent  # Updated this line
sys.path.append(str(project_root))

# Import our visualization module
from src.visualization import RadiationTimeVisualizer  # This should now work


# # Add project root to path
# project_root = Path.cwd().parent
# sys.path.append(str(project_root))

# # Import our visualization module
# from src.visualization import RadiationTimeVisualizer

def load_data(data_dir):
    """Load all required data files"""
    # Load static sensor locations
    static_locations = pd.read_csv(data_dir / 'StaticSensorLocations.csv')
    
    # Load static sensor readings
    static_data = pd.read_csv(
        data_dir / 'StaticSensorReadings.csv',
        parse_dates=['Timestamp']
    )
    
    # Load mobile sensor readings
    mobile_data = pd.read_csv(
        data_dir / 'MobileSensorReadings.csv',
        parse_dates=['Timestamp']
    )
    
    # Merge static data with locations
    static_data_with_loc = pd.merge(
        static_data,
        static_locations,
        on='Sensor-id'
    )
    
    return static_data_with_loc, mobile_data

def main():
    # Set up paths
    project_root = Path.cwd().parent
    data_dir = project_root / 'data' / 'raw'
    
    # Load data
    print("Loading data...")
    static_data, mobile_data = load_data(data_dir)
    print("Data loaded successfully!")
    
    # Create visualizer
    visualizer = RadiationTimeVisualizer(static_data, mobile_data)
    
    # Create visualization
    print("Creating visualization...")
    fig = visualizer.plot_time_series(
        start_time=pd.Timestamp('2020-04-06'),
        end_time=pd.Timestamp('2020-04-10')
    )
    
    # Add anomaly highlighting
    fig = visualizer.add_anomaly_highlighting(fig, threshold=3.0)
    
    # Save the figure
    output_path = project_root / 'notebooks' / 'radiation_time_series.html'
    fig.write_html(str(output_path))
    print(f"Visualization saved to: {output_path}")
    
    # Show the figure in browser
    fig.show()

if __name__ == "__main__":
    main()