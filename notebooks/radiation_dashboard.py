import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
from pathlib import Path
from datetime import datetime
import json
import geopandas as gpd

class RadiationDashboard:
    def __init__(self, data_dir, output_dir):
        """
        Initialize the dashboard with data directory path
        
        Args:
            data_dir: Path to raw data directory
            output_dir: Path to save output files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.load_data()
        
    def load_data(self):
        """Load all necessary data files"""
        print("Loading data...")
        try:
            # Load shapefile
            shapefile_path = self.data_dir.parent / "StHimarkNeighborhoodShapefile" / "StHimark.shp"
            print(f"Loading shapefile from {shapefile_path}")
            self.gdf = gpd.read_file(shapefile_path)
            
            # Load static sensor locations
            print("Loading static sensor locations...")
            self.static_locations = pd.read_csv(
                self.data_dir / 'StaticSensorLocations.csv'
            )
            print("Static location columns:", self.static_locations.columns.tolist())
            
            # Load static sensor readings
            print("Loading static sensor readings...")
            self.static_data = pd.read_csv(
                self.data_dir / 'StaticSensorReadings.csv',
                parse_dates=['Timestamp']
            )
            print("Static readings columns:", self.static_data.columns.tolist())
            
            # Merge static data with locations
            print("Merging static data with locations...")
            self.static_data_with_loc = pd.merge(
                self.static_data,
                self.static_locations,
                on='Sensor-id'
            )
            
            print("Data loaded successfully!")
            
        except Exception as e:
            print(f"\nError loading data: {str(e)}")
            print("\nDebug information:")
            print(f"Current directory: {Path.cwd()}")
            print(f"Data directory: {self.data_dir}")
            print("Files in data directory:", list(self.data_dir.glob("*")))
            raise
    
    def create_coverage_points(self, lat, lon, num_points=20, radius=500):
        """Create points around a center location to simulate coverage"""
        points = []
        for angle in np.linspace(0, 2*np.pi, num_points):
            r = radius * np.sqrt(np.random.random())
            dlat = r * np.cos(angle) / 111000  # Convert meters to degrees
            dlon = r * np.sin(angle) / (111000 * np.cos(np.radians(lat)))
            points.append([lat + dlat, lon + dlon, 1])
        return points
    
    def create_map(self, timestamp):
        """Create map visualization for a specific timestamp"""
        # Calculate map center
        center_lat = self.static_locations['Lat'].mean()
        center_long = self.static_locations['Long'].mean()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_long],
            zoom_start=12,
            tiles='CartoDB positron'
        )
        
        # Add neighborhood boundaries
        folium.GeoJson(
            self.gdf,
            name='Neighborhoods',
            style_function=lambda x: {
                'fillColor': '#ffff99',
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.1
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['Nbrhood'],
                aliases=['Neighborhood']
            )
        ).add_to(m)
        
        # Add static sensors with coverage circles
        for _, row in self.static_locations.iterrows():
            # Add sensor marker
            folium.CircleMarker(
                location=[row['Lat'], row['Long']],
                radius=5,
                color='red',
                fill=True,
                popup=f"Static Sensor {row['Sensor-id']}",
                tooltip=f"Static Sensor {row['Sensor-id']}",
                name='Static Sensors'
            ).add_to(m)
            
            # Add coverage circle
            folium.Circle(
                location=[row['Lat'], row['Long']],
                radius=500,  # in meters
                color='red',
                fill=True,
                fillOpacity=0.1,
                name='Sensor Coverage'
            ).add_to(m)
        
        # Create coverage heatmap
        coverage_points = []
        for _, row in self.static_locations.iterrows():
            coverage_points.extend(
                self.create_coverage_points(row['Lat'], row['Long'])
            )
        
        # Add coverage heatmap
        HeatMap(
            coverage_points,
            name='Coverage Heatmap',
            radius=15,
            min_opacity=0.3,
            gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'},
            overlay=True,
            show=False
        ).add_to(m)
        
        # Add radiation heatmap
        data_at_time = self.static_data_with_loc[
            self.static_data_with_loc['Timestamp'].dt.floor('h') == timestamp
        ]
        heat_data = [[row['Lat'], row['Long'], row['Value']] 
                    for _, row in data_at_time.iterrows()]
        
        HeatMap(
            heat_data,
            name='Radiation Levels',
            radius=15,
            min_opacity=0.3,
            gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'},
            overlay=True
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl(
            collapsed=False,
            autoZIndex=True
        ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 180px;
                    border:2px solid grey; z-index:9999; background-color:white;
                    opacity:0.8;
                    padding: 10px;
                    font-size: 14px;">
            <p><strong>Sensor Types</strong></p>
            <p><i class="fa fa-circle" style="color:red"></i> Static Sensors</p>
            <p><strong>Coverage</strong></p>
            <p style="color:red;opacity:0.3;">⬤</p> Sensor Range (500m)
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save map to HTML in the output directory
        map_path = self.output_dir / 'temp_map.html'
        m.save(str(map_path))
        
        return 'temp_map.html'
    
    def create_time_series(self):
        """Create time series visualization"""
        # Calculate hourly averages for static sensors only
        static_hourly = (self.static_data_with_loc
                        .groupby([pd.Grouper(key='Timestamp', freq='h')])
                        ['Value'].mean()
                        .reset_index())
        
        fig = go.Figure()
        
        # Add static sensor average
        fig.add_trace(
            go.Scatter(
                x=static_hourly['Timestamp'],
                y=static_hourly['Value'],
                name='Static Sensors Avg',
                line=dict(color='blue', width=2)
            )
        )
        
        fig.update_layout(
            title='Average Radiation Levels Over Time',
            xaxis_title='Time',
            yaxis_title='Radiation Level (cpm)',
            hovermode='x unified'
        )
        
        return fig
    
    def create_statistics_plot(self):
        """Create statistics visualization"""
        # Calculate statistics for static sensors only
        static_stats = self.static_data['Value'].describe()
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(name='Static Sensors', 
                  x=['Mean', 'Median', 'Std Dev', 'Max'],
                  y=[static_stats['mean'], static_stats['50%'], 
                     static_stats['std'], static_stats['max']])
        ])
        
        fig.update_layout(
            title='Radiation Level Statistics',
            yaxis_title='Value (cpm)'
        )
        
        return fig
    
    def create_dashboard(self):
        """Create and save the complete dashboard"""
        # Get first timestamp for initial map
        initial_timestamp = self.static_data['Timestamp'].min().floor('h')
        
        # Create map first
        map_file = self.create_map(initial_timestamp)
        
        # Create HTML template with all visualizations
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Radiation Monitoring Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ margin: 0; padding: 20px; font-family: Arial, sans-serif; }}
                .dashboard-container {{ display: grid; gap: 20px; }}
                .dashboard-item {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .row {{ display: flex; gap: 20px; margin-bottom: 20px; }}
                .plot {{ width: 100%; height: 500px; }}
                .map-container {{ width: 100%; height: 600px; }}
                h1 {{ color: #333; }}
                .status {{ padding: 10px; background: #f0f0f0; border-radius: 4px; margin-bottom: 10px; }}
            </style>
        </head>
        <body>
            <h1>St. Himark Radiation Monitoring Dashboard</h1>
            
            <div class="dashboard-container">
                <div class="status">
                    <h3>Current Status</h3>
                    <p>Last Updated: <span id="timestamp">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span></p>
                    <p style="color: orange;">Note: Currently showing static sensor data only.</p>
                </div>
                
                <div class="row">
                    <div class="dashboard-item" style="flex: 2;">
                        <h2>Radiation Levels Over Time</h2>
                        <div id="timeSeries" class="plot"></div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="dashboard-item" style="flex: 1;">
                        <h2>Radiation Statistics</h2>
                        <div id="statistics" class="plot"></div>
                    </div>
                    <div class="dashboard-item" style="flex: 1;">
                        <h2>Radiation Map</h2>
                        <iframe id="map" class="map-container" src="{map_file}"></iframe>
                    </div>
                </div>
            </div>
            
            <script>
                // Add time series plot
                const timeSeriesData = {self.create_time_series().to_json()};
                Plotly.newPlot('timeSeries', timeSeriesData.data, timeSeriesData.layout);
                
                // Add statistics plot
                const statisticsData = {self.create_statistics_plot().to_json()};
                Plotly.newPlot('statistics', statisticsData.data, statisticsData.layout);
            </script>
        </body>
        </html>
        """
        
        # Save dashboard HTML to output directory
        dashboard_path = self.output_dir / 'radiation_dashboard.html'
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
            
        print(f"Dashboard created successfully! Open {dashboard_path} in a web browser to view.")
        return dashboard_path

def main():
    # Set up paths
    project_root = Path.cwd().parent
    data_dir = project_root / 'data' / 'raw'
    output_dir = project_root / 'notebooks' / 'output'
    
    # Create and generate dashboard
    dashboard = RadiationDashboard(data_dir, output_dir)
    dashboard_path = dashboard.create_dashboard()
    
    print("\nDashboard files created at:")
    print(f"- Main dashboard: {dashboard_path}")
    print(f"- Map file: {output_dir / 'temp_map.html'}")
    print("\nOpen the main dashboard file in your web browser to view the complete dashboard.")

if __name__ == "__main__":
    main()