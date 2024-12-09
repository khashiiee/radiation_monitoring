import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster, TimestampedGeoJson
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
import json
import geopandas as gpd
from typing import Dict, List, Optional, Tuple, Union
import logging

class RadiationDashboard:
    def __init__(self, data_dir: Union[str, Path], output_dir: Union[str, Path]):
        """
        Initialize the radiation monitoring dashboard.
        
        Args:
            data_dir: Path to raw data directory
            output_dir: Path to save output files
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize paths
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data holders
        self.has_mobile_data = False
        self.gdf = None
        self.static_locations = None
        self.static_data = None
        self.static_data_with_loc = None
        self.mobile_data = None
        
        # Load data
        self.load_data()
        
        # Constants
        self.SENSOR_COVERAGE_RADIUS = 500  # meters
        self.DEFAULT_ZOOM = 12
        self.MAP_STYLE = 'CartoDB positron'
        
    def load_data(self) -> None:
        """Load and preprocess all necessary data files."""
        try:
            # Load shapefile
            shapefile_path = self.data_dir.parent / "StHimarkNeighborhoodShapefile" / "StHimark.shp"
            self.logger.info(f"Loading shapefile from {shapefile_path}")
            self.gdf = gpd.read_file(shapefile_path)
            
            # Load and process static sensor data
            self._load_static_data()
            
            # Load and process mobile sensor data
            self._load_mobile_data()
            
            # Validate loaded data
            self._validate_data()
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _load_static_data(self) -> None:
        """Load and process static sensor data."""
        self.logger.info("Loading static sensor data...")
        
        # Load locations
        self.static_locations = pd.read_csv(self.data_dir / 'StaticSensorLocations.csv')
        
        # Load readings
        self.static_data = pd.read_csv(
            self.data_dir / 'StaticSensorReadings.csv',
            parse_dates=['Timestamp']
        )
        
        # Merge locations with readings
        self.static_data_with_loc = pd.merge(
            self.static_data,
            self.static_locations,
            on='Sensor-id'
        )
        
        self.logger.info("Static data loaded successfully")

    def _load_mobile_data(self) -> None:
        """Load and process mobile sensor data."""
        self.logger.info("Loading mobile sensor data...")
        
        try:
            # Check if file is Git LFS pointer
            mobile_temp = pd.read_csv(self.data_dir / 'MobileSensorReadings.csv', nrows=5)
            
            if 'version https://git-lfs.github.com/spec/v1' in str(mobile_temp.columns):
                self.logger.warning("Mobile data file is a Git LFS pointer")
                self.has_mobile_data = False
                self._create_empty_mobile_data()
            else:
                self._process_mobile_data()
                
        except Exception as e:
            self.logger.warning(f"Could not load mobile sensor data: {str(e)}")
            self.has_mobile_data = False
            self._create_empty_mobile_data()

    def _create_empty_mobile_data(self) -> None:
        """Create empty mobile data structure."""
        self.mobile_data = pd.DataFrame(
            columns=['Timestamp', 'Sensor-id', 'Value', 'Lat', 'Long']
        )

    def _validate_data(self) -> None:
        """Validate loaded data for consistency and completeness."""
        # Check static data
        assert not self.static_data_with_loc.empty, "Static sensor data is empty"
        assert all(col in self.static_data_with_loc.columns 
                  for col in ['Timestamp', 'Sensor-id', 'Value', 'Lat', 'Long']), \
            "Missing required columns in static data"
        
        # Validate coordinates
        self._validate_coordinates(self.static_data_with_loc)
        if self.has_mobile_data:
            self._validate_coordinates(self.mobile_data)

    def _validate_coordinates(self, df: pd.DataFrame) -> None:
        """Validate coordinate values in dataframe."""
        assert df['Lat'].between(0, 90).all(), "Invalid latitude values"
        assert df['Long'].between(-180, 180).all(), "Invalid longitude values"

    def _add_neighborhood_layer(self, m: folium.Map) -> None:
        """Add neighborhood boundaries to the map."""
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

    def _add_static_sensor_layer(self, m: folium.Map, timestamp: datetime) -> None:
        """Add static sensor markers and radiation heatmap."""
        # Add static sensor markers
        static_layer = folium.FeatureGroup(name='Static Sensors')
        for _, row in self.static_locations.iterrows():
            # Add sensor marker
            folium.CircleMarker(
                location=[row['Lat'], row['Long']],
                radius=5,
                color='red',
                fill=True,
                popup=f"Static Sensor {row['Sensor-id']}",
                tooltip=f"Static Sensor {row['Sensor-id']}"
            ).add_to(static_layer)
            
            # Add coverage circle
            folium.Circle(
                location=[row['Lat'], row['Long']],
                radius=self.SENSOR_COVERAGE_RADIUS,
                color='red',
                fill=True,
                fillOpacity=0.1
            ).add_to(static_layer)
        
        static_layer.add_to(m)

        # Add radiation heatmap
        static_data_at_time = self.static_data_with_loc[
            self.static_data_with_loc['Timestamp'].dt.floor('h') == timestamp
        ]
        
        if not static_data_at_time.empty:
            HeatMap(
                data=[[row['Lat'], row['Long'], row['Value']] 
                      for _, row in static_data_at_time.iterrows()],
                name='Static Radiation Levels',
                radius=15,
                min_opacity=0.3,
                gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'},
                show=True
            ).add_to(m)

    def _add_mobile_sensor_layer(self, m: folium.Map, timestamp: datetime) -> None:
        """Add mobile sensor clusters and radiation heatmap."""
        if not self.has_mobile_data:
            return

        mobile_data_at_time = self.mobile_data[
            self.mobile_data['Timestamp'].dt.floor('h') == timestamp
        ]
        
        if mobile_data_at_time.empty:
            return

        # Add mobile sensor clusters
        marker_cluster = MarkerCluster(name='Mobile Sensors')
        for _, row in mobile_data_at_time.iterrows():
            folium.CircleMarker(
                location=[row['Lat'], row['Long']],
                radius=3,
                color='blue',
                fill=True,
                popup=f"Mobile Sensor {row['Sensor-id']}<br>Value: {row['Value']}",
                tooltip=f"Mobile Sensor {row['Sensor-id']}"
            ).add_to(marker_cluster)
        marker_cluster.add_to(m)

        # Add mobile radiation heatmap
        HeatMap(
            data=[[row['Lat'], row['Long'], row['Value']] 
                  for _, row in mobile_data_at_time.iterrows()],
            name='Mobile Radiation Levels',
            radius=15,
            min_opacity=0.3,
            gradient={0.4: 'purple', 0.65: 'pink', 0.8: 'orange', 1: 'red'},
            show=True
        ).add_to(m)

    def _add_map_controls(self, m: folium.Map, timestamp: datetime) -> None:
        """Add layer controls and timestamp information to map."""
        # Add layer control
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Add timestamp information
        timestamp_html = f'''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 200px;
                        background-color: white;
                        z-index:9999; font-size:14px;
                        padding: 10px;
                        border: 2px solid grey;">
                <b>Time:</b> {timestamp.strftime('%Y-%m-%d %H:%M')}
            </div>
        '''
        m.get_root().html.add_child(folium.Element(timestamp_html))

    def create_time_series(self, show_static: bool = True, show_mobile: bool = True) -> go.Figure:
        """Create interactive time series plot."""
        fig = go.Figure()

        if show_static:
            static_hourly = (self.static_data_with_loc
                           .groupby([pd.Grouper(key='Timestamp', freq='h')])
                           ['Value'].mean()
                           .reset_index())
            fig.add_trace(
                go.Scatter(
                    x=static_hourly['Timestamp'],
                    y=static_hourly['Value'],
                    name='Static Sensors',
                    line=dict(color='red', width=2)
                )
            )

        if show_mobile and self.has_mobile_data:
            mobile_hourly = (self.mobile_data
                           .groupby([pd.Grouper(key='Timestamp', freq='h')])
                           ['Value'].mean()
                           .reset_index())
            fig.add_trace(
                go.Scatter(
                    x=mobile_hourly['Timestamp'],
                    y=mobile_hourly['Value'],
                    name='Mobile Sensors',
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

    def create_statistics_plot(self) -> go.Figure:
        """Create statistical comparison plot."""
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=('Distribution', 'Key Metrics'))
        
        # Add static sensor distribution
        static_values = self.static_data_with_loc['Value']
        fig.add_trace(
            go.Histogram(x=static_values, name='Static Sensors', 
                        nbinsx=30, opacity=0.7),
            row=1, col=1
        )

        # Add mobile sensor distribution if available
        if self.has_mobile_data:
            mobile_values = self.mobile_data['Value']
            fig.add_trace(
                go.Histogram(x=mobile_values, name='Mobile Sensors', 
                            nbinsx=30, opacity=0.7),
                row=1, col=1
            )

        # Add key metrics
        metrics = ['mean', 'median', 'std', 'max']
        static_stats = [static_values.mean(), static_values.median(), 
                       static_values.std(), static_values.max()]
        
        fig.add_trace(
            go.Bar(x=metrics, y=static_stats, name='Static Sensors'),
            row=1, col=2
        )

        if self.has_mobile_data:
            mobile_stats = [mobile_values.mean(), mobile_values.median(), 
                          mobile_values.std(), mobile_values.max()]
            fig.add_trace(
                go.Bar(x=metrics, y=mobile_stats, name='Mobile Sensors'),
                row=1, col=2
            )

        fig.update_layout(
            height=400,
            title_text="Radiation Level Statistics",
            showlegend=True
        )
        
        return fig

    def create_map(self, timestamp: datetime, show_static: bool = True, 
                  show_mobile: bool = True) -> str:
        """Create interactive map with specified layers and filters."""
        # Create base map
        center_lat = self.static_locations['Lat'].mean()
        center_long = self.static_locations['Long'].mean()
        
        m = folium.Map(
            location=[center_lat, center_long],
            zoom_start=self.DEFAULT_ZOOM,
            tiles=self.MAP_STYLE
        )
        
        # Add layers
        self._add_neighborhood_layer(m)
        if show_static:
            self._add_static_sensor_layer(m, timestamp)
        if show_mobile and self.has_mobile_data:
            self._add_mobile_sensor_layer(m, timestamp)
        
        # Add controls
        self._add_map_controls(m, timestamp)
        
        # Save and return
        map_path = self.output_dir / 'radiation_map.html'
        m.save(str(map_path))
        return 'radiation_map.html'
    
    

    def create_time_slider_map(self) -> str:
        """Create a map with working timeline drag and visible radiation levels."""
        # Create base map
        center_lat = self.static_locations['Lat'].mean()
        center_long = self.static_locations['Long'].mean()
        
        m = folium.Map(
            location=[center_lat, center_long],
            zoom_start=12,
            tiles='CartoDB positron'
        )
        
        # Add neighborhood layer
        self._add_neighborhood_layer(m)

        # Prepare hourly data for visualization
        hourly_data = self.static_data_with_loc.copy()
        hourly_data['hour'] = hourly_data['Timestamp'].dt.floor('H')
        hours = sorted(hourly_data['hour'].unique())
        hours_json = [h.strftime('%Y-%m-%dT%H:%M:%S') for h in hours]
        
        # Convert data to JSON for JavaScript
        data_for_js = []
        for hour in hours:
            hour_data = hourly_data[hourly_data['hour'] == hour]
            data_for_js.append({
                'timestamp': hour.strftime('%Y-%m-%dT%H:%M:%S'),
                'readings': [{
                    'lat': row['Lat'],
                    'lng': row['Long'],
                    'value': row['Value'],
                    'sensor_id': row['Sensor-id']
                } for _, row in hour_data.iterrows()]
            })

        css = """
        <style>
        .time-control-panel {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            z-index: 1000;
            width: 90%;
            max-width: 800px;
        }
        .slider-container {
            width: 100%;
            background: #f5f5f5;
            border-radius: 4px;
            position: relative;
            height: 60px;
            margin-top: 10px;
        }
        .slider-track {
            position: absolute;
            width: 100%;
            height: 10px;
            background: #ddd;
            top: 25px;
            border-radius: 5px;
        }
        .slider-handle {
            position: absolute;
            width: 4px;
            height: 40px;
            background: #007bff;
            top: 10px;
            margin-left: -2px;
            cursor: pointer;
        }
        .time-labels {
            display: flex;
            justify-content: space-between;
            margin-top: 45px;
            font-size: 12px;
            color: #666;
        }
        .controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .play-button {
            background: #007bff;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
        }
        .time-display {
            font-family: monospace;
            font-size: 14px;
            padding: 5px 10px;
            background: #f8f9fa;
            border-radius: 4px;
        }
        </style>
        """

        js = """
        <div class="time-control-panel">
            <div class="controls">
                <button id="playButton" class="play-button">Play</button>
                <div id="timeDisplay" class="time-display"></div>
                <div id="radiationDisplay" class="time-display"></div>
                <select id="speedControl">
                    <option value="2000">0.5x</option>
                    <option value="1000" selected>1x</option>
                    <option value="500">2x</option>
                    <option value="250">4x</option>
                </select>
            </div>
            <div class="slider-container" id="sliderContainer">
                <div class="slider-track"></div>
                <div class="slider-handle" id="sliderHandle"></div>
                <div class="time-labels" id="timeLabels"></div>
            </div>
        </div>

        <script>
        (function() {
            const timeData = """ + str(data_for_js) + """;
            let currentIndex = 0;
            let isPlaying = false;
            let playInterval;
            let heatmapLayer = null;
            
            const sliderContainer = document.getElementById('sliderContainer');
            const sliderHandle = document.getElementById('sliderHandle');
            const timeLabels = document.getElementById('timeLabels');
            const playButton = document.getElementById('playButton');
            const timeDisplay = document.getElementById('timeDisplay');
            const radiationDisplay = document.getElementById('radiationDisplay');
            const speedControl = document.getElementById('speedControl');

            // Add time labels
            const labelCount = 6;
            for (let i = 0; i < labelCount; i++) {
                const label = document.createElement('div');
                const date = new Date(timeData[Math.floor(i * (timeData.length - 1) / (labelCount - 1))].timestamp);
                label.textContent = date.toLocaleString([], {
                    month: 'numeric',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit'
                });
                timeLabels.appendChild(label);
            }

            function updateVisualization(index) {
    const currentData = timeData[index];
    const timestamp = new Date(currentData.timestamp);
    
    // Debug output
    console.log('Updating visualization for index:', index);
    console.log('Number of readings:', currentData.readings.length);
    
    // Update displays
    timeDisplay.textContent = timestamp.toLocaleString();
    const avgRadiation = currentData.readings.reduce((sum, r) => sum + r.value, 0) / currentData.readings.length;
    radiationDisplay.textContent = `Avg: ${avgRadiation.toFixed(1)} cpm`;

    // Update slider position
    const percent = (index / (timeData.length - 1)) * 100;
    sliderHandle.style.left = `${percent}%`;

    // Clear previous layers
    if (heatmapLayer) {
        map.removeLayer(heatmapLayer);
        heatmapLayer = null;
    }

    // Debug heatmap data creation
    const heatData = currentData.readings.map(r => {
        console.log('Processing reading:', r.value);
        return [
            parseFloat(r.lat),
            parseFloat(r.lng),
            parseFloat(r.value)
        ];
    });
    
    console.log('Created heat data:', heatData.slice(0, 2));

    // Create new heatmap with debugging
    try {
        heatmapLayer = L.heatLayer(heatData, {
            radius: 25,          // Reduced radius for testing
            blur: 15,            // Reduced blur for testing
            maxZoom: 10,
            max: 1000,          // Maximum radiation value
            minOpacity: 0.5,    // Increased for visibility
            gradient: {
                0.2: 'blue',
                0.4: 'cyan',
                0.6: 'lime',
                0.8: 'yellow',
                1.0: 'red'
            }
        }).addTo(map);
        console.log('Heatmap added successfully');
    } catch (error) {
        console.error('Error creating heatmap:', error);
    }

    // Add sensor markers with values for debugging
    currentData.readings.forEach(r => {
        L.circleMarker([r.lat, r.lng], {
            radius: 4,
            color: getRadiationColor(r.value),
            fillColor: getRadiationColor(r.value),
            fillOpacity: 0.8
        }).bindPopup(`Value: ${r.value.toFixed(1)} cpm`)
          .addTo(map);
    });
}

function getRadiationColor(value) {
    if (value > 800) return 'red';
    if (value > 400) return 'yellow';
    return 'blue';
}

// Add this helper function to check data validity
function validateData(data) {
    return data.every(reading => {
        const isValid = !isNaN(reading[0]) && !isNaN(reading[1]) && !isNaN(reading[2]);
        if (!isValid) {
            console.error('Invalid reading:', reading);
        }
        return isValid;
    });
}

            // Handle slider drag
            let isDragging = false;
            
            sliderHandle.addEventListener('mousedown', (e) => {
                isDragging = true;
                e.preventDefault();
            });

            document.addEventListener('mousemove', (e) => {
                if (!isDragging) return;
                
                const rect = sliderContainer.getBoundingClientRect();
                let percent = (e.clientX - rect.left) / rect.width;
                percent = Math.max(0, Math.min(1, percent));
                
                currentIndex = Math.round(percent * (timeData.length - 1));
                updateVisualization(currentIndex);
            });

            document.addEventListener('mouseup', () => {
                isDragging = false;
            });

            // Handle slider click
            sliderContainer.addEventListener('click', (e) => {
                if (e.target === sliderHandle) return;
                
                const rect = sliderContainer.getBoundingClientRect();
                let percent = (e.clientX - rect.left) / rect.width;
                percent = Math.max(0, Math.min(1, percent));
                
                currentIndex = Math.round(percent * (timeData.length - 1));
                updateVisualization(currentIndex);
            });

            // Play button handler
            playButton.onclick = function() {
                if (isPlaying) {
                    clearInterval(playInterval);
                    this.textContent = 'Play';
                } else {
                    this.textContent = 'Pause';
                    playInterval = setInterval(() => {
                        currentIndex = (currentIndex + 1) % timeData.length;
                        updateVisualization(currentIndex);
                    }, parseInt(speedControl.value));
                }
                isPlaying = !isPlaying;
            };

            // Speed control handler
            speedControl.onchange = function() {
                if (isPlaying) {
                    clearInterval(playInterval);
                    playInterval = setInterval(() => {
                        currentIndex = (currentIndex + 1) % timeData.length;
                        updateVisualization(currentIndex);
                    }, parseInt(this.value));
                }
            };

            // Initial visualization
            updateVisualization(0);
        })();
        </script>
        """

        # Add elements to map
        m.get_root().html.add_child(folium.Element(css + js))
        
        # Save map
        map_path = self.output_dir / 'radiation_time_map.html'
        m.save(str(map_path))
        
        return 'radiation_time_map.html' 

    def _generate_dashboard_html(self, map_file: str) -> str:
        """Generate HTML content for the dashboard."""
        time_map_file = self.create_time_slider_map()
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>St. Himark Radiation Monitoring Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ margin: 0; padding: 20px; font-family: Arial, sans-serif; background-color: #f5f5f5; }}
                .dashboard-container {{ display: grid; gap: 20px; }}
                .dashboard-item {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .row {{ display: flex; gap: 20px; margin-bottom: 20px; }}
                .plot {{ width: 100%; height: 500px; }}
                .map-container {{ width: 100%; height: 600px; }}
                .tabs {{ display: flex; gap: 10px; margin-bottom: 10px; }}
                .tab {{ padding: 10px 20px; cursor: pointer; border-radius: 4px; }}
                .tab.active {{ background-color: #007bff; color: white; }}
                .tab-content {{ display: none; }}
                .tab-content.active {{ display: block; }}
            </style>
        </head>
        <body>
            <div class="dashboard-container">
                <h1>St. Himark Radiation Monitoring Dashboard</h1>
                
                <div class="status">
                    <h3>System Status</h3>
                    <p>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    {'<p class="status-warning">⚠️ Note: Mobile sensor data not available</p>' if not self.has_mobile_data else ''}
                </div>
                
                <div class="dashboard-item">
                    <div class="tabs">
                        <div class="tab active" onclick="switchTab('current-map')">Current View</div>
                        <div class="tab" onclick="switchTab('time-map')">Time Animation</div>
                    </div>
                    
                    <div id="current-map" class="tab-content active">
                        <h2>Current Radiation Levels</h2>
                        <iframe class="map-container" src="{map_file}" frameborder="0"></iframe>
                    </div>
                    
                    <div id="time-map" class="tab-content">
                        <h2>Radiation Levels Over Time</h2>
                        <iframe class="map-container" src="{time_map_file}" frameborder="0"></iframe>
                    </div>
                </div>

                <div class="row">
                    <div class="dashboard-item" style="flex: 1;">
                        <h2>Temporal Analysis</h2>
                        <div id="timeSeries" class="plot"></div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="dashboard-item" style="flex: 1;">
                        <h2>Statistical Analysis</h2>
                        <div id="statistics" class="plot"></div>
                    </div>
                </div>
            </div>

            <script>
                // Tab switching functionality
                function switchTab(tabId) {{
                    // Update tab buttons
                    document.querySelectorAll('.tab').forEach(tab => {{
                        tab.classList.remove('active');
                    }});
                    document.querySelector(`[onclick="switchTab('${{tabId}}')"]`).classList.add('active');
                    
                    // Update tab content
                    document.querySelectorAll('.tab-content').forEach(content => {{
                        content.classList.remove('active');
                    }});
                    document.getElementById(tabId).classList.add('active');
                }}
                
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

    def create_dashboard(self) -> Path:
        """
        Create complete dashboard with all components.
        
        Returns:
            Path: Path to the generated dashboard HTML file
        """
        try:
            # Create initial map
            initial_timestamp = self.static_data['Timestamp'].min().floor('h')
            map_file = self.create_map(initial_timestamp)
            
            # Generate dashboard HTML
            dashboard_html = self._generate_dashboard_html(map_file)
            
            # Save dashboard
            dashboard_path = self.output_dir / 'radiation_dashboard.html'
            with open(dashboard_path, 'w', encoding='utf-8') as f:
                f.write(dashboard_html)
            
            self.logger.info(f"Dashboard created successfully at {dashboard_path}")
            return dashboard_path
            
        except Exception as e:
            self.logger.error(f"Error creating dashboard: {str(e)}")
            raise

    def _load_mobile_data(self) -> None:
        """Load and process mobile sensor data."""
        self.logger.info("Loading mobile sensor data...")
        
        try:
            mobile_file = self.data_dir / 'MobileSensorReadings.csv'
            
            # Check if file exists
            if not mobile_file.exists():
                self.logger.warning(f"Mobile sensor data file not found at {mobile_file}")
                self._create_empty_mobile_data()
                return
                
            # Read file in chunks to handle large file size
            chunk_size = 100000  # Adjust based on available memory
            chunks = []
            
            try:
                # Read first chunk to check format
                first_chunk = pd.read_csv(mobile_file, nrows=5)
                
                if 'version https://git-lfs.github.com/spec/v1' in str(first_chunk.columns):
                    self.logger.warning("Mobile data file is a Git LFS pointer")
                    self._create_empty_mobile_data()
                    return
                    
                # Process file in chunks
                for chunk in pd.read_csv(mobile_file, chunksize=chunk_size):
                    # Clean chunk data
                    chunk = self._clean_mobile_chunk(chunk)
                    chunks.append(chunk)
                
                # Combine chunks
                self.mobile_data = pd.concat(chunks, ignore_index=True)
                
                # Post-process combined data
                self._post_process_mobile_data()
                
                self.has_mobile_data = True
                self.logger.info(f"Mobile data loaded successfully: {len(self.mobile_data)} records")
                
            except pd.errors.EmptyDataError:
                self.logger.warning("Mobile data file is empty")
                self._create_empty_mobile_data()
                
        except Exception as e:
            self.logger.warning(f"Could not load mobile sensor data: {str(e)}")
            self._create_empty_mobile_data()

    def _clean_mobile_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate a chunk of mobile sensor data."""
        # Convert timestamp
        if 'Timestamp' not in chunk.columns:
            timestamp_cols = [col for col in chunk.columns 
                            if 'time' in col.lower() or 'timestamp' in col.lower()]
            if timestamp_cols:
                chunk = chunk.rename(columns={timestamp_cols[0]: 'Timestamp'})
        
        chunk['Timestamp'] = pd.to_datetime(chunk['Timestamp'])
        
        # Clean sensor IDs
        if 'Sensor-id' not in chunk.columns:
            sensor_cols = [col for col in chunk.columns 
                        if 'sensor' in col.lower() or 'id' in col.lower()]
            if sensor_cols:
                chunk = chunk.rename(columns={sensor_cols[0]: 'Sensor-id'})
        
        # Remove any rows with missing essential data
        chunk = chunk.dropna(subset=['Timestamp', 'Sensor-id', 'Value', 'Lat', 'Long'])
        
        # Remove any obvious outliers
        chunk = chunk[
            (chunk['Value'] >= 0) &  # No negative radiation values
            (chunk['Value'] < chunk['Value'].quantile(0.999))  # Remove extreme outliers
        ]
        
        return chunk

    def _post_process_mobile_data(self) -> None:
        """Post-process the complete mobile dataset."""
        # Sort by timestamp
        self.mobile_data = self.mobile_data.sort_values('Timestamp')
        
        # Calculate additional metrics
        self.mobile_data['hour'] = self.mobile_data['Timestamp'].dt.floor('H')
        
        # Calculate movement metrics for each sensor
        sensor_movements = self.mobile_data.groupby('Sensor-id').agg({
            'Lat': ['min', 'max', 'std'],
            'Long': ['min', 'max', 'std']
        }).reset_index()
        
        # Flag potentially malfunctioning sensors (those that don't move)
        sensor_movements['is_stationary'] = (
            (sensor_movements['Lat']['std'] < 0.0001) & 
            (sensor_movements['Long']['std'] < 0.0001)
        )
        
        # Remove data from malfunctioning sensors
        valid_sensors = sensor_movements[~sensor_movements['is_stationary']]['Sensor-id']
        self.mobile_data = self.mobile_data[
            self.mobile_data['Sensor-id'].isin(valid_sensors)
        ]
        
        # Calculate coverage statistics
        self.mobile_coverage = self._calculate_mobile_coverage()

    def _calculate_mobile_coverage(self) -> Dict:
        """Calculate coverage statistics for mobile sensors."""
        coverage = {
            'total_sensors': self.mobile_data['Sensor-id'].nunique(),
            'total_readings': len(self.mobile_data),
            'time_range': {
                'start': self.mobile_data['Timestamp'].min(),
                'end': self.mobile_data['Timestamp'].max()
            },
            'readings_per_hour': self.mobile_data.groupby('hour').size().mean(),
            'active_areas': len(self.mobile_data.groupby(['Lat', 'Long']).size())
        }
        return coverage

    def _add_mobile_sensor_layer(self, m: folium.Map, timestamp: datetime) -> None:
        """Add mobile sensor clusters and radiation heatmap."""
        if not self.has_mobile_data:
            return

        # Get data for the specified timestamp with a time window
        time_window = 30  # minutes
        mobile_data_at_time = self.mobile_data[
            (self.mobile_data['Timestamp'] >= timestamp - timedelta(minutes=time_window)) &
            (self.mobile_data['Timestamp'] <= timestamp + timedelta(minutes=time_window))
        ]
        
        if mobile_data_at_time.empty:
            return

        # Add mobile sensor clusters with trails
        marker_cluster = MarkerCluster(name='Mobile Sensors')
        
        # Group by sensor to create trails
        for sensor_id, sensor_data in mobile_data_at_time.groupby('Sensor-id'):
            # Create trail
            points = sensor_data[['Lat', 'Long']].values
            if len(points) > 1:
                folium.PolyLine(
                    points,
                    color='blue',
                    weight=2,
                    opacity=0.8,
                    name=f'Sensor {sensor_id} Trail'
                ).add_to(m)
            
            # Add current position marker
            latest_pos = sensor_data.iloc[-1]
            folium.CircleMarker(
                location=[latest_pos['Lat'], latest_pos['Long']],
                radius=3,
                color='blue',
                fill=True,
                popup=f"""
                <div style="min-width: 150px;">
                    <b>Mobile Sensor {sensor_id}</b><br>
                    Value: {latest_pos['Value']:.2f} cpm<br>
                    Time: {latest_pos['Timestamp'].strftime('%H:%M:%S')}<br>
                    Readings in window: {len(sensor_data)}
                </div>
                """,
                tooltip=f"Mobile Sensor {sensor_id}"
            ).add_to(marker_cluster)
        
        marker_cluster.add_to(m)

        # Add mobile radiation heatmap with temporal weighting
        weighted_data = []
        for _, row in mobile_data_at_time.iterrows():
            time_diff = abs((row['Timestamp'] - timestamp).total_seconds() / 60)
            weight = 1 - (time_diff / time_window)  # Higher weight for more recent readings
            weighted_data.append([
                row['Lat'],
                row['Long'],
                row['Value'] * weight
            ])

        HeatMap(
            data=weighted_data,
            name='Mobile Radiation Levels',
            radius=15,
            min_opacity=0.3,
            gradient={0.4: 'purple', 0.65: 'pink', 0.8: 'orange', 1: 'red'},
            show=True
        ).add_to(m)

    def create_coverage_analysis(self) -> go.Figure:
        """Create a coverage analysis visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Sensor Activity Over Time',
                'Coverage Heatmap',
                'Readings Distribution',
                'Sensor Movement Patterns'
            )
        )
        
        # Sensor activity over time
        static_hourly = self.static_data.groupby(
            pd.Grouper(key='Timestamp', freq='1H')
        ).size()
        
        fig.add_trace(
            go.Scatter(
                x=static_hourly.index,
                y=static_hourly.values,
                name='Static Sensors',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        if self.has_mobile_data:
            mobile_hourly = self.mobile_data.groupby(
                pd.Grouper(key='Timestamp', freq='1H')
            ).size()
            
            fig.add_trace(
                go.Scatter(
                    x=mobile_hourly.index,
                    y=mobile_hourly.values,
                    name='Mobile Sensors',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Add mobile-specific visualizations
            # Coverage heatmap
            fig.add_trace(
                go.Histogram2d(
                    x=self.mobile_data['Long'],
                    y=self.mobile_data['Lat'],
                    colorscale='Viridis',
                    name='Coverage Density'
                ),
                row=1, col=2
            )
            
            # Readings distribution
            fig.add_trace(
                go.Histogram(
                    x=self.mobile_data['Value'],
                    name='Mobile Readings',
                    nbinsx=50
                ),
                row=2, col=1
            )
            
            # Sensor movement patterns
            for sensor_id in self.mobile_data['Sensor-id'].unique()[:10]:  # Show first 10 sensors
                sensor_data = self.mobile_data[self.mobile_data['Sensor-id'] == sensor_id]
                fig.add_trace(
                    go.Scatter(
                        x=sensor_data['Long'],
                        y=sensor_data['Lat'],
                        mode='lines',
                        name=f'Sensor {sensor_id}',
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Sensor Coverage Analysis"
        )
        
        return fig
def main():
        """Main function to create the dashboard."""
        try:
            # Setup paths
            project_root = Path.cwd().parent
            data_dir = project_root / 'data' / 'raw'
            output_dir = project_root / 'notebooks' / 'output'
            
            # Create dashboard
            dashboard = RadiationDashboard(data_dir, output_dir)
            dashboard_path = dashboard.create_dashboard()
            
            print("\nDashboard created successfully!")
            print(f"Main dashboard: {dashboard_path}")
            print(f"Map file: {output_dir / 'radiation_map.html'}")
            print("\nOpen the main dashboard file in your web browser to view.")
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            raise

if __name__ == "__main__":
    main()