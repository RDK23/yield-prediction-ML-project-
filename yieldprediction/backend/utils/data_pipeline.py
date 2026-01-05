import ee
import pandas as pd
from datetime import datetime
import requests
import time
import numpy as np

# 1. Authenticate with GEE
try:
    ee.Initialize(project="yield-prediction-mini-project")
    print("GEE initialized successfully")
except Exception as e:
    print("GEE not initialized, authenticating...")
    ee.Authenticate()
    ee.Initialize(project="yield-prediction-mini-project")
    print("GEE authenticated and initialized")

# 2. Helper Functions
def get_dekad_from_date(date_str):
    """Convert date to dekadal (10-day) period"""
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    dekad = min((date_obj.day - 1) // 10 + 1, 3)
    return f"{date_obj.year}-{date_obj.month:02d}-{dekad}"

def fetch_weather_data(lat, lon, start_date='2024-01-01', end_date='2024-12-31'):
    """Fetch weather data from Open-Meteo API"""
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'daily': ['temperature_2m_mean', 'precipitation_sum'],
        'timezone': 'auto',
        'models': ['era5']
    }
    
    try:
        print(f"Fetching weather data for lat:{lat}, lon:{lon}")
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        print("Weather data retrieved successfully")
        return data
    except Exception as e:
        print(f"Weather API error: {e}")
        return None

def process_seasonal_weather(weather_data, season_year):
    """Process weather data for entire season"""
    if not weather_data or 'daily' not in weather_data:
        return None, None, None
    
    try:
        # Create DataFrame from weather data
        dates = weather_data['daily']['time']
        precip = weather_data['daily']['precipitation_sum']
        temp = weather_data['daily']['temperature_2m_mean']
        
        weather_df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'precipitation': precip,
            'temperature': temp
        })
        
        # Filter for growing season (Nov-Apr)
        season_mask = (weather_df['date'].dt.month >= 11) | (weather_df['date'].dt.month <= 4)
        season_weather = weather_df[season_mask]
        
        if season_weather.empty:
            return None, None, None
        
        # Calculate seasonal aggregates
        total_rain = season_weather['precipitation'].sum()
        mean_temp = season_weather['temperature'].mean()
        rainy_days = (season_weather['precipitation'] > 10).sum()
        
        return total_rain, mean_temp, rainy_days
        
    except Exception as e:
        print(f"Error processing weather data: {e}")
        return None, None, None

def create_seasonal_features(dekadal_data):
    """Convert dekadal time series to seasonal features"""
    seasonal_features = []
    
    for farm_id in dekadal_data['farm_id'].unique():
        farm_data = dekadal_data[dekadal_data['farm_id'] == farm_id].copy()
        
        if farm_data.empty:
            continue
            
        # Get season from dekad (use the most common year)
        seasons = farm_data['dekad'].str.split('-').str[0].unique()
        season = seasons[0] if len(seasons) > 0 else '2024'
        
        # NDVI features
        mean_ndvi = farm_data['ndvi'].mean()
        max_ndvi = farm_data['ndvi'].max()
        std_ndvi = farm_data['ndvi'].std()
        
        # Early and late season NDVI (first and last 3 dekads)
        sorted_data = farm_data.sort_values('dekad')
        early_ndvi = sorted_data.head(3)['ndvi'].mean() if len(sorted_data) >= 3 else mean_ndvi
        late_ndvi = sorted_data.tail(3)['ndvi'].mean() if len(sorted_data) >= 3 else mean_ndvi
        
        # Weather features (already aggregated per farm)
        total_rain = farm_data['precipitation_mm'].sum() if 'precipitation_mm' in farm_data else 0
        mean_temp = farm_data['temperature_c'].mean() if 'temperature_c' in farm_data else 0
        rainy_dekads = (farm_data['precipitation_mm'] > 10).sum() if 'precipitation_mm' in farm_data else 0
        
        # Handle missing values
        if pd.isna(std_ndvi):
            std_ndvi = 0.1
        if pd.isna(mean_temp):
            mean_temp = 24.0
        
        features = {
            'farm_id': farm_id,
            'season': int(season),
            'mean_ndvi': mean_ndvi,
            'max_ndvi': max_ndvi,
            'std_ndvi': std_ndvi,
            'early_ndvi': early_ndvi,
            'late_ndvi': late_ndvi,
            'total_rain': total_rain,
            'mean_temp': mean_temp,
            'rainy_dekads': rainy_dekads,
            'lat': farm_data['lat'].iloc[0],
            'lon': farm_data['lon'].iloc[0]
        }
        
        seasonal_features.append(features)
    
    return pd.DataFrame(seasonal_features)

# 3. Load farms from CSV
print("Loading farm data...")
farms_df = pd.read_csv("/home/rdk/workspace/yieldprediction/project/yieldprediction/backend/data/farms.csv")
print(f"Loaded {len(farms_df)} farms")

# 4. Process each farm
all_dekadal_data = []

for _, farm in farms_df.iterrows():
    farm_id = farm["farm_id"]
    lat = farm["lat"]
    lon = farm["lon"]
    area = farm["area_m2"]
    
    print(f"\nProcessing {farm_id} (lat: {lat}, lon: {lon})...")
    
    try:
        # Create farm geometry
        radius_meters = (area ** 0.5) / 2
        farm_geom = ee.Geometry.Point(lon, lat).buffer(radius_meters)
        
        # Define growing season for Zambia (Nov to Apr)
        start_date = '2024-11-01'
        end_date = '2025-06-30'
        
        # Get Sentinel-2 collection
        s2_collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start_date, end_date)
            .filterBounds(farm_geom)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            .map(lambda img: img.addBands(
                img.normalizedDifference(['B8', 'B4']).rename('NDVI')
            ))
        )
        
        # Extract NDVI features
        def extract_ndvi_feature(img):
            reduction = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=farm_geom,
                scale=10,
                bestEffort=True
            )
            return ee.Feature(None, {
                'date': ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
                'ndvi': reduction.get('NDVI'),
                'cloud_cover': img.get('CLOUDY_PIXEL_PERCENTAGE')
            })
        
        ndvi_fc = s2_collection.map(extract_ndvi_feature)
        ndvi_info = ndvi_fc.getInfo()
        
        if not ndvi_info or 'features' not in ndvi_info:
            print(f"No Sentinel-2 data found for {farm_id}")
            continue
            
        # Process NDVI data
        ndvi_records = []
        for feature in ndvi_info['features']:
            props = feature['properties']
            if props['ndvi'] is not None:
                ndvi_records.append({
                    'date': props['date'],
                    'ndvi': props['ndvi'],
                    'cloud_cover': props.get('cloud_cover', None)
                })
        
        if not ndvi_records:
            print(f"No valid NDVI values for {farm_id}")
            continue
            
        print(f"Found {len(ndvi_records)} NDVI observations")
        
        # Convert to DataFrame and aggregate to dekadal
        ndvi_df = pd.DataFrame(ndvi_records)
        ndvi_df['date'] = pd.to_datetime(ndvi_df['date'])
        ndvi_df = ndvi_df.sort_values('date')
        
        ndvi_df['dekad'] = ndvi_df['date'].apply(
            lambda x: f"{x.year}-{x.month:02d}-{min((x.day-1)//10 + 1, 3)}"
        )
        
        dekadal_ndvi = ndvi_df.groupby('dekad').agg({
            'ndvi': 'mean',
            'cloud_cover': 'mean'
        }).reset_index()
        
        # Fetch weather data for the entire season
        weather_data = fetch_weather_data(lat, lon, '2024-11-01', '2025-04-30')
        total_rain, mean_temp, rainy_days = process_seasonal_weather(weather_data, '2024')
        
        # Add farm info and weather to each dekadal record
        for _, row in dekadal_ndvi.iterrows():
            entry = {
                'farm_id': farm_id,
                'dekad': row['dekad'],
                'ndvi': row['ndvi'],
                'cloud_cover_pct': row['cloud_cover'],
                'precipitation_mm': total_rain / len(dekadal_ndvi) if total_rain else 0,  # Distribute evenly
                'temperature_c': mean_temp if mean_temp else 24.0,
                'lat': lat,
                'lon': lon,
                'area_m2': area
            }
            
            all_dekadal_data.append(entry)
            
        time.sleep(1)  # Rate limiting
            
    except Exception as e:
        print(f"Error processing {farm_id}: {str(e)}")
        continue

# 5. Create seasonal features and save
if all_dekadal_data:
    # Save dekadal time series (for reference)
    dekadal_df = pd.DataFrame(all_dekadal_data)
    dekadal_df = dekadal_df.sort_values(['farm_id', 'dekad'])
    dekadal_df.to_csv("zambia_farms_time_series_2.csv", index=False)
    print(f"Saved dekadal time series: {len(dekadal_df)} records")
    
    # Create seasonal features for model training
    seasonal_features = create_seasonal_features(dekadal_df)
    
    if not seasonal_features.empty:
        # Save seasonal features (for Module 3)
        output_file = "farm_features.csv"
        seasonal_features.to_csv(output_file, index=False)
        
        print(f"Success! Saved seasonal features for {len(seasonal_features)} farms to {output_file}")
        print("Feature summary:")
        print(f"   - Farms processed: {len(seasonal_features)}")
        print(f"   - Features per farm: {len(seasonal_features.columns)}")
        print(f"   - NDVI range: {seasonal_features['mean_ndvi'].min():.3f} - {seasonal_features['mean_ndvi'].max():.3f}")
        print(f"   - Rainfall range: {seasonal_features['total_rain'].min():.1f} - {seasonal_features['total_rain'].max():.1f} mm")
        
        print("\nSample of seasonal features:")
        print(seasonal_features.head())
    else:
        print("No seasonal features were created")
else:
    print("No data was processed.")