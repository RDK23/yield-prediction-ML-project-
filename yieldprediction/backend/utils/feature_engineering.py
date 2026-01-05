import pandas as pd
import numpy as np
import os

def engineer_features(input_file="/workspace/yieldprediction/project/yieldprediction/backend/data/zambia_farms_time_series_2.csv", 
                      output_file="farm_features.csv"):
    """
    Convert dekadal time series to seasonal features for yield prediction
    Ensures compatibility with Module 3 (model training) and Module 4 (Flask API)
    """
    
    # Load dekadal time-series
    try:
        df = pd.read_csv(input_file)
        print(f" Loaded {len(df)} dekadal records from {input_file}")
    except FileNotFoundError:
        print(f" File not found: {input_file}")
        return None

    # Extract season (year from dekadal string: "YYYY-MM-D")
    df['season'] = df['dekad'].apply(lambda x: int(x.split("-")[0]))

    feature_rows = []

    # Group by farm and season
    for (farm_id, season), group in df.groupby(['farm_id', 'season']):
        # Skip if no data
        if group.empty:
            continue
            
        # NDVI features with robust missing value handling
        mean_ndvi = group['ndvi'].mean()
        max_ndvi = group['ndvi'].max()
        std_ndvi = group['ndvi'].std()
        
        # Handle cases with insufficient data for std calculation
        if pd.isna(std_ndvi) or len(group) < 2:
            std_ndvi = 0.1  # Reasonable default
        
        # Early and late season NDVI (first and last 3 dekads)
        sorted_group = group.sort_values('dekad')
        early_ndvi = sorted_group.head(min(len(sorted_group), 3))['ndvi'].mean()
        late_ndvi = sorted_group.tail(min(len(sorted_group), 3))['ndvi'].mean()
        
        # Handle NaN in early/late NDVI
        if pd.isna(early_ndvi):
            early_ndvi = mean_ndvi
        if pd.isna(late_ndvi):
            late_ndvi = mean_ndvi

        # Weather features with robust missing value handling
        total_rain = group['precipitation_mm'].fillna(0).sum()
        
        # Better temperature handling - use median if mean has issues
        temp_values = group['temperature_c'].dropna()
        if len(temp_values) > 0:
            mean_temp = temp_values.mean()
        else:
            mean_temp = 24.0  # Reasonable default for Zambia
            
        rainy_dekads = (group['precipitation_mm'].fillna(0) > 10).sum()

        # Get farm location (from first record)
        lat = group['lat'].iloc[0]
        lon = group['lon'].iloc[0]
        area_m2 = group['area_m2'].iloc[0] if 'area_m2' in group.columns else 10000

        # Create feature row with EXACT 8 features expected by Modules 3 & 4
        feature_row = {
            "farm_id": farm_id,
            "season": season,
            "mean_ndvi": mean_ndvi,
            "max_ndvi": max_ndvi,
            "std_ndvi": std_ndvi,
            "early_ndvi": early_ndvi,
            "late_ndvi": late_ndvi,
            "total_rain": total_rain,
            "mean_temp": mean_temp,
            "rainy_dekads": rainy_dekads,
            "lat": lat,
            "lon": lon,
            "area_m2": area_m2,
            "yield_kg_ha": yield_kg_ha
        }
        
        feature_rows.append(feature_row)

    # Create features DataFrame
    features_df = pd.DataFrame(feature_rows)
    
    # Final validation of required features
    required_features = ['mean_ndvi', 'max_ndvi', 'std_ndvi', 'early_ndvi', 
                        'late_ndvi', 'total_rain', 'mean_temp', 'rainy_dekads']
    
    missing_features = [f for f in required_features if f not in features_df.columns]
    if missing_features:
        print(f"CRITICAL: Missing required features: {missing_features}")
        return None
    
    # Save features
    features_df.to_csv(output_file, index=False)
    
    print(f"Saved features for {len(features_df)} farm-seasons to {output_file}")
    print(f"Feature summary:")
    print(f"   - Farms processed: {len(features_df)}")
    print(f"   - Features created: {len(required_features)}")
    print(f"   - NDVI range: {features_df['mean_ndvi'].min():.3f} - {features_df['mean_ndvi'].max():.3f}")
    print(f"   - Data quality check:")
    
    # Data quality report
    for feature in required_features:
        missing = features_df[feature].isna().sum()
        if missing > 0:
            print(f"     {feature}: {missing} missing values")
        else:
            print(f"     {feature}: No missing values")
    
    print("\nSample of features (showing only the 8 model features):")
    print(features_df[required_features].head())
    
    return features_df

if __name__ == "__main__":
    # Use relative path for better portability
    engineer_features(f"/home/rdk/workspace/yieldprediction/project/yieldprediction/backend/data/zambia_farms_time_series_2.csv", "farm_features.csv")