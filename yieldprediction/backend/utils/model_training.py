import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------
# Paths
# ---------------------
# DATA_PATH should be a string, not a DataFrame
DATA_PATH = "/home/rdk/workspace/yieldprediction/project/yieldprediction/backend/farm_features.csv"

# Create models directory - os.makedirs returns None, so we need to define the path separately
MODEL_DIR = "/home/rdk/workspace/yieldprediction/project/yieldprediction/backend/models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "yield_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")

# ---------------------
# Expected feature set
# ---------------------
FEATURES = [
    "mean_ndvi",
    "max_ndvi",
    "std_ndvi",
    "total_rain",
    "mean_temp",
    "rainy_dekads",
    "early_ndvi",
    "late_ndvi",
]

TARGET = "yield_kg_ha"  # target column in CSV

# ---------------------
# Load dataset
# ---------------------
print("Loading features dataset...")

try:
    df = pd.read_csv(DATA_PATH)
    print(f"Features dataset loaded. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Create mock yield for development
    print("\nCreating mock yield data for development/testing...")
    
    # Calculate realistic yield based on NDVI and rainfall
    # Typical maize yield in Zambia: 1500-5000 kg/ha
    df['yield_kg_ha'] = (
        2500 +  # Base yield
        1500 * df['max_ndvi'] +  # NDVI contribution (0-1 scale)
        0.3 * df['total_rain'] +  # Rainfall contribution
        np.random.normal(0, 300, len(df))  # Random variation
    )
    
    # Ensure realistic range
    df['yield_kg_ha'] = df['yield_kg_ha'].clip(1500, 5500)
    
    print(f"Mock yield created. Range: {df['yield_kg_ha'].min():.0f}-{df['yield_kg_ha'].max():.0f} kg/ha")
    
except Exception as e:
    raise ValueError(f"Error reading CSV file: {e}")

# Basic validation
missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    print(f"Missing columns in dataset: {missing}")
    print(f"Available columns: {list(df.columns)}")
    raise ValueError(f"Missing required columns in dataset: {missing}")

# Drop rows with missing target
initial_len = len(df)
df = df.dropna(subset=[TARGET])
if initial_len != len(df):
    print(f"Dropped {initial_len - len(df)} rows with missing target")
    print(f"Remaining rows: {len(df)}")

# Fill missing feature values with conservative defaults
fill_defaults = {
    "std_ndvi": 0.1,
    "mean_temp": 24.0,
    "rainy_dekads": 3,
}

# Apply fill only to columns that exist
for col, default_val in fill_defaults.items():
    if col in df.columns:
        df[col] = df[col].fillna(default_val)
    else:
        print(f"Warning: Column '{col}' not found for filling defaults")

# Check for remaining NaN values
nan_counts = df[FEATURES].isna().sum()
if nan_counts.any():
    print("Warning: Some features still contain NaN values:")
    for col in FEATURES:
        if col in df.columns and df[col].isna().sum() > 0:
            print(f"   {col}: {df[col].isna().sum()} NaN values")
    # Fill any remaining NaN with column means
    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

# ---------------------
# Train / validation split
# ---------------------
X = df[FEATURES].astype(float)
y = df[TARGET].astype(float)

print(f"\nData Summary:")
print(f"   Features shape: {X.shape}")
print(f"   Target shape: {y.shape}")
print(f"   Target range: {y.min():.1f} - {y.max():.1f} kg/ha")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\nTrain/Test Split:")
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Testing samples: {X_test.shape[0]}")

# ---------------------
# Feature scaling
# ---------------------
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------
# Model definition
# ---------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=14,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,  # Use all available CPU cores
    verbose=1   # Show progress during training
)

# ---------------------
# Training
# ---------------------
print("\nTraining Random Forest model...")
model.fit(X_train_scaled, y_train)
print("Model training completed")

# ---------------------
# Evaluation
# ---------------------
print("\nModel Evaluation:")
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=" * 40)
print(f"MAE:  {mae:.2f} kg/ha")
print(f"R2:   {r2:.3f}")
print("=" * 40)

# Calculate percentage error for better interpretation
y_mean = y_test.mean()
mae_percent = (mae / y_mean) * 100 if y_mean > 0 else 0
print(f"MAE (% of mean yield): {mae_percent:.1f}%")

# ---------------------
# Feature importance (for analytics)
# ---------------------
importances = pd.Series(
    model.feature_importances_, index=FEATURES
).sort_values(ascending=False)

print("\nFeature Importance:")
print("-" * 30)
for k, v in importances.items():
    print(f"  {k:15s}: {v:.4f}")

# ---------------------
# Persist artifacts
# ---------------------
print("\nSaving artifacts...")
try:
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to: {SCALER_PATH}")
    
    # Save feature importance for reference
    importance_df = pd.DataFrame({
        'feature': importances.index,
        'importance': importances.values
    })
    importance_path = os.path.join(MODEL_DIR, "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"Feature importance saved to: {importance_path}")
    
except Exception as e:
    print(f"Error saving artifacts: {e}")
    raise

# ---------------------
# Create prediction function for API use
# ---------------------
def predict_yield_single(features_dict):
    """
    Predict yield for a single farm instance
    
    Args:
        features_dict: Dictionary with feature names as keys
        
    Returns:
        dict: Prediction result
    """
    try:
        # Convert input to DataFrame with correct feature order
        features_df = pd.DataFrame([features_dict])[FEATURES]
        
        # Scale features
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        return {
            "success": True,
            "predicted_yield_kg_ha": float(prediction),
            "predicted_yield_t_ha": float(prediction) / 1000.0,
            "features_used": FEATURES
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Test prediction with sample data
print("\nSample Prediction Test:")
sample_features = {
    "mean_ndvi": 0.65,
    "max_ndvi": 0.75,
    "std_ndvi": 0.1,
    "total_rain": 850.0,
    "mean_temp": 24.5,
    "rainy_dekads": 6,
    "early_ndvi": 0.55,
    "late_ndvi": 0.70
}
result = predict_yield_single(sample_features)
if result["success"]:
    print(f"   Predicted yield: {result['predicted_yield_kg_ha']:.1f} kg/ha")
    print(f"                     {result['predicted_yield_t_ha']:.2f} t/ha")
else:
    print(f"   Test failed: {result['error']}")

print("\n" + "=" * 50)
print("Module 3 training complete!")
print("=" * 50)