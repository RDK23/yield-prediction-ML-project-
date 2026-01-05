from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import io
import json
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# app setup
app = Flask(__name__)
CORS(app)

# paths
MODEL_PATH = r"/home/rdk/workspace/yieldprediction/project/yieldprediction/backend/models/yield_model.pkl"
SCALER_PATH = r"/home/rdk/workspace/yieldprediction/project/yieldprediction/backend/models/feature_scaler.pkl"
FEATURES_CSV = r"/home/rdk/workspace/yieldprediction/project/yieldprediction/backend/data/farm_features.csv"
DB_PATH = r"/home/rdk/workspace/yieldprediction/project/yieldprediction/backend/data/predictions.db"

# load model and scaler
MODEL = joblib.load(MODEL_PATH)
SCALER = joblib.load(SCALER_PATH)
FEATURE_DF = pd.read_csv(FEATURES_CSV)

EXPECTED_FEATURES = [
    "mean_ndvi",
    "max_ndvi",
    "std_ndvi",
    "total_rain",
    "mean_temp",
    "rainy_dekads",
    "early_ndvi",
    "late_ndvi",
]   

# database setup
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute(""""
                CREATE TABLE INF NOT EXISTS farms(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                latitude REAL,
                longitude REAL
                area_m2 REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
    cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                latitude REAL,
                longitude REAL,
                predicted_yield REAL,
                confidence REAL,
                features TEXT,
                timestamp TEXT
                )
""")
    
    conn.commit()
    conn.close()

init_db()

# utilities
def to_native(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    return obj

def calculate_confidence(pred, scaled):
    feature_std = float(np.std(scaled))
    base = 0.8 if 800 <= pred <= 5000 else 0.4
    penalty = min(0.4, feature_std * 0.3)
    return max(0.1, min(0.95, base - penalty))

def get_features_from_coordinates(lat, lon, max_distance=1.0):
    df = FEATURE_DF.copy()
    df["dist"] = np.sqrt((df["lat"] - lat) ** 2 + 
                         (df["lon"] - lon) ** 2)
    idx = df["dist"].idxmin()
    row = df.loc[idx]

    if float(row["dist"]) > max_distance:
        return None
    def safe_float(v, d=0.0):
        try:
            return float(v)
        except:
            return d
    def safe_int(v, d=0):
        try:
            return int(v)
        except Exception:
            return d

    return {
        "mean_ndvi": safe_float(row.get("mean_ndvi")),
        "max_ndvi": safe_float(row.get("max_ndvi")),
        "std_ndvi": safe_float(row.get("std_ndvi"), 0.1),
        "total_rain": safe_float(row.get("total_rain")),
        "mean_temp": safe_float(row.get("mean_temp"), 24.0),
        "rainy_dekads": safe_int(row.get("rainy_dekads"), 3),
        "early_ndvi": safe_float(row.get("early_ndvi")),
        "late_ndvi": safe_float(row.get("late_ndvi")),
        }     

# APIs
# farms
@app.route('/api/farms/add', methods =["POST"])
def add_farm():
    data = request.get_json()
    name = data.get("name", "Farm")
    lat =  data.get("lat")
    lon = data.get("lon")
    area_m2 = data.get("area_m2", 10000)

    if lat is None or lon is None:
        return jsonify({"error": "latitude and longitude are required"}), 400
    
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
                "INSERT INTO farms (name, latitude, longitude, area_m2) VALUES (?, ?, ?, ?)",
                (name, lat, lon, area_m2),
                )
    conn.commit()
    farm_id = cur.lastrowid
    conn.close()

    return jsonify({
        "success": True,
        "farm_id":{ 
            "id":farm_id,
            "name": name,
            "lat": lat,
            "lon": lon,
            "area_m2": area_m2}
    }), 201

@app.route('api/farms/list', methods = ["GET"])
def list_farms():
    conn = get_db()
    rows = conn.execute("""
SELECT id, name, latitude AS lat, longitude AS lon, area_m2, created_at, createsd_at
FROM farms ORDER BY created_at DESC
""").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows]), 200

# predictions
@app.route('/api/predict', methods = ["POST"])
def predict_yield():
    data = request.get_json()
    coords = data.get("coordinates")

    if not coords:
        return jsonify({"error": "Missing coordinates"}), 404
    
    lat = coords.get("lat")
    lon = coords.get("lon")

    features = get_features_from_coordinates(lat, lon)
    if not features:
        return jsonify({"error": "No features found for the given coordinates"}), 404
    
    df = pd.DataFrame([[features[f] for f in EXPECTED_FEATURES]], columns = EXPECTED_FEATURES)
    scaled = SCALER.transform(df)
    pred = float(MODEL.predict(scaled)[0])
    confidence = calculate_confidence(pred, scaled)

    conn = get_db()
    cur = conn.cursor()
    cur.execute(""" 
               INSERT INTO predictions (latitude, longitude, predicted_yield, confidence, features, timestamp) VALUES (?, ?, ?, ?, ?, ?)
   """, (
       lat,
       lon,
       pred,
       confidence,
       json.dumps(to_native(features)),
       datetime.now().isoformat()
   ))
    conn.commit()
    conn.close()

    return jsonify({
        "predicted_yield": pred,
        "confidence": confidence,
        "features": features
    })

@app.route('/api/farms', methods = ["GET"])
def list_predictions():
    conn = get_db()
    rows = conn.execute("""
SELECT id, latitude, longitude, predicted_yield, confidence, features, timestamp
FROM predictions ORDER BY timestamp DESC LIMIT 100
""").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows]), 200

# dashboard
@app.route('/api/dashboard/summary', methods = ["GET"])
def dashboard_summary():
    conn = get_db()
    cur = conn.cursor()

    total = cur.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    avg_y, avg_c = cur.execute("""
        SELECT AVG(predicted_yield), AVG(confidence) FROM predictions
    """).fetchone()

    trend = cur.execute("""
                        SELECT id, timestamp, predicted_yield FROM predictions 
                        ORDER BY timestamp DESC LIMIT 10
                        """).fetchall()
    conn.close()

    return jsonify({
        "total_predictons": total,
        "average_yield": round(avg_y or 0,2),
        "average_confidence": round(avg_c or 0,3),
        "trend_data":[{
            "prediction_id": t[0],
            "timestamp": t[1],
            "yield": round(t[2], 2)
        } for t in reversed(trend)
        ]
    })

# feature importance
@app.route('/api/model/feature_importance', methods = ["GET"])
def feature_importance():
    if not hasattr(MODEL, "feature_importances_"):
        return jsonify({"error": "Model does not expose feature importance"}), 400
    
    return jsonify({
        "feature_importance": dict(
            zip(EXPECTED_FEATURES, MODEL.feature_importances_)
        )
    })

# health
@app.route('/api/health')
def health_check():
    return jsonify({"status": "ok"}), 200

# Run
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    