"""
AutoVal — Flask Backend
Loads trained scikit-learn models and serves prediction + metrics API.

Endpoints:
    GET  /                          -> serves templates/index.html
    POST /api/predict               -> real ML prediction
    GET  /api/metrics?algo=rf|lr    -> KPI metrics
    GET  /api/metrics/scatter       -> scatter plot data
    GET  /api/metrics/residuals     -> residual histogram data
    GET  /api/metrics/importance    -> feature importance
    GET  /api/metrics/age-error     -> MAE by vehicle age
    GET  /api/encoders              -> valid categorical values
    GET  /api/health                -> health check
"""

import json
import os

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

app = Flask(__name__, template_folder="templates")
CORS(app)

# ── Load artefacts ───────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{filename}' not found. Run `python train.py` first.")
    return joblib.load(path)

rf_model = _load("rf_model.pkl")
lr_model = _load("lr_model.pkl")
scaler   = _load("scaler.pkl")
features = _load("features.pkl")

with open(os.path.join(BASE_DIR, "encoder_classes.json")) as f:
    ENCODER_CLASSES = json.load(f)

with open(os.path.join(BASE_DIR, "metrics.json")) as f:
    METRICS = json.load(f)

# ── Encoding helpers ─────────────────────────────────────────────────────────

def _encode(col, value):
    classes = ENCODER_CLASSES.get(col, [])
    return classes.index(value) if value in classes else 0

def _build_input_vector(brand, vehicle_age, km_driven, seller_type,
                        fuel_type, transmission_type, mileage, engine, max_power, seats):
    row = [
        _encode("brand",             brand),
        int(vehicle_age),
        int(km_driven),
        _encode("seller_type",       seller_type),
        _encode("fuel_type",         fuel_type),
        _encode("transmission_type", transmission_type),
        float(mileage),
        int(engine),
        float(max_power),
        int(seats),
    ]
    return np.array(row, dtype=float).reshape(1, -1)

def _build_factors(brand, vehicle_age, km_driven, fuel_type,
                   transmission_type, engine, max_power):
    return [
        {"name": "Brand",            "impact": 1,
         "label": brand},
        {"name": "Vehicle Age",      "impact": -1 if vehicle_age > 7 else (0 if vehicle_age > 3 else 1),
         "label": f"{vehicle_age} yrs"},
        {"name": "Fuel Type",        "impact": 1 if fuel_type == "Electric" else (0.5 if fuel_type == "Diesel" else 0),
         "label": fuel_type},
        {"name": "Transmission",     "impact": 1 if transmission_type == "Automatic" else 0,
         "label": transmission_type},
        {"name": "Engine",           "impact": 1 if engine > 1800 else (-1 if engine < 900 else 0),
         "label": f"{engine} cc"},
        {"name": "Max Power",        "impact": 1 if max_power > 120 else (-1 if max_power < 60 else 0),
         "label": f"{max_power} bhp"},
        {"name": "Kilometres Driven","impact": -1 if km_driven > 100000 else (1 if km_driven < 20000 else 0),
         "label": f"{round(km_driven/1000)}k km"},
    ]

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    required = ["brand","vehicle_age","km_driven","seller_type","fuel_type",
                "transmission_type","mileage","engine","max_power","seats","model"]
    missing = [f for f in required if data.get(f) in (None, "")]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    brand             = str(data["brand"])
    vehicle_age       = int(data["vehicle_age"])
    km_driven         = int(data["km_driven"])
    seller_type       = str(data["seller_type"])
    fuel_type         = str(data["fuel_type"])
    transmission_type = str(data["transmission_type"])
    mileage           = float(data["mileage"])
    engine            = int(data["engine"])
    max_power         = float(data["max_power"])
    seats             = int(data["seats"])
    model_choice      = str(data["model"])

    X = _build_input_vector(brand, vehicle_age, km_driven, seller_type,
                            fuel_type, transmission_type, mileage, engine, max_power, seats)

    if model_choice == "lr":
        price = float(np.clip(lr_model.predict(scaler.transform(X))[0], 0, None))
        margin, model_name = 0.15, "Linear Regression"
    else:
        price = float(rf_model.predict(X)[0])
        margin, model_name = 0.08, "Random Forest Regressor"

    base_conf = 93.85 if model_choice == "rf" else 68.88
    conf = round(min(96.0, max(50.0, base_conf - vehicle_age * 0.5 - (km_driven / 500000) * 8)), 1)

    return jsonify({
        "price":      round(price, 2),
        "price_low":  round(price * (1 - margin), 2),
        "price_high": round(price * (1 + margin), 2),
        "confidence": conf,
        "model_name": model_name,
        "factors":    _build_factors(brand, vehicle_age, km_driven,
                                     fuel_type, transmission_type, engine, max_power),
    })

@app.route("/api/metrics")
def metrics():
    algo = request.args.get("algo", "rf")
    if algo not in METRICS:
        return jsonify({"error": "Use algo=rf or algo=lr"}), 400
    return jsonify(METRICS[algo])

@app.route("/api/metrics/scatter")
def metrics_scatter():
    algo = request.args.get("algo", "rf")
    return jsonify({"algo": algo, "points": METRICS["scatter"][algo]})

@app.route("/api/metrics/residuals")
def metrics_residuals():
    algo = request.args.get("algo", "rf")
    return jsonify({"algo": algo, **METRICS["residuals"][algo]})

@app.route("/api/metrics/importance")
def metrics_importance():
    return jsonify(METRICS["feature_importance"])

@app.route("/api/metrics/age-error")
def metrics_age_error():
    return jsonify(METRICS["age_error"])

@app.route("/api/encoders")
def encoders():
    return jsonify(ENCODER_CLASSES)

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "AutoVal API", "models_loaded": True})

if __name__ == "__main__":
    print("\n AutoVal server starting on http://localhost:5000\n")
    app.run(debug=True, port=5000)
