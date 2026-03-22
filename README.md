# AutoVal — Car Price Prediction Web App

A full-stack machine learning web application that predicts used car resale prices using **Random Forest Regressor** and **Linear Regression**, trained on the real **CarDekho dataset** (15,397 records). Served via a **Flask** backend with a dark-themed, chart-rich frontend.

---

## Project Structure

```
AutoVal/
├── app.py                    # Flask server — API routes, loads trained models
├── train.py                  # Training pipeline — run once to generate model files
├── requirements.txt          # Python dependencies
├── README.md
├── cardekho_dataset.csv      # Source dataset (place here before training)
│
├── templates/
│   └── index.html            # Frontend — form, result panel, metrics dashboard
│
└── (generated after training)
    ├── rf_model.pkl           # Trained Random Forest model
    ├── lr_model.pkl           # Trained Linear Regression model
    ├── scaler.pkl             # StandardScaler fitted for LR
    ├── features.pkl           # Ordered feature name list
    ├── encoder_classes.json   # LabelEncoder class lists for categorical fields
    └── metrics.json           # Real evaluation metrics + chart data
```

---

## Features

- **Real ML Prediction** — scikit-learn RandomForestRegressor and LinearRegression trained on 15,397 CarDekho records
- **10 Input Features** — brand, vehicle age, km driven, seller type, fuel type, transmission, mileage, engine cc, max power, seats
- **Confidence Score** — per-prediction confidence interval
- **Price Influencers** — shows which factors drove the price up or down
- **Live Metrics Dashboard** — R², MAE, RMSE, MAPE, scatter plot, residual histogram, feature importance, radar comparison, MAE-by-age chart — all sourced from the actual trained models
- **Retrain anytime** — just run `python train.py` to regenerate everything

---

## Input Features

| Feature | Type | Description |
|---|---|---|
| `brand` | categorical | Car manufacturer (Maruti, Hyundai, BMW, etc.) |
| `vehicle_age` | integer | Age of the vehicle in years |
| `km_driven` | integer | Total kilometres driven |
| `seller_type` | categorical | Individual / Dealer / Trustmark Dealer |
| `fuel_type` | categorical | Petrol / Diesel / CNG / LPG / Electric |
| `transmission_type` | categorical | Manual / Automatic |
| `mileage` | float | Fuel efficiency in kmpl |
| `engine` | integer | Engine displacement in cc |
| `max_power` | float | Maximum power output in bhp |
| `seats` | integer | Number of seats |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend |
| `POST` | `/api/predict` | Returns predicted price, range, confidence, factors |
| `GET` | `/api/metrics?algo=rf\|lr` | R², MAE, RMSE, MAPE |
| `GET` | `/api/metrics/scatter?algo=rf\|lr` | Actual vs predicted scatter data |
| `GET` | `/api/metrics/residuals?algo=rf\|lr` | Residual histogram data |
| `GET` | `/api/metrics/importance` | Feature importance for both models |
| `GET` | `/api/metrics/age-error` | MAE by vehicle age |
| `GET` | `/api/encoders` | Valid values for all categorical fields |
| `GET` | `/api/health` | Health check |

### Example — POST /api/predict

**Request**
```json
{
  "brand": "Hyundai",
  "vehicle_age": 6,
  "km_driven": 45000,
  "seller_type": "Individual",
  "fuel_type": "Petrol",
  "transmission_type": "Manual",
  "mileage": 18.9,
  "engine": 1197,
  "max_power": 82.0,
  "seats": 5,
  "model": "rf"
}
```

**Response**
```json
{
  "price": 5.42,
  "price_low": 4.99,
  "price_high": 5.85,
  "confidence": 88.3,
  "model_name": "Random Forest Regressor",
  "factors": [
    { "name": "Brand",            "impact": 1,    "label": "Hyundai" },
    { "name": "Vehicle Age",      "impact": 0,    "label": "6 yrs" },
    { "name": "Fuel Type",        "impact": 0,    "label": "Petrol" },
    { "name": "Transmission",     "impact": 0,    "label": "Manual" },
    { "name": "Engine",           "impact": 0,    "label": "1197 cc" },
    { "name": "Max Power",        "impact": 0,    "label": "82.0 bhp" },
    { "name": "Kilometres Driven","impact": 0,    "label": "45k km" }
  ]
}
```

---

## Setup & Running

### 1. Clone / download the project

```
AutoVal/
├── app.py
├── train.py
├── requirements.txt
├── cardekho_dataset.csv
└── templates/
    └── index.html
```

### 2. Create and activate virtual environment

```bash
# Create
python3 -m venv venv

# Activate — macOS / Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the models

```bash
python train.py --data cardekho_dataset.csv
```

This generates `rf_model.pkl`, `lr_model.pkl`, `scaler.pkl`, `features.pkl`, `encoder_classes.json`, and `metrics.json` in the project root. You only need to do this once (or whenever you want to retrain on new data).

Expected output:
```
── AutoVal Training Pipeline ──────────────────────────────────
[1/5] Loading & cleaning data...
  Loaded 15,397 rows, 11 columns after cleaning.
[2/5] Encoding categoricals...
[3/5] Splitting train / test (80 / 20)...
  Train: 12,317   Test: 3,080
[4/5] Training models...
  → Random Forest Regressor...
  → Linear Regression (with StandardScaler)...
[5/5] Saving artefacts...

── Results ─────────────────────────────────────────────────────
  Random Forest:      R²=0.938  MAE=₹0.94L  RMSE=₹1.98L  MAPE=13.4%
  Linear Regression:  R²=0.689  MAE=₹2.45L  RMSE=₹4.46L  MAPE=40.1%
```

### 5. Start the Flask server

```bash
python app.py
```

### 6. Open the app

Visit **http://localhost:5000**

---

## Real Model Evaluation Metrics

Computed from the CarDekho dataset — 80/20 train-test split, random state 42.

| Metric | Random Forest | Linear Regression |
|---|---|---|
| **R² Score** | **0.938** | 0.689 |
| **MAE (₹ Lakhs)** | **0.94** | 2.45 |
| **RMSE (₹ Lakhs)** | **1.98** | 4.46 |
| **MAPE** | **13.4%** | 40.1% |

### Top Feature Importances (Random Forest)

| Feature | Importance |
|---|---|
| Max Power (bhp) | 68.1% |
| Vehicle Age | 13.0% |
| Mileage | 8.1% |
| Kilometres Driven | 5.1% |
| Engine CC | 3.5% |

Max power is by far the strongest predictor of resale price — higher performance cars retain value significantly better than lower power equivalents regardless of age or usage.

---

## Dataset

**CarDekho Used Car Dataset**
- Source: [Kaggle — nehalbirla/vehicle-dataset-from-cardekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
- Records: 15,411 (15,397 after cleaning)
- Features used: 10
- Target: `selling_price` (converted to ₹ Lakhs)
- Cleaning applied: removed `km_driven > 500,000` (12 rows), removed `seats == 0` (2 rows)

---

## License

MIT License. Free to use and modify for academic or personal projects.
