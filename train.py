"""
AutoVal — Model Training Script
Run once to train both models and generate metrics.json.

Usage:
    python train.py --data cardekho_dataset.csv

Output files:
    rf_model.pkl          — trained Random Forest Regressor
    lr_model.pkl          — trained Linear Regression
    scaler.pkl            — StandardScaler fitted on training data (for LR)
    features.pkl          — ordered feature name list
    encoder_classes.json  — LabelEncoder class lists for all categoricals
    metrics.json          — real evaluation metrics + chart data for the UI
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─── Config ──────────────────────────────────────────────────────────────────

FEATURES = [
    "brand", "vehicle_age", "km_driven", "seller_type",
    "fuel_type", "transmission_type", "mileage", "engine", "max_power", "seats",
]

TARGET = "selling_price_lakhs"

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop unused columns
    drop_cols = [c for c in ["Unnamed: 0", "car_name", "model"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Outlier removal
    df = df[df["km_driven"] <= 500_000]
    df = df[df["seats"] > 0]

    # Normalise brand casing
    df["brand"] = df["brand"].str.strip()
    df.loc[df["brand"] == "Isuzu", "brand"] = "ISUZU"

    # Target in ₹ Lakhs
    df[TARGET] = df["selling_price"] / 100_000
    df = df.drop(columns=["selling_price"])

    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns after cleaning.")
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cat_cols = ["brand", "seller_type", "fuel_type", "transmission_type"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


def evaluate(y_true, y_pred) -> dict:
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-9, None))) * 100)
    return {
        "r2":   round(float(r2_score(y_true, y_pred)), 3),
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "mape": round(mape, 1),
    }


def build_metrics(
    y_test, y_pred_rf, y_pred_lr,
    rf_metrics, lr_metrics,
    rf_importances, X_test,
) -> dict:

    # Scatter — 60 random test samples
    rng = np.random.RandomState(42)
    idx = rng.choice(len(y_test), size=min(60, len(y_test)), replace=False)
    y_arr = y_test.values

    scatter = {
        "rf": [{"actual": round(float(y_arr[i]), 2),
                "predicted": round(float(y_pred_rf[i]), 2)} for i in idx],
        "lr": [{"actual": round(float(y_arr[i]), 2),
                "predicted": round(float(y_pred_lr[i]), 2)} for i in idx],
    }

    # Residual histograms
    bins = np.linspace(-12, 12, 17)
    rf_hist, edges = np.histogram(y_pred_rf - y_arr, bins=bins)
    lr_hist, _     = np.histogram(y_pred_lr - y_arr, bins=bins)
    centers = [round((float(edges[i]) + float(edges[i+1])) / 2, 2)
               for i in range(len(edges) - 1)]

    # Age-bucket MAE
    age_labels, age_rf, age_lr = [], [], []
    for age in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]:
        mask = X_test["vehicle_age"] == age
        if mask.sum() < 5:
            continue
        age_labels.append(f"{age} yr")
        age_rf.append(round(float(mean_absolute_error(y_arr[mask], y_pred_rf[mask])), 2))
        age_lr.append(round(float(mean_absolute_error(y_arr[mask.values], y_pred_lr[mask.values])), 2))

    # Feature importance (RF)
    fi_rf = sorted(
        [{"feature": FEATURES[i], "importance": round(float(rf_importances[i]) * 100, 2)}
         for i in range(len(FEATURES))],
        key=lambda x: -x["importance"],
    )
    # LR: use equal weights (coefficients vary with scale — not directly comparable)
    fi_lr = [{"feature": f, "importance": round(100 / len(FEATURES), 1)} for f in FEATURES]

    return {
        "rf":  {**rf_metrics, "label": "Random Forest Regressor"},
        "lr":  {**lr_metrics, "label": "Linear Regression"},
        "feature_importance": {"rf": fi_rf, "lr": fi_lr},
        "scatter":    scatter,
        "residuals":  {
            "rf": {"bins": centers, "counts": rf_hist.tolist()},
            "lr": {"bins": centers, "counts": lr_hist.tolist()},
        },
        "age_error":  {"ages": age_labels, "rf": age_rf, "lr": age_lr},
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train AutoVal models.")
    parser.add_argument("--data", default="cardekho_dataset.csv",
                        help="Path to CarDekho CSV dataset")
    args = parser.parse_args()

    print("\n── AutoVal Training Pipeline ──────────────────────────────────")

    # 1. Load & clean
    print("\n[1/5] Loading & cleaning data...")
    df = load_and_clean(args.data)

    # 2. Encode categoricals
    print("[2/5] Encoding categoricals...")
    df, encoders = encode_categoricals(df)
    encoder_map = {col: list(enc.classes_) for col, enc in encoders.items()}

    # 3. Split
    print("[3/5] Splitting train / test (80 / 20)...")
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")

    # 4. Train models
    print("[4/5] Training models...")

    print("  → Random Forest Regressor...")
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_metrics = evaluate(y_test.values, y_pred_rf)

    print("  → Linear Regression (with StandardScaler)...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    y_pred_lr = np.clip(lr.predict(X_test_s), 0, None)
    lr_metrics = evaluate(y_test.values, y_pred_lr)

    # 5. Save artefacts
    print("[5/5] Saving artefacts...")
    joblib.dump(rf,       "rf_model.pkl")
    joblib.dump(lr,       "lr_model.pkl")
    joblib.dump(scaler,   "scaler.pkl")
    joblib.dump(FEATURES, "features.pkl")

    with open("encoder_classes.json", "w") as f:
        json.dump(encoder_map, f, indent=2)

    metrics = build_metrics(
        y_test, y_pred_rf, y_pred_lr,
        rf_metrics, lr_metrics,
        rf.feature_importances_, X_test,
    )
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n── Results ─────────────────────────────────────────────────────")
    print(f"\n  Random Forest:      R²={rf_metrics['r2']}  "
          f"MAE=₹{rf_metrics['mae']}L  "
          f"RMSE=₹{rf_metrics['rmse']}L  "
          f"MAPE={rf_metrics['mape']}%")
    print(f"  Linear Regression:  R²={lr_metrics['r2']}  "
          f"MAE=₹{lr_metrics['mae']}L  "
          f"RMSE=₹{lr_metrics['rmse']}L  "
          f"MAPE={lr_metrics['mape']}%")
    print()
    print("  Top 5 feature importances (RF):")
    for item in metrics["feature_importance"]["rf"][:5]:
        print(f"    {item['feature']:<22} {item['importance']}%")
    print()
    print("  Saved: rf_model.pkl  lr_model.pkl  scaler.pkl  "
          "features.pkl  encoder_classes.json  metrics.json")
    print("\n── Done. Run `python app.py` to start the server. ────────────\n")


if __name__ == "__main__":
    main()
