"""
train_models.py
----------------
Trains 4 classifiers (Random Forest, XGBoost, KNN, Logistic Regression)
to predict a bucketed weather condition (Clear / Partly Cloudy / Cloudy /
Mist-Fog / Light Rain / Heavy Rain / Severe Weather) from live meteorological
readings (temperature, humidity, wind, pressure, cloud cover, etc.).

Run:  python train_models.py
Outputs (in ./models/):
    preprocessor.joblib      - fitted ColumnTransformer (scaling)
    label_encoder.joblib     - target label encoder
    model_random_forest.joblib
    model_xgboost.joblib
    model_knn.joblib
    model_logistic_regression.joblib
    metrics.json             - accuracy / precision / recall / f1 per model
    feature_columns.json     - ordered list of input feature names
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_PATH = "data/IndianWeatherRepository.csv"
MODELS_DIR = "models"

# ---------------------------------------------------------------------------
# 1. Bucket the 29 raw condition_text values into 7 balanced-ish classes
# ---------------------------------------------------------------------------
def bucket_condition(c: str) -> str:
    c = c.lower()
    if "thunder" in c or "snow" in c or "sleet" in c or "freezing" in c:
        return "Severe Weather"
    if "torrential" in c or "heavy rain" in c or "moderate rain" in c:
        return "Heavy Rain"
    if "drizzle" in c or "light rain" in c or "patchy rain" in c or "rain shower" in c:
        return "Light Rain"
    if "mist" in c or "fog" in c:
        return "Mist/Fog"
    if "overcast" in c or c == "cloudy":
        return "Cloudy"
    if "partly cloudy" in c:
        return "Partly Cloudy"
    if "clear" in c or "sunny" in c:
        return "Clear"
    return "Other"


FEATURE_COLUMNS = [
    "temperature_celsius",
    "humidity",
    "wind_kph",
    "pressure_mb",
    "precip_mm",
    "cloud",
    "visibility_km",
    "uv_index",
    "gust_kph",
    "air_quality_PM2.5",
    "air_quality_PM10",
    "latitude",
    "longitude",
]


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df["weather_bucket"] = df["condition_text"].apply(bucket_condition)
    df = df[df["weather_bucket"] != "Other"].reset_index(drop=True)

    X = df[FEATURE_COLUMNS].copy()
    y = df["weather_bucket"].copy()
    return X, y, df


def main():
    print("Loading and preparing data...")
    X, y, df = load_and_prepare()
    print(f"Rows after cleaning: {len(X)}")
    print(y.value_counts())

    # Encode target
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    # Stratified split so rare classes appear in both train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )

    # Preprocessing: scale numeric features (helps KNN/LogReg; harmless for trees)
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), FEATURE_COLUMNS)]
    )
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        ),
        "knn": KNeighborsClassifier(n_neighbors=9, weights="distance"),
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
    }

    metrics = {}
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_test_proc)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        print(f"{name}: accuracy={acc:.4f} macro_f1={f1:.4f}")
        print(
            classification_report(
                y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
            )
        )

        metrics[name] = {
            "accuracy": round(acc, 4),
            "macro_precision": round(prec, 4),
            "macro_recall": round(rec, 4),
            "macro_f1": round(f1, 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        if name == "xgboost":
            # Native XGBoost serialization is more robust across OS/versions
            # than pickling the sklearn wrapper.
            model.save_model(f"{MODELS_DIR}/model_xgboost.json")
        else:
            joblib.dump(model, f"{MODELS_DIR}/model_{name}.joblib")

    joblib.dump(preprocessor, f"{MODELS_DIR}/preprocessor.joblib")
    joblib.dump(label_encoder, f"{MODELS_DIR}/label_encoder.joblib")

    with open(f"{MODELS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(f"{MODELS_DIR}/feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    # Save feature ranges (for building sensible UI sliders)
    ranges = {
        col: {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "mean": float(X[col].mean()),
        }
        for col in FEATURE_COLUMNS
    }
    with open(f"{MODELS_DIR}/feature_ranges.json", "w") as f:
        json.dump(ranges, f, indent=2)

    # Save class distribution + location list for the app
    locations = (
        df[["location_name", "region", "latitude", "longitude"]]
        .drop_duplicates(subset=["location_name"])
        .sort_values("location_name")
    )
    locations.to_csv(f"{MODELS_DIR}/locations.csv", index=False)

    best_model = max(metrics, key=lambda k: metrics[k]["macro_f1"])
    print(f"\nBest model by macro F1: {best_model} ({metrics[best_model]['macro_f1']})")
    with open(f"{MODELS_DIR}/best_model.json", "w") as f:
        json.dump({"best_model": best_model}, f)

    print("\nDone. Artifacts saved to ./models/")


if __name__ == "__main__":
    main()
