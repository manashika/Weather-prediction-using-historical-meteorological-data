"""
Weather Condition Predictor — Streamlit app
Predicts a bucketed weather condition (Clear / Partly Cloudy / Cloudy /
Mist-Fog / Light Rain / Heavy Rain / Severe Weather) from live meteorological
inputs, using models trained in train_models.py on the Indian Weather
Repository dataset.
"""

import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Weather Condition Predictor",
    page_icon="🌦️",
    layout="wide",
)

MODELS_DIR = "models"

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(f"{MODELS_DIR}/preprocessor.joblib")
    label_encoder = joblib.load(f"{MODELS_DIR}/label_encoder.joblib")
    xgb_model = XGBClassifier()
    xgb_model.load_model(f"{MODELS_DIR}/model_xgboost.json")

    models = {
        "Random Forest": joblib.load(f"{MODELS_DIR}/model_random_forest.joblib"),
        "XGBoost": xgb_model,
        "KNN": joblib.load(f"{MODELS_DIR}/model_knn.joblib"),
        "Logistic Regression": joblib.load(
            f"{MODELS_DIR}/model_logistic_regression.joblib"
        ),
    }
    with open(f"{MODELS_DIR}/feature_columns.json") as f:
        feature_columns = json.load(f)
    with open(f"{MODELS_DIR}/feature_ranges.json") as f:
        feature_ranges = json.load(f)
    with open(f"{MODELS_DIR}/metrics.json") as f:
        metrics = json.load(f)
    with open(f"{MODELS_DIR}/best_model.json") as f:
        best_model_key = json.load(f)["best_model"]
    locations = pd.read_csv(f"{MODELS_DIR}/locations.csv")

    name_map = {
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "knn": "KNN",
        "logistic_regression": "Logistic Regression",
    }

    return {
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "models": models,
        "feature_columns": feature_columns,
        "feature_ranges": feature_ranges,
        "metrics": {name_map[k]: v for k, v in metrics.items()},
        "best_model": name_map[best_model_key],
        "locations": locations,
    }


artifacts = load_artifacts()
CLASSES = list(artifacts["label_encoder"].classes_)

CONDITION_EMOJI = {
    "Clear": "☀️",
    "Partly Cloudy": "🌤️",
    "Cloudy": "☁️",
    "Mist/Fog": "🌫️",
    "Light Rain": "🌦️",
    "Heavy Rain": "🌧️",
    "Severe Weather": "⛈️",
}

CONDITION_COLOR = {
    "Clear": "#F5B041",
    "Partly Cloudy": "#85C1E9",
    "Cloudy": "#AAB7B8",
    "Mist/Fog": "#D5DBDB",
    "Light Rain": "#5DADE2",
    "Heavy Rain": "#2874A6",
    "Severe Weather": "#943126",
}

# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
st.sidebar.title("🌦️ Weather Inputs")
st.sidebar.caption(
    "Enter current meteorological readings, or pick a location to prefill "
    "with its historical average."
)

locations_df = artifacts["locations"]
loc_names = ["— Manual entry —"] + sorted(locations_df["location_name"].tolist())
selected_loc = st.sidebar.selectbox("Location (optional, prefills lat/long)", loc_names)

if selected_loc != "— Manual entry —":
    loc_row = locations_df[locations_df["location_name"] == selected_loc].iloc[0]
    default_lat, default_lon = float(loc_row["latitude"]), float(loc_row["longitude"])
    st.sidebar.caption(f"Region: {loc_row['region']}")
else:
    default_lat = artifacts["feature_ranges"]["latitude"]["mean"]
    default_lon = artifacts["feature_ranges"]["longitude"]["mean"]

ranges = artifacts["feature_ranges"]


def slider(label, key, step=0.1, fmt="%.1f"):
    r = ranges[key]
    return st.sidebar.slider(
        label,
        min_value=float(r["min"]),
        max_value=float(r["max"]),
        value=float(round(r["mean"], 1)),
        step=step,
        format=fmt,
    )


temperature = slider("Temperature (°C)", "temperature_celsius")
humidity = slider("Humidity (%)", "humidity", step=1.0, fmt="%.0f")
wind_kph = slider("Wind Speed (kph)", "wind_kph")
pressure_mb = slider("Pressure (mb)", "pressure_mb", step=0.5)
precip_mm = slider("Precipitation (mm)", "precip_mm")
cloud = slider("Cloud Cover (%)", "cloud", step=1.0, fmt="%.0f")
visibility_km = slider("Visibility (km)", "visibility_km")
uv_index = slider("UV Index", "uv_index", step=0.5)
gust_kph = slider("Gust Speed (kph)", "gust_kph")
pm25 = slider("Air Quality — PM2.5", "air_quality_PM2.5", step=1.0, fmt="%.0f")
pm10 = slider("Air Quality — PM10", "air_quality_PM10", step=1.0, fmt="%.0f")
latitude = st.sidebar.number_input("Latitude", value=default_lat, format="%.2f")
longitude = st.sidebar.number_input("Longitude", value=default_lon, format="%.2f")

model_choice = st.sidebar.radio(
    "Model to use",
    list(artifacts["models"].keys()),
    index=list(artifacts["models"].keys()).index(artifacts["best_model"]),
)
st.sidebar.caption(f"⭐ Best model on test data: **{artifacts['best_model']}**")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("🌦️ Weather Condition Predictor")
st.caption(
    "Predicts the general weather condition from live meteorological readings, "
    "using models trained on the Indian Weather Repository dataset "
    "(34,466 station-day records across 543 Indian locations)."
)

tab_predict, tab_compare, tab_explore, tab_about = st.tabs(
    ["🔮 Predict", "📊 Compare Models", "🗺️ Explore Data", "ℹ️ About"]
)

input_row = pd.DataFrame(
    [
        {
            "temperature_celsius": temperature,
            "humidity": humidity,
            "wind_kph": wind_kph,
            "pressure_mb": pressure_mb,
            "precip_mm": precip_mm,
            "cloud": cloud,
            "visibility_km": visibility_km,
            "uv_index": uv_index,
            "gust_kph": gust_kph,
            "air_quality_PM2.5": pm25,
            "air_quality_PM10": pm10,
            "latitude": latitude,
            "longitude": longitude,
        }
    ]
)[artifacts["feature_columns"]]

X_proc = artifacts["preprocessor"].transform(input_row)

with tab_predict:
    model = artifacts["models"][model_choice]
    pred_idx = model.predict(X_proc)[0]
    pred_label = artifacts["label_encoder"].inverse_transform([pred_idx])[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_proc)[0]
    else:
        proba = np.zeros(len(CLASSES))
        proba[pred_idx] = 1.0

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"## {CONDITION_EMOJI.get(pred_label,'')} {pred_label}")
        st.markdown(f"**Model used:** {model_choice}")
        confidence = proba[pred_idx] * 100
        st.metric("Confidence", f"{confidence:.1f}%")

    with col2:
        prob_df = pd.DataFrame(
            {"Condition": CLASSES, "Probability": proba}
        ).sort_values("Probability", ascending=True)
        fig = px.bar(
            prob_df,
            x="Probability",
            y="Condition",
            orientation="h",
            color="Condition",
            color_discrete_map=CONDITION_COLOR,
            text=prob_df["Probability"].apply(lambda p: f"{p*100:.1f}%"),
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title="Probability",
            yaxis_title="",
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### All model predictions for these inputs")
    rows = []
    for name, m in artifacts["models"].items():
        idx = m.predict(X_proc)[0]
        label = artifacts["label_encoder"].inverse_transform([idx])[0]
        conf = m.predict_proba(X_proc)[0][idx] * 100 if hasattr(m, "predict_proba") else None
        rows.append(
            {
                "Model": name,
                "Prediction": f"{CONDITION_EMOJI.get(label,'')} {label}",
                "Confidence": f"{conf:.1f}%" if conf is not None else "—",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab_compare:
    st.markdown("### Model performance on held-out test data (25% split)")
    metrics = artifacts["metrics"]
    comp_df = pd.DataFrame(
        [
            {
                "Model": name,
                "Accuracy": m["accuracy"],
                "Macro Precision": m["macro_precision"],
                "Macro Recall": m["macro_recall"],
                "Macro F1": m["macro_f1"],
            }
            for name, m in metrics.items()
        ]
    ).sort_values("Macro F1", ascending=False)

    st.dataframe(
        comp_df.style.format(
            {c: "{:.1%}" for c in ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]}
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        comp_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text_auto=".1%",
    )
    fig.update_layout(yaxis_tickformat=".0%", height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Confusion matrix")
    cm_model = st.selectbox("Model", list(metrics.keys()), key="cm_model")
    cm = np.array(metrics[cm_model]["confusion_matrix"])
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=CLASSES,
            y=CLASSES,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
        )
    )
    fig_cm.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=500,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.info(
        "**Why these numbers are trustworthy:** the target is a genuine categorical "
        "weather condition, not a continuous variable artificially cut into bins. "
        "Macro-averaged metrics are shown (not just accuracy) because classes are "
        "imbalanced — 'Clear' is ~45% of rows, 'Severe Weather' is <1%."
    )

with tab_explore:
    st.markdown("### Dataset overview")
    df_full = pd.read_csv("data/IndianWeatherRepository.csv")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total records", f"{len(df_full):,}")
    c2.metric("Unique locations", df_full["location_name"].nunique())
    c3.metric("Date range", "Aug–Oct 2023")

    st.markdown("#### Weather condition distribution")

    def bucket_condition(c):
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

    df_full["weather_bucket"] = df_full["condition_text"].apply(bucket_condition)
    dist = df_full["weather_bucket"].value_counts().reset_index()
    dist.columns = ["Condition", "Count"]
    fig_dist = px.bar(
        dist,
        x="Condition",
        y="Count",
        color="Condition",
        color_discrete_map=CONDITION_COLOR,
    )
    fig_dist.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("#### Locations on the map")
    map_df = df_full.groupby("location_name").agg(
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        avg_temp=("temperature_celsius", "mean"),
        region=("region", "first"),
    ).reset_index()

    fig_map = px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        color="avg_temp",
        hover_name="location_name",
        hover_data=["region", "avg_temp"],
        color_continuous_scale="RdYlBu_r",
        zoom=3.3,
        height=500,
    )
    fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("#### Feature correlation with weather condition")
    st.caption("Box plots showing how key readings vary across condition buckets")
    feat_for_box = st.selectbox(
        "Feature", artifacts["feature_columns"], index=0, key="box_feat"
    )
    fig_box = px.box(
        df_full[df_full["weather_bucket"] != "Other"],
        x="weather_bucket",
        y=feat_for_box,
        color="weather_bucket",
        color_discrete_map=CONDITION_COLOR,
        category_orders={"weather_bucket": CLASSES},
    )
    fig_box.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_box, use_container_width=True)

with tab_about:
    st.markdown(
        """
        ### About this app

        This app is an extension of a B.Sc. Data Science capstone project,
        *"Weather Prediction Using Historical Meteorological Data."*
        It upgrades the original work in a few concrete ways:

        1. **Genuine classification target.** The original project predicted
           temperature by cutting a continuous variable into 9 arbitrary bins.
           This app predicts the actual weather condition category
           (Clear, Rain, Fog, etc.), which is what "weather condition
           classification" should mean.
        2. **Class imbalance handled properly.** Macro-averaged precision/recall/F1
           are reported alongside accuracy, and models use `class_weight="balanced"`
           where supported, since 45% of the data is "Clear" and <1% is "Severe
           Weather."
        3. **Deployed, interactive interface** instead of a notebook with
           `input()` prompts.

        **Models:** Random Forest, XGBoost, K-Nearest Neighbors, Logistic
        Regression — all trained on the same 13 meteorological/geographic
        features, with a stratified 75/25 train-test split.

        **Dataset:** [Indian Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/indian-weather-repository-daily-snapshot)
        (Kaggle), 34,466 records, 543 locations across India, Aug–Oct 2023.

        **Limitations (same spirit as the original report):**
        - This is *nowcasting* (classifying current conditions from current
          readings), not forecasting future weather from past data — the
          dataset doesn't have enough sequential history per location for
          proper time-series forecasting.
        - "Severe Weather" (thunderstorms, snow) is rare in this dataset
          (<1% of rows), so predictions for that class are less reliable —
          visible in the confusion matrix.
        - The model's accuracy partly reflects that cloud cover and
          precipitation are near-direct proxies for the target label; a
          true *forecasting* system would need to predict conditions
          *before* those readings are available.
        """
    )
