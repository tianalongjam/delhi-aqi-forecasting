# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from xgboost import XGBRegressor

# import joblib

# feature_columns = joblib.load("feature_columns.pkl")


# st.set_page_config(page_title="Delhi AQI Predictor", layout="wide")

# st.title("🌫️ Delhi AQI Forecasting Dashboard")

# # Load processed data
# df = pd.read_csv("processed_data.csv")

# # Sidebar
# st.sidebar.header("Input Parameters")

# location_cols = [col for col in df.columns if col.startswith("location_")]
# selected_location = st.sidebar.selectbox("Select Location", location_cols)

# hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
# temp = st.sidebar.slider("Temperature (°C)", 0, 50, 25)
# humidity = st.sidebar.slider("Humidity (%)", 10, 100, 50)
# wind = st.sidebar.slider("Wind Speed (kph)", 0.0, 20.0, 5.0)
# pm25 = st.sidebar.slider("PM2.5", 0.0, 500.0, 150.0)
# pm10 = st.sidebar.slider("PM10", 0.0, 500.0, 200.0)

# model = XGBRegressor()
# model.load_model("xgb_model.json")

# # Create input row
# input_data = df.iloc[0:1].copy()
# input_data[:] = 0

# input_data["hour"] = hour
# input_data["temp_c"] = temp
# input_data["humidity"] = humidity
# input_data["windspeed_kph"] = wind
# input_data["pm2_5"] = pm25
# input_data["pm10"] = pm10
# input_data[selected_location] = 1

# # Prediction
# prediction = model.predict(input_data)[0]

# st.metric("Predicted AQI", round(prediction, 2))

# # AQI Category
# def aqi_category(aqi):
#     if aqi <= 50:
#         return "Good 🟢"
#     elif aqi <= 100:
#         return "Moderate 🟡"
#     elif aqi <= 200:
#         return "Poor 🟠"
#     elif aqi <= 300:
#         return "Very Poor 🔴"
#     else:
#         return "Severe ⚫"

# st.subheader(f"AQI Category: {aqi_category(prediction)}")

# st.markdown("""
# ### 📌 About
# This dashboard predicts AQI based on weather and pollution parameters using an XGBoost model trained on Delhi's 2025 data.
# """)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Delhi AQI Predictor",
    layout="wide"
)

st.title("🌫️ Delhi AQI Forecasting Dashboard")
st.markdown(
    "Predict Air Quality Index (AQI) using weather and pollution parameters "
    "with a machine learning model trained on Delhi's 2025 data."
)

# --------------------------------------------------
# Load data (for UI only, NOT for prediction)
# --------------------------------------------------
df = pd.read_csv("processed_data.csv")

# --------------------------------------------------
# Load model and feature schema
# --------------------------------------------------
model = XGBRegressor()
model.load_model("xgb_model.json")

feature_columns = joblib.load("feature_columns.pkl")

# --------------------------------------------------
# Sidebar inputs
# --------------------------------------------------
st.sidebar.header("🔧 Input Parameters")

hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
temp = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 30.0)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
wind = st.sidebar.slider("Wind Speed (kph)", 0.0, 30.0, 5.0)
pm25 = st.sidebar.slider("PM2.5 (µg/m³)", 0.0, 500.0, 80.0)
pm10 = st.sidebar.slider("PM10 (µg/m³)", 0.0, 500.0, 120.0)

# Location selection (based on training data)
location_cols = [c for c in feature_columns if c.startswith("location_")]
locations = [c.replace("location_", "") for c in location_cols]
selected_location = st.sidebar.selectbox("Location", locations)

# --------------------------------------------------
# Build input row (CRITICAL PART)
# --------------------------------------------------
input_data = pd.DataFrame(
    np.zeros((1, len(feature_columns))),
    columns=feature_columns
)

# Fill numeric features
input_data["hour"] = hour
input_data["temp_c"] = temp
input_data["humidity"] = humidity
input_data["windspeed_kph"] = wind
input_data["pm2_5"] = pm25
input_data["pm10"] = pm10

# One-hot encode location
location_col = f"location_{selected_location}"
if location_col in input_data.columns:
    input_data[location_col] = 1

# Force numeric dtype (important for XGBoost)
input_data = input_data.astype(np.float64)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
prediction = model.predict(input_data)[0]

st.metric("📊 Predicted AQI", round(prediction, 2))

# --------------------------------------------------
# AQI Category
# --------------------------------------------------
def aqi_category(aqi):
    if aqi <= 50:
        return "Good 🟢"
    elif aqi <= 100:
        return "Moderate 🟡"
    elif aqi <= 200:
        return "Poor 🟠"
    elif aqi <= 300:
        return "Very Poor 🔴"
    else:
        return "Severe ⚫"

st.subheader(f"AQI Category: {aqi_category(prediction)}")

# --------------------------------------------------
# Info section
# --------------------------------------------------
st.markdown(
    """
### 📌 About This Project
- **Model:** XGBoost Regressor (tuned)
- **Data:** Hourly weather & pollution data for Delhi (2025)
- **Features:** Weather, pollutants, time, and location
- **Explainability:** SHAP used during analysis phase

Built as a **portfolio-grade ML project** for internship applications.
"""
)
