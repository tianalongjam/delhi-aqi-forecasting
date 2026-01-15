# 🌫️ Delhi AQI Forecasting Using Weather & Pollution Data (2025)

## 📌 Project Overview
Air pollution is a major public health challenge in Delhi, with Air Quality Index (AQI) levels frequently reaching hazardous levels, especially during winter months. This project focuses on **analyzing, understanding, and predicting AQI trends in Delhi** using high-resolution hourly weather and pollution data collected across multiple locations in 2025.

The goal of this project is to build a **robust machine learning pipeline** that leverages meteorological factors and historical pollution patterns to **forecast AQI levels ahead of time**, enabling early warnings and data-driven decision-making.

---

## 🎯 Project Objectives
- Perform **in-depth exploratory data analysis (EDA)** to understand temporal, spatial, and seasonal pollution patterns in Delhi.
- Study the **relationship between weather conditions and air pollution**, including the impact of wind speed, humidity, and temperature.
- Build and compare **machine learning models** to predict future AQI levels.
- Evaluate model performance under **normal and extreme pollution conditions**.
- Derive **actionable insights** relevant to public health, urban planning, and environmental monitoring.

---

## 📊 Dataset Description
- **Source:** Aggregated weather and AQI monitoring APIs  
- **Time Period:** January 2025 – December 2025  
- **Granularity:** Hourly observations  
- **Rows:** ~52,000  
- **Locations:** Multiple zones across Delhi (commercial, residential, industrial)

### Key Features
- **Weather:** Temperature, humidity, pressure, wind speed, weather condition
- **Pollution:** AQI, PM2.5, PM10, CO, NO₂
- **Spatial:** Location name, latitude, longitude
- **Temporal:** Date and time (IST)

---

## 🔍 Exploratory Data Analysis (EDA)
The EDA phase focuses on uncovering meaningful patterns such as:
- Hourly and seasonal AQI trends
- Location-wise pollution differences
- Winter vs summer pollution behavior
- Correlation between weather variables and AQI
- Identification of extreme pollution events

Key insights from EDA guide **feature engineering and model selection**.

---

## 🛠️ Feature Engineering
To improve predictive performance, several derived features are created:
- **Time-based features:** hour, day of week, month, season
- **Lag features:** previous AQI values (t-1, t-3, t-6 hours)
- **Rolling statistics:** rolling mean AQI and PM2.5
- **Interaction features:** wind speed × PM2.5, humidity × AQI
- **Location encoding:** spatial differentiation across Delhi zones

These features help the model capture **temporal dependence and non-linear relationships**.

---

## 🤖 Modeling Approach
The modeling strategy follows a progressive approach:
1. **Baseline Models:** Linear Regression, Decision Tree  
2. **Ensemble Models:** Random Forest, Gradient Boosting (XGBoost / LightGBM)  

Models are evaluated using:
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **R² Score**

Special attention is given to **model performance during high-pollution episodes**, where accurate prediction is most critical.

---

## 📈 Results & Evaluation
- Ensemble models significantly outperform baseline approaches.
- Historical AQI and PM2.5 levels emerge as the strongest predictors.
- Weather variables such as wind speed and humidity play a crucial role during winter pollution spikes.
- The model shows strong generalization across locations, with slightly higher error during extreme AQI events.

---

## 🧠 Key Insights
- Low wind speed and high humidity strongly contribute to severe AQI episodes.
- Industrial and high-traffic zones consistently show higher pollution levels.
- Temporal patterns (hour of day, season) are essential for accurate AQI forecasting.

---

## 🌍 Real-World Applications
- **Early warning systems** for residents and health authorities
- **Public health advisories** for vulnerable populations
- **Urban planning insights** for pollution mitigation
- Decision support for **policy makers and environmental agencies**

---

## ⚠️ Limitations
- Traffic density and industrial activity data are not included.
- Sudden pollution events (e.g., crop burning, festivals) are difficult to predict precisely.
- The model is trained on a single year of data.

---

## 🔮 Future Improvements
- Incorporate traffic and satellite data
- Extend forecasting horizon using deep learning (LSTM/GRU)
- Deploy an interactive **AQI forecasting dashboard**
- Expand the framework to other Indian cities

---

## 👨‍💻 Tech Stack
- Python, Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn, XGBoost / LightGBM  
