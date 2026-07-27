"""
AI-ML Assignment 6
Weather Condition Classification using SVM and Open-Meteo API
"""

import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# TASK 1: DATA COLLECTION AND UNDERSTANDING
# ============================================================

LATITUDE = 28.6139     # New Delhi
LONGITUDE = 77.2090
API_URL = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={LATITUDE}&longitude={LONGITUDE}"
    "&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m"
    "&forecast_days=7"
)


def fetch_weather_data(url):
    """
    Fetch hourly weather data from the Open-Meteo API and return it as a
    pandas DataFrame. Falls back to a locally generated (but realistically
    distributed) dataset only if the live API cannot be reached, so the
    notebook always runs end-to-end.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        hourly = data["hourly"]
        df = pd.DataFrame({
            "time": hourly["time"],
            "Temperature": hourly["temperature_2m"],
            "Relative_Humidity": hourly["relative_humidity_2m"],
            "Surface_Pressure": hourly["surface_pressure"],
            "Wind_Speed": hourly["wind_speed_10m"],
        })
        print("Live data successfully fetched from Open-Meteo API.")
        return df
    except Exception as e:
        print(f"Live API call failed ({e}). Using locally generated data "
              f"with the same structure as the Open-Meteo response so the "
              f"pipeline can still be demonstrated end-to-end.")
        rng = np.random.default_rng(42)
        n_hours = 24 * 7  # forecast_days=7
        times = pd.date_range("2026-07-27", periods=n_hours, freq="h")

        hour_of_day = times.hour.values
        # Diurnal temperature cycle typical of Delhi in late July (monsoon season)
        base_temp = 29 + 6 * np.sin((hour_of_day - 9) * np.pi / 12)
        temperature = base_temp + rng.normal(0, 1.5, n_hours)

        humidity = 75 - (temperature - 29) * 3 + rng.normal(0, 5, n_hours)
        humidity = np.clip(humidity, 30, 100)

        pressure = 1000 + rng.normal(0, 2, n_hours)
        wind_speed = np.abs(8 + rng.normal(0, 4, n_hours))

        df = pd.DataFrame({
            "time": times,
            "Temperature": np.round(temperature, 1),
            "Relative_Humidity": np.round(humidity, 1),
            "Surface_Pressure": np.round(pressure, 1),
            "Wind_Speed": np.round(wind_speed, 1),
        })
        return df


df = fetch_weather_data(API_URL)

print("\nFirst five records:")
print(df.head())

input_features = ["Temperature", "Relative_Humidity", "Surface_Pressure", "Wind_Speed"]
target_variable = "Weather_Class"
print(f"\nInput features: {input_features}")
print(f"Target variable: {target_variable}")

# Create target column: Warm if Temperature >= 25C else Cool
df["Weather_Class"] = np.where(df["Temperature"] >= 25, "Warm", "Cool")

print("\nClass distribution:")
print(df["Weather_Class"].value_counts())

# ============================================================
# TASK 2: DATA PREPROCESSING
# ============================================================

print("\nMissing values per column:")
print(df.isnull().sum())

# Remove unnecessary columns (timestamp not needed for modeling)
model_df = df.drop(columns=["time"])

# Encode target variable
le = LabelEncoder()
model_df["Weather_Class_Encoded"] = le.fit_transform(model_df["Weather_Class"])
print(f"\nLabel encoding mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

X = model_df[input_features]
y = model_df["Weather_Class_Encoded"]

# 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# TASK 3: MODEL DEVELOPMENT
# ============================================================

svm_model = SVC(kernel="rbf", random_state=42)
svm_model.fit(X_train_scaled, y_train)

y_pred = svm_model.predict(X_test_scaled)

# ============================================================
# TASK 4: MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM (RBF Kernel)")
plt.tight_layout()
plt.savefig("/home/claude/confusion_matrix.png", dpi=150)
plt.close()
print("\nConfusion matrix plot saved.")
