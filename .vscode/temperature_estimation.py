import pandas as pd

# Load your full dataset
df = pd.read_csv('phl_exoplanet_catalog_2019.csv')  # Replace with actual file name

# Select key features
df_clean = df[['P_NAME', 'P_RADIUS_EST', 'P_MASS_EST', 'P_SEMI_MAJOR_AXIS_EST']].copy()

# Drop rows with any missing values in selected columns
df_clean = df_clean.dropna()

# Rename for clarity
df_clean.columns = ['Name', 'Radius', 'Mass', 'Distance']

# Show a preview
print(df_clean.head())

# Assumed albedo (Earth-like)
albedo = 0.3

# Calculate temperature using simplified blackbody model
df_clean['Temperature'] = 278 * ((1 - albedo) / (df_clean['Distance'] ** 2)) ** 0.25

# Show updated dataset
print(df_clean[['Name', 'Distance', 'Temperature']].head())

#Step 3: Train Machine Learning Models to Predict Temperature
#We'll train models using:

#Features: Mass, Radius, Distance

#Target: Temperature

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define input features and target
X = df_clean[['Mass', 'Radius', 'Distance']]
y = df_clean['Temperature']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
import joblib
joblib.dump(rf, 'random_forest_model.pkl')

# Evaluate
print("Linear Regression MAE:", mean_absolute_error(y_test, y_pred_lr))
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))

print("Random Forest MAE:", mean_absolute_error(y_test, y_pred_rf))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Temperature (K)')
plt.ylabel('Predicted Temperature (K)')
plt.title('Random Forest Prediction vs Actual')
plt.grid(True)
plt.show()

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64

# Load trained model
model = joblib.load("random_forest_model.pkl")

# Set background image using CSS
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("planet_bg.jpg")  # 🔁 Replace with your own image!

st.title("🌍 Planet Temperature Estimator")

st.markdown("Enter the planet's characteristics below to estimate its surface temperature using ML!")

# --- Sliders for prediction ---
st.header("🔢 Predict from Input Sliders")
mass = st.slider("Planet Mass (in Earth Masses)", 0.0, 10000.0, 1000.0)
radius = st.slider("Planet Radius (in Earth Radii)", 0.0, 100.0, 10.0)
distance = st.slider("Distance from Star (AU)", 0.0, 10.0, 1.0)

input_data = np.array([[mass, radius, distance]])

if st.button("🚀 Predict Temperature"):
    predicted_temp = model.predict(input_data)[0]
    st.success(f"🌡️ Predicted Surface Temperature: **{predicted_temp:.2f} K**")

    # Sample comparison plot
    st.subheader("📊 Comparison to Known Exoplanets")
    sample_data = pd.read_csv("known_exoplanets_sample.csv")  # 👈 Make sure you have this!
    plt.figure(figsize=(8, 5))
    plt.scatter(sample_data["P_TEMP"], sample_data["P_RADIUS_EST"], label="Known Planets", alpha=0.6)
    plt.scatter(predicted_temp, radius, color='red', label="Your Planet", s=100)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Radius (Earth radii)")
    plt.title("Predicted vs Known Planets")
    plt.legend()
    st.pyplot(plt)

# --- Batch Prediction ---
st.header("📁 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV with columns: P_MASS_EST, P_RADIUS_EST, P_SEMI_MAJOR_AXIS_EST", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    required_cols = ["P_MASS_EST", "P_RADIUS_EST", "P_SEMI_MAJOR_AXIS_EST"]
    if all(col in data.columns for col in required_cols):
        predictions = model.predict(data[required_cols])
        data["Predicted_Temperature"] = predictions
        st.write(data)

        csv_download = data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", csv_download, "predicted_temperatures.csv", "text/csv")
    else:
        st.error("Missing required columns in uploaded file.")

