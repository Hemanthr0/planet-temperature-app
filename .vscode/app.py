import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64

# Load model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Optional background image
def add_bg_from_local(image_file):
    try:
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
    except FileNotFoundError:
        st.warning("Background image not found.")

add_bg_from_local("planet_bg.jpg")  # Optional background image

st.title("🌍 Planet Temperature Estimator")
st.markdown("Enter the planet's characteristics below to estimate surface temperature using ML!")

# Sliders for user input
st.header("🔢 Predict from Input Sliders")
mass = st.slider("Planet Mass (Earth Masses)", 0.0, 10000.0, 1000.0)
radius = st.slider("Planet Radius (Earth Radii)", 0.0, 100.0, 10.0)
distance = st.slider("Distance from Star (AU)", 0.0, 10.0, 1.0)

# Format input and scale
input_data = np.array([[radius, mass, distance]])  # Order matters
input_scaled = scaler.transform(input_data)

if st.button("🚀 Predict Temperature"):
    predicted_temp = model.predict(input_scaled)[0]
    st.success(f"🌡️ Predicted Surface Temperature: **{predicted_temp:.2f} K**")

    # Comparison Plot
    st.subheader("📊 Comparison to Known Exoplanets")
    try:
        sample_data = pd.read_csv("known_exoplanets_with_temp.csv")
        plt.figure(figsize=(8, 5))
        plt.scatter(sample_data["Temperature"], sample_data["P_RADIUS_EST"], alpha=0.6, label="Known Planets")
        plt.scatter(predicted_temp, radius, color='red', label="Your Planet", s=100)
        plt.xlabel("Temperature (K)")
        plt.ylabel("Radius (Earth Radii)")
        plt.title("Predicted vs Known Planets")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
    except FileNotFoundError:
        st.error("known_exoplanets_with_temp.csv not found!")

# Batch prediction
st.header("📁 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV with columns: P_MASS_EST, P_RADIUS_EST, P_SEMI_MAJOR_AXIS_EST", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    required_cols = ["P_MASS_EST", "P_RADIUS_EST", "P_SEMI_MAJOR_AXIS_EST"]
    if all(col in data.columns for col in required_cols):
        inputs = data[["P_RADIUS_EST", "P_MASS_EST", "P_SEMI_MAJOR_AXIS_EST"]].to_numpy()
        inputs_scaled = scaler.transform(inputs)
        predictions = model.predict(inputs_scaled)
        data["Predicted_Temperature"] = predictions
        st.write(data)

        csv_download = data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", csv_download, "predicted_temperatures.csv", "text/csv")
    else:
        st.error("Missing required columns.")

