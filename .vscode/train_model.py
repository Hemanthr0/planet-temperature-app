import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('known_exoplanets_sample.csv')

# Select and clean features
df_clean = df[['P_RADIUS_EST', 'P_MASS_EST', 'P_SEMI_MAJOR_AXIS_EST']].dropna()

# Calculate temperature using semi-major axis as distance
albedo = 0.3
df_clean['Temperature'] = 278 * ((1 - albedo) / (df_clean['P_SEMI_MAJOR_AXIS_EST'] ** 2)) ** 0.25

# Save updated dataset for use in app.py
df_clean.to_csv('known_exoplanets_with_temp.csv', index=False)

# Split data
X = df_clean[['P_RADIUS_EST', 'P_MASS_EST', 'P_SEMI_MAJOR_AXIS_EST']]
y = df_clean['Temperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(rf, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Evaluate model
y_pred_rf = rf.predict(X_test_scaled)
print("Random Forest MAE:", mean_absolute_error(y_test, y_pred_rf))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))

# Optional: show scatter plot
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Actual Temperature (K)')
plt.ylabel('Predicted Temperature (K)')
plt.title('Random Forest Prediction vs Actual')
plt.grid(True)
plt.show()
