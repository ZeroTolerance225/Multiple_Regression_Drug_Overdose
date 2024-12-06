import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a backend that does not require a GUI
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor  # Fixing the import to xgboost

# Step 1: Load and Preprocess Data
data = pd.read_csv("VSRR_Provisional_Drug_Overdose_Death_Counts.csv", low_memory=False)
data.columns = data.columns.str.strip()
data['Data Value'] = pd.to_numeric(data['Data Value'], errors='coerce')

selected_indicators = [
    'Heroin (T40.1)',
    'Cocaine (T40.5)',
    'Psychostimulants with abuse potential (T43.6)',
    'Natural & semi-synthetic opioids (T40.2)',
    'Synthetic opioids, excl. methadone (T40.4)'
]

filtered_data = data[data['Indicator'].isin(selected_indicators)]
pivoted_data = filtered_data.pivot_table(
    index=['State', 'Year', 'Month'],
    columns='Indicator',
    values='Data Value'
).reset_index()

cleaned_data = pivoted_data.dropna()

# Define Features (X) and Target (y)
X = cleaned_data[
    ['Heroin (T40.1)', 'Cocaine (T40.5)', 'Psychostimulants with abuse potential (T43.6)', 'Natural & semi-synthetic opioids (T40.2)']
]
y = cleaned_data['Synthetic opioids, excl. methadone (T40.4)']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 2: Train XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

# Step 3: Make Predictions
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

# Step 4: Evaluate Model Performance
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

# Create the folder for saving plots if it doesn't exist
output_dir = "xgboost_plots"
os.makedirs(output_dir, exist_ok=True)

# Step 5: Plot Regression Results
# Predicted vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, color="blue", alpha=0.7, label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Ideal Fit")
plt.title("Synthetic Opioid Death Rates: Predicted vs Actual (XGBoost)")
plt.xlabel("Actual Deaths")
plt.ylabel("Predicted Deaths")
plt.legend()
plt.grid()
plt.savefig(f"{output_dir}/xgboost_predicted_vs_actual.png")

# Residuals vs Predicted Values
residuals = y_test - y_pred_test
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_test, residuals, color="green", alpha=0.7, label="Residuals vs Predicted")
plt.axhline(0, color='red', linestyle='--', label="Zero Residual Line")
plt.title("Residuals vs Predicted Values (XGBoost)")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend()
plt.grid()
plt.savefig(f"{output_dir}/xgboost_residuals_vs_predicted.png")

# Residual Distribution
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, color="purple", alpha=0.7, edgecolor="black", label="Residual Distribution")
plt.axvline(0, color='red', linestyle='--', label="Zero Line")
plt.title("Residual Distribution (XGBoost)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.legend()
plt.grid()
plt.savefig(f"{output_dir}/xgboost_residuals_distribution.png")

# Step 6: Output Results
output_summary = {
    "R-squared (Train)": r2_train,
    "R-squared (Test)": r2_test,
    "RMSE (Train)": rmse_train,
    "RMSE (Test)": rmse_test,
    "Feature Importances": xgb_model.feature_importances_.tolist(),
    "Plots Saved": [
        f"{output_dir}/xgboost_predicted_vs_actual.png",
        f"{output_dir}/xgboost_residuals_vs_predicted.png",
        f"{output_dir}/xgboost_residuals_distribution.png"
    ]
}

print("\nXGBoost Regression Summary:")
for key, value in output_summary.items():
    if isinstance(value, list):
        print(f"{key}:\n  {value}")
    else:
        print(f"{key}: {value}")
