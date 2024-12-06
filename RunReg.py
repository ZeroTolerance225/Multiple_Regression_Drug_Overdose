import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a backend that does not require a GUI
import matplotlib.pyplot as plt
from scipy.stats import t
from statsmodels.stats.diagnostic import het_white

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

X = cleaned_data[
    ['Heroin (T40.1)', 'Cocaine (T40.5)', 'Psychostimulants with abuse potential (T43.6)', 'Natural & semi-synthetic opioids (T40.2)']
].values
y = cleaned_data['Synthetic opioids, excl. methadone (T40.4)'].values

X = np.column_stack((np.ones(X.shape[0]), X))

# Step 2: Compute Regression Coefficients
beta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)

# Step 3: Make Predictions
y_pred = np.dot(X, beta)

# Step 4: Calculate Residuals
residuals = y - y_pred

# Step 5: Evaluate Model Performance
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_residuals = np.sum(residuals ** 2)
r_squared = 1 - (ss_residuals / ss_total)
adjusted_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - X.shape[1])
rss = ss_residuals
tss = ss_total

# Step 6: Calculate Additional Metrics
n = len(y)
p = X.shape[1] - 1
f_statistic = (r_squared / p) / ((1 - r_squared) / (n - p - 1))
se_beta = np.sqrt(np.diag(np.linalg.inv(np.dot(X.T, X)) * (ss_residuals / (n - p - 1))))
t_statistics = beta / se_beta
p_values = [2 * (1 - t.cdf(abs(t_stat), df=n - p - 1)) for t_stat in t_statistics]
white_test_stat, white_p_value, _, _ = het_white(residuals, X)

# Step 7: Plot Regression Results
import os

# Create the folder for saving plots if it doesn't exist
output_dir = "regression_plots"
os.makedirs(output_dir, exist_ok=True)

# Predicted vs Actual Plot
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, color="blue", label="Predicted vs Actual", alpha=0.7)
plt.plot([min(y), max(y)], [min(y), max(y)], color="red", linestyle="--", label="Ideal Fit")
plt.title("Synthetic Opioid Death Rates: Predicted vs Actual")
plt.xlabel("Actual Deaths")
plt.ylabel("Predicted Deaths")
plt.legend()
plt.grid()
output_file_actual_vs_predicted = f"{output_dir}/predicted_vs_actual.png"
plt.savefig(output_file_actual_vs_predicted)

# Residuals vs Predicted Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color="green", alpha=0.7)
plt.axhline(0, color='red', linestyle='--', label="Zero Residual Line")
plt.title("Residuals vs Predicted Values")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend()
plt.grid()
output_file_residuals = f"{output_dir}/residuals_vs_predicted.png"
plt.savefig(output_file_residuals)


# Step 9: Output All Results
output_summary = {
    "R-squared": r_squared,
    "Adjusted R-squared": adjusted_r_squared,
    "F-statistic": f_statistic,
    "Residual Sum of Squares (RSS)": rss,
    "Total Sum of Squares (TSS)": tss,
    "White's Test Statistic": white_test_stat,
    "White's Test p-Value": f"{white_p_value:.4e}",
    "Coefficients (Beta)": beta.tolist(),
    "Standard Errors (SE)": se_beta.tolist(),
    "T-Statistics": t_statistics.tolist(),
    "P-Values": p_values,
    "Mean Residual": np.mean(residuals),
    "Residual Standard Deviation": np.std(residuals),
    "Plot Saved As": "predicted_vs_actual.png"
}

# Display results in a readable format
print("\nFinal Regression Summary:")
for key, value in output_summary.items():
    if isinstance(value, list):
        print(f"{key}:\n  {value}")
    else:
        print(f"{key}: {value}")

print("\nWhite's Test Results:")
print(f"  Test Statistic: {white_test_stat:.4f}")
print(f"  p-Value: {white_p_value:.4e}")
if white_p_value < 0.05:
    print("  Conclusion: Evidence of heteroscedasticity (variance of residuals is not constant).")
else:
    print("  Conclusion: No evidence of heteroscedasticity (residual variance is constant).")
