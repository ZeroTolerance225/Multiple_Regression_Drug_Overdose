import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a backend that does not require a GUI
import matplotlib.pyplot as plt
from scipy.stats import t

# Step 1: Load and Preprocess Data
print("Step 1: Starting data loading and preprocessing.")

# Load the dataset
print("Loading dataset...")
data = pd.read_csv("VSRR_Provisional_Drug_Overdose_Death_Counts.csv", low_memory=False)
print("Dataset loaded successfully.")

# Strip extra spaces in column names
print("Stripping extra spaces in column names...")
data.columns = data.columns.str.strip()
print("Column names cleaned.")

# Convert 'Data Value' to numeric, coercing non-numeric to NaN
print("Converting 'Data Value' to numeric...")
data['Data Value'] = pd.to_numeric(data['Data Value'], errors='coerce')
print("Conversion completed. Any non-numeric entries were coerced to NaN.")

# Define selected indicators
selected_indicators = [
    'Heroin (T40.1)',
    'Cocaine (T40.5)',
    'Psychostimulants with abuse potential (T43.6)',
    'Natural & semi-synthetic opioids (T40.2)',
    'Synthetic opioids, excl. methadone (T40.4)'
]
print(f"Selected indicators: {selected_indicators}")

# Filter and pivot the dataset
print("Filtering dataset for selected indicators...")
filtered_data = data[data['Indicator'].isin(selected_indicators)]
print(f"Filtered dataset contains {len(filtered_data)} rows.")

print("Pivoting dataset...")
pivoted_data = filtered_data.pivot_table(
    index=['State', 'Year', 'Month'],
    columns='Indicator',
    values='Data Value'
).reset_index()
print(f"Pivoted dataset shape: {pivoted_data.shape}")

# Drop rows with missing data
print("Dropping rows with missing data...")
cleaned_data = pivoted_data.dropna()
print(f"Cleaned dataset contains {len(cleaned_data)} rows.")

# Separate predictors and response variable
print("Separating predictors and response variable...")
X = cleaned_data[
    ['Heroin (T40.1)', 'Cocaine (T40.5)', 'Psychostimulants with abuse potential (T43.6)', 'Natural & semi-synthetic opioids (T40.2)']
].values
y = cleaned_data['Synthetic opioids, excl. methadone (T40.4)'].values
print(f"Predictors shape: {X.shape}, Response variable shape: {y.shape}")

# Add an intercept to the predictors
print("Adding intercept to predictors...")
X = np.column_stack((np.ones(X.shape[0]), X))
print("Intercept added. Final predictors shape:", X.shape)

print("Step 1 completed successfully.")
print("---------------------------------------------------------------------------")


# Step 2: Compute Regression Coefficients
print("Step 2: Computing regression coefficients...")

# Compute beta using the formula beta = (X^T X)^(-1) X^T y
beta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)

# Print beta values for debugging
print("Regression coefficients (beta):")
print(beta)

print("Step 2 completed successfully.")
print("---------------------------------------------------------------------------")

# Step 3: Make Predictions
print("Step 3: Making predictions...")

# Compute predicted values using y_pred = X @ beta
y_pred = np.dot(X, beta)

# Print a summary of predictions for debugging
print("Predictions made successfully.")
print(f"First 10 predicted values: {y_pred[:10]}")

print("Step 3 completed successfully.")


print("---------------------------------------------------------------------------")
# Step 4: Calculate Residuals
print("Step 4: Calculating residuals...")

# Compute residuals as the difference between actual and predicted values
residuals = y - y_pred

# Print a summary of residuals for debugging
print("Residuals calculated successfully.")
print(f"First 10 residuals: {residuals[:10]}")
print(f"Mean residual: {np.mean(residuals):.4f}, Standard deviation: {np.std(residuals):.4f}")

print("Step 4 completed successfully.")


print("---------------------------------------------------------------------------")
# Step 5: Evaluate Model Performance
print("Step 5: Evaluating model performance...")

# Compute total sum of squares (SS_total) and residual sum of squares (SS_residuals)
ss_total = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
ss_residuals = np.sum(residuals ** 2)  # Residual sum of squares

# Calculate R-squared
r_squared = 1 - (ss_residuals / ss_total)

# Print R-squared value for debugging
print(f"R-squared: {r_squared:.4f}")
print("Model explains {:.2f}% of the variance in the data.".format(r_squared * 100))

print("Step 5 completed successfully.")


print("---------------------------------------------------------------------------")
# Step 6: Calculate Additional Metrics
print("Step 6: Calculating F-statistic, T-statistics, and p-values...")

# Degrees of freedom
n = len(y)  # Number of observations
p = X.shape[1] - 1  # Number of predictors (excluding intercept)

# F-statistic
f_statistic = (r_squared / p) / ((1 - r_squared) / (n - p - 1))

# Standard error of beta coefficients
se_beta = np.sqrt(np.diag(np.linalg.inv(np.dot(X.T, X)) * (ss_residuals / (n - p - 1))))

# T-statistics for each coefficient
t_statistics = beta / se_beta

# P-values for T-statistics
p_values = [2 * (1 - t.cdf(abs(t_stat), df=n - p - 1)) for t_stat in t_statistics]

# Print results for debugging
print(f"F-statistic: {f_statistic:.4f}")
print("Coefficients (Beta):", beta)
print("Standard Errors (SE):", se_beta)
print("T-statistics:", t_statistics)
print("P-values:", p_values)

print("Step 6 completed successfully.")


print("---------------------------------------------------------------------------")
# Step 7: Print Results
print("Step 7: Printing results...")

# Summary of regression results
print("\nRegression Summary:")
print(f"R-squared: {r_squared:.4f}")
print(f"F-statistic: {f_statistic:.4f}\n")

print("Coefficients (Beta):")
for i, coef in enumerate(beta):
    print(f"  Beta[{i}] = {coef:.4f}")

print("\nStandard Errors (SE):")
for i, se in enumerate(se_beta):
    print(f"  SE[{i}] = {se:.4f}")

print("\nT-Statistics:")
for i, t_stat in enumerate(t_statistics):
    print(f"  t[{i}] = {t_stat:.4f}")

print("\nP-Values:")
for i, p_val in enumerate(p_values):
    print(f"  p[{i}] = {p_val:.4e}")

print("\nResiduals Summary:")
print(f"Mean residual: {np.mean(residuals):.4f}")
print(f"Residual standard deviation: {np.std(residuals):.4f}")

print("\nStep 7 completed successfully.")


print("---------------------------------------------------------------------------")
# Step 8: Plot Regression Results
print("Step 8: Plotting regression results...")

# Scatter plot of actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, color="blue", label="Predicted vs Actual", alpha=0.7)

# Plot ideal fit line (y = y_pred)
plt.plot([min(y), max(y)], [min(y), max(y)], color="red", linestyle="--", label="Ideal Fit")

# Add plot labels and title
plt.title("Synthetic Opioid Death Rates: Predicted vs Actual")
plt.xlabel("Actual Deaths")
plt.ylabel("Predicted Deaths")
plt.legend()
plt.grid()

# Save the plot as an image
output_file = "predicted_vs_actual.png"
plt.savefig(output_file)

print(f"Plot saved successfully as '{output_file}'.")
print("Step 8 completed successfully.")


print("---------------------------------------------------------------------------")
# Step 9: Output All Results
print("Step 9: Outputting all results...")

# Summary of the regression results
output_summary = {
    "R-squared": r_squared,
    "F-statistic": f_statistic,
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

print("Step 9 completed successfully.")
