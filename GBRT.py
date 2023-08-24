import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import math

# Load data from Excel file
data = pd.read_excel('data_set2.xlsx')

# Assuming the file has columns 'Year' and 'Consumption'
X = data['Year'].values.reshape(-1, 1)  # Input feature: Year
y = data['Consumption'].values          # Target: Power Consumption

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# Generate random test indices (set to 10)
num_random_samples = 10  # Number of random samples
random_indices = np.random.choice(len(X_test), size=num_random_samples, replace=False)

# Use the model to predict power consumption for the random samples
predicted_consumption = gb_model.predict(X_test[random_indices])

# Calculate RMSE on test data
rmse = math.sqrt(mean_squared_error(y_test, gb_model.predict(X_test)))

# Calculate R-squared score
r2 = r2_score(y_test, gb_model.predict(X_test))

# Calculate NMAE and MAE
nmae = mean_absolute_error(y_test, gb_model.predict(X_test)) / np.mean(y)
mae = mean_absolute_error(y_test, gb_model.predict(X_test))

print(f"RMSE on test data: {rmse:.2f} kWh per capita")
print(f"NMAE: {nmae:.4f}")
print(f"MAE: {mae:.2f} kWh per capita")
print(f"R-squared score on test data: {r2:.2f}")


# Print predicted values for random test samples
for index, consumption in zip(random_indices, predicted_consumption):
    print(f"Year: {X_test[index][0]}, Actual Consumption: {y_test[index]:.2f} kWh per capita, Predicted Consumption: {consumption:.2f} kWh per capita")
