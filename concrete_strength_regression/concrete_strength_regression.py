"""
Concrete Compressive Strength Regression Project
-------------------------------------------------
Author: [Your Name]
Course: Machine Learning (IT4210B)
Description: This script builds and evaluates a regression model to predict 
             the compressive strength of concrete based on its ingredients.
"""

# Step 1: Import Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure plot aesthetics
sns.set(style='whitegrid')

# Step 2: Load Dataset
csv_path = os.path.join('extract', 'Concrete_Data.xls')  # Updated path
df = pd.read_csv(csv_path)

# Step 3: Exploratory Data Analysis (EDA)
print("Dataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# Plot target distribution
sns.histplot(df["Concrete compressive strength (MPa) "], kde=True)
plt.title("Distribution of Compressive Strength")
plt.xlabel("Strength (MPa)")
plt.savefig("strength_distribution.png")
plt.clf()

# Heatmap to check correlations
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.savefig("correlation_matrix.png")
plt.clf()

# Step 4: Data Preprocessing
# Features and target separation
X = df.drop("Concrete compressive strength (MPa) ", axis=1)
y = df["Concrete compressive strength (MPa) "]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 6: Model Selection and Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Prediction and Evaluation
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")
