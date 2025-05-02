"""
Titanic Survival Classification Project
---------------------------------------
Author: Arona
Course: Machine Learning (IT4210B)
Description: This script builds a classification model to predict passenger survival on the Titanic.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Set visual style
sns.set(style='whitegrid')

# Step 1: Extract ZIP file (if not already extracted)
zip_path = 'Titanic/titanic.zip'  # relative to current working directory
extract_path = 'Titanic/extracted'

# Only extract if not already done
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ ZIP file extracted.")
else:
    print("ℹ️ ZIP already extracted.")

# Step 2: Load Dataset
csv_path = os.path.join(extract_path, 'train.csv')
df = pd.read_csv(csv_path)

# Step 3: EDA
print("Dataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# Visualizations
sns.countplot(x='Survived', data=df)
plt.title('Survival Distribution')
plt.savefig("survival_distribution.png")
plt.clf()

sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival by Gender')
plt.savefig("survival_by_gender.png")
plt.clf()

sns.histplot(df['Age'].dropna(), kde=True)
plt.title('Age Distribution')
plt.savefig("age_distribution.png")
plt.clf()

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.savefig("feature_correlation.png")
plt.clf()

# Step 4: Preprocessing
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'], inplace=True)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])  # female=0, male=1
df['Embarked'] = le.fit_transform(df['Embarked'])

X = df.drop('Survived', axis=1)
y = df['Survived']

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
