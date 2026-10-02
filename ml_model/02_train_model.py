"""
02_train_model.py
Trains a Machine Learning model to predict Pressure Drop (Delta P)
based on cleaned vision and physics features.
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_pressure_model(input_file="cleaned_data.xlsx", model_file="pressure_model.pkl"):
    print("=== ML Phase 2: Model Training ===\n")

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run 01_data_cleaning.py first.")
        return

    # 1. Load Cleaned Data
    df = pd.read_excel(input_file)
    print(f"Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns.")

    if df.shape[0] < 2:
        print("\n[Warning] Not enough data to perform a meaningful train/test split.")
        print("The model will still be trained on available data, but metrics will be unreliable.")

    # 2. Define Features (X) and Target (y)
    # We drop 'Video Name' as it's just an identifier
    X = df.drop(columns=["Video Name", "Delta P (Pa)"])
    y = df["Delta P (Pa)"]

    # 3. Preprocessing Pipeline
    # Identify categorical and numerical columns
    categorical_cols = ["Material"]
    numerical_cols = [col for col in X.columns if col != "Material"]

    # Create a transformer: One-Hot Encode Material, keep numbers as is
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ]
    )

    # Create a pipeline that first preprocesses then trains a Random Forest
    # Random Forest is chosen for its ability to handle non-linear physical data
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # 4. Train/Test Split
    # Handle extremely small datasets: only split if we have at least 2 samples
    if df.shape[0] > 1:
        test_size = 0.2 if df.shape[0] > 5 else 0.1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        print("\n[Notice] Dataset too small to split. Training on the single available sample.")
        X_train, X_test, y_train, y_test = X, X, y, y

    # 5. Training
    print("Training Random Forest Regressor...")
    model_pipeline.fit(X_train, y_train)

    # 6. Evaluation
    y_pred = model_pipeline.predict(X_test)

    if len(y_test) > 0:
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print("\n--- Model Performance Metrics ---")
        print(f"Mean Absolute Error (MAE): {mae:.2f} Pa")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Pa")
        print(f"R-squared Score (R2): {r2:.4f}")
        print("----------------------------------\n")

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, color='blue', edgecolors='k', alpha=0.7)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel("Actual Delta P (Pa)")
        plt.ylabel("Predicted Delta P (Pa)")
        plt.title("Actual vs Predicted Pressure Drop")
        plt.grid(True)
        plt.savefig("model_performance.png")
        print("Performance plot saved as 'model_performance.png'")
    else:
        print("\n[Notice] Test set was empty due to small dataset size. Skipping metrics.")

    # 7. Save the trained model
    joblib.dump(model_pipeline, model_file)
    print(f"Trained model saved to: {model_file}")

if __name__ == "__main__":
    train_pressure_model()
