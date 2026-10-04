"""
ml_model/02_train_model.py
Trains a Random Forest Regressor to predict pressure drop from cleaned data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_pressure_model(input_file="cleaned_data.xlsx", model_file="pressure_model.pkl"):
    print(f"Loading cleaned data from {input_file}...")
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if df.empty:
        print("Error: Cleaned dataset is empty.")
        return

    # 1. Define Features and Target
    target_col = 'Delta P (Pa)'
    if target_col not in df.columns:
        print(f"Error: Target {target_col} not found in dataset.")
        return

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify feature types
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"Categorical features: {categorical_cols}")
    print(f"Numerical features: {numeric_cols}")

    # 2. Preprocessing Pipeline
    # One-Hot Encoding for materials, Scaling for numbers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    # 3. Model Pipeline
    # Combine preprocessing and regressor into one object
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    # 5. Train Model
    print("Training Random Forest model...")
    pipeline.fit(X_train, y_train)

    # 6. Evaluation
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n--- MODEL PERFORMANCE ---")
    print(f"MAE:  {mae:.2f} Pa")
    print(f"RMSE: {rmse:.2f} Pa")
    print(f"R2:   {r2:.4f}")
    print("-------------------------")

    # 7. Visualization: Actual vs Predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue')

    # Add identity line (y=x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2, label='Perfect Prediction')

    plt.xlabel("Actual Delta P (Pa)")
    plt.ylabel("Predicted Delta P (Pa)")
    plt.title("Actual vs Predicted Pressure Drop")
    plt.legend()
    plt.grid(True)
    plt.savefig("prediction_plot.jpg")
    print("Visualization saved to 'prediction_plot.jpg'")

    # 8. Save Model
    try:
        joblib.dump(pipeline, model_file)
        print(f"Model pipeline saved to {model_file}")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    train_pressure_model()
