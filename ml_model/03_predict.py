"""
ml_model/03_predict.py
Interface for predicting pressure drop for new experimental parameters.
"""

import pandas as pd
import numpy as np
import joblib
import os

def predict_pressure(params_dict, model_path="pressure_model.pkl"):
    """
    Takes a dictionary of parameters and predicts the pressure drop.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Please train the model first.")
        return None

    try:
        # Load the entire pipeline (includes scaler and encoder)
        pipeline = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Convert input dictionary to DataFrame
    # The columns MUST be in the exact order as the training set
    df_input = pd.DataFrame([params_dict])

    try:
        prediction = pipeline.predict(df_input)[0]
        return prediction
    except Exception as e:
        print(f"Prediction failed: {e}")
        print("Ensure all required features are present and formatted correctly.")
        return None

if __name__ == "__main__":
    print("--- Plug Flow Pressure Drop Predictor ---")

    # Example input - in a real scenario, these could come from pipeline_main's output
    # Replace these with actual values for testing
    test_params = {
        "Material": "plastic bead",
        "Pipe Diameter (mm)": 55.98,
        "Discharge (L/min)": 450.0,
        "Initial Height (mm)": 13.76,
        "Final Height (mm)": 20.10,
        "Max Dynamic Height (mm)": 55.62,
        "Plug Length (mm)": 296.63,
        "Avg Velocity (m/s)": 1.58,
        "Front Angle (deg)": 44.38
    }

    print("\nInput Parameters:")
    for k, v in test_params.items():
        print(f"  {k}: {v}")

    pred = predict_pressure(test_params)

    if pred is not None:
        print(f"\nPredicted Total Pressure Drop (Delta P): {pred:.2f} Pa")
    else:
        print("\nPrediction could not be completed.")
