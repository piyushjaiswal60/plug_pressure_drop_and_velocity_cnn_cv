"""
ml_model/03_predict.py
Interface for predicting pressure drop for the most recent video in raw_data.xlsx.
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

    # 1. Load the most recent data from raw_data.xlsx
    raw_data_path = "raw_data.xlsx"
    if not os.path.exists(raw_data_path):
        print(f"Error: {raw_data_path} not found. Please run the analysis pipeline first.")
        exit()

    try:
        df_raw = pd.read_excel(raw_data_path)
        if df_raw.empty:
            print("Error: raw_data.xlsx is empty.")
            exit()

        # Get the last row (most recent video)
        last_row = df_raw.iloc[-1]
        video_name = last_row.get('Video Name', 'Unknown')
        print(f"\nDetected most recent video: {video_name}")
    except Exception as e:
        print(f"Error reading raw_data.xlsx: {e}")
        exit()

    # 2. Extract only the features needed by the model
    # These must match the order and names used in 01_data_cleaning.py
    required_features = {
        "Material": last_row.get("Material"),
        "Pipe Diameter (mm)": last_row.get("Pipe Diameter (mm)"),
        "Discharge (L/min)": last_row.get("Discharge (L/min)"),
        "Initial Height (mm)": last_row.get("Initial Height (mm)"),
        "Final Height (mm)": last_row.get("Final Height (mm)"),
        "Max Dynamic Height (mm)": last_row.get("Max Dynamic Height (mm)"),
        "Plug Length (mm)": last_row.get("Plug Length (mm)"),
        "Avg Velocity (m/s)": last_row.get("Avg Velocity (m/s)"),
        "Front Angle (deg)": last_row.get("Front Angle (deg)")
    }

    print("\n--- Current Features for Prediction ---")
    for k, v in required_features.items():
        print(f"  {k}: {v}")
    print("---------------------------------------")

    # 3. User Verification and Manual Override
    confirm = input("\nAre these values correct? (y/n): ").strip().lower()
    if confirm == 'n':
        print("\nPlease enter the correct values (leave blank to keep existing value):")
        for key in required_features:
            val = input(f"Enter {key} [{required_features[key]}]: ").strip()
            if val:
                # Try to convert to float if it's not the Material column
                if key != "Material":
                    try:
                        required_features[key] = float(val)
                    except ValueError:
                        print("Invalid number. Keeping previous value.")
                else:
                    required_features[key] = val

    # 4. Run Prediction
    pred = predict_pressure(required_features)

    if pred is not None:
        print(f"\nPredicted Total Pressure Drop (Delta P): {pred:.2f} Pa")
    else:
        print("\nPrediction could not be completed.")
