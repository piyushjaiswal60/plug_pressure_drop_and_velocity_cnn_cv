"""
03_predict.py
Uses the trained pressure_model.pkl to predict the Pressure Drop (Delta P)
for a new set of vision and physics parameters.
"""

import pandas as pd
import numpy as np
import joblib
import os

def predict_pressure():
    print("=== ML Phase 3: Pressure Drop Prediction ===\n")

    model_path = "ml_model/pressure_model.pkl"
    if not os.path.exists(model_path):
        print(f"Error: Trained model file {model_path} not found. Please run 02_train_model.py first.")
        return

    # Load the trained pipeline (including the preprocessor)
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Please enter the parameters for the new video. Leave blank to use defaults.")

    # We define the exact columns the model expects (excluding Video Name and Delta P)
    # Based on 02_train_model.py's feature set
    params = {
        "Material": input("Material (plastic bead, potash, zeolite) [plastic bead]: ").strip() or "plastic bead",
        "Pipe Diameter (mm)": float(input("Pipe Diameter (mm) [57.1]: ") or 57.1),
        "Discharge (L/min)": float(input("Discharge (L/min) [300]: ") or 300),
        "Initial Height (mm)": float(input("Initial Height (mm) [15.0]: ") or 15.0),
        "Final Height (mm)": float(input("Final Height (mm) [12.0]: ") or 12.0),
        "Transit Time (s)": float(input("Transit Time (s) [0.25]: ") or 0.25),
        "Avg Velocity (m/s)": float(input("Avg Velocity (m/s) [1.3]: ") or 1.3),
        "Plug Length (mm)": float(input("Plug Length (mm) [320]: ") or 320),
        "Max Dynamic Height (mm)": float(input("Max Dynamic Height (mm) [57]: ") or 57),
        "Front Angle (deg)": float(input("Front Angle (deg) [37]: ") or 37),
        "Edge Slope": float(input("Edge Slope [-0.76]: ") or -0.76),
        "U_sf": float(input("Superficial Air Velocity (U_sf) [1.9]: ") or 1.9),
        "Ar": float(input("Archimedes Number (Ar) [2400000]: ") or 2400000),
        "U_par": float(input("Particle Velocity (U_par) [0.9]: ") or 0.9),
        "U_plu": float(input("Plug Velocity (U_plu) [1.3]: ") or 1.3),
        "mu_e": float(input("Effective Friction Tangent (mu_e) [1.1]: ") or 1.1),
        "K": float(input("Stress Transmission Ratio (K) [0.13]: ") or 0.13),
        "epsilon": float(input("Plug Void Fraction (epsilon) [0.4]: ") or 0.4),
        "alpha": float(input("Stationary Layer Area Fraction (alpha) [0.16]: ") or 0.16),
        "part_a": float(input("Integral Parameter A (part_a) [18.0]: ") or 18.0),
        "B": float(input("Momentum & Resistance Parameter B [2000]: ") or 2000),
    }

    # Convert to DataFrame (the model expects a DF because of the ColumnTransformer)
    input_df = pd.DataFrame([params])

    # Predict
    try:
        prediction = model.predict(input_df)[0]
        print("\n" + "="*40)
        print(f"PREDICTED PRESSURE DROP: {prediction:.2f} Pa")
        print("="*40 + "\n")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    predict_pressure()
