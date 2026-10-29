"""
ml_model/03_predict.py
Interface for predicting pressure drop for all videos in raw_data.xlsx.
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

def predict_pressure(params_dict, model_path="models/ml/pressure_model.pkl"):
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
    print("--- Plug Flow Batch Pressure Predictor ---")

    # 1. Load all data from raw_data.xlsx
    raw_data_path = "data/video_extracted_data/raw_data.xlsx"
    if not os.path.exists(raw_data_path):
        print(f"Error: {raw_data_path} not found. Please run the analysis pipeline first.")
        exit()

    try:
        df_raw = pd.read_excel(raw_data_path)
        if df_raw.empty:
            print("Error: raw_data.xlsx is empty.")
            exit()
        print(f"Loaded {len(df_raw)} records from {raw_data_path}")
    except Exception as e:
        print(f"Error reading raw_data.xlsx: {e}")
        exit()

    # 2. Define the features needed by the model
    required_cols = [
        "Material",
        "Pipe Diameter (mm)",
        "Discharge (L/min)",
        "Initial Height (mm)",
        "Final Height (mm)",
        "Max Dynamic Height (mm)",
        "Plug Length (mm)",
        "Avg Velocity (m/s)",
        "Front Angle (deg)"
    ]

    missing_cols = [col for col in required_cols if col not in df_raw.columns]
    if missing_cols:
        print(f"Error: Missing required columns in excel: {missing_cols}")
        exit()

    # Create the feature set for all rows
    X_all = df_raw[required_cols]

    # 3. Run Predictions for all rows
    print("\nPredicting pressure drop for all records...")

    try:
        pipeline = joblib.load("models/ml/pressure_model.pkl")
        predictions = pipeline.predict(X_all)
        df_raw["Predicted Delta P (Pa)"] = predictions

        print("\n--- Prediction Results ---")
        display_cols = ["Video Name", "Material"] + required_cols + ["Predicted Delta P (Pa)"]
        existing_display_cols = [c for c in display_cols if c in df_raw.columns]
        print(df_raw[existing_display_cols].to_string(index=False))

        # Save results to data/ml_data/
        output_path = "data/ml_data/predictions_results.xlsx"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_raw.to_excel(output_path, index=False)
        print(f"\nFull results saved to: {output_path}")

        # Generate Visualization (Actual vs Predicted)
        if "Delta P (Pa)" in df_raw.columns:
            actual = df_raw["Delta P (Pa)"]
            predicted = df_raw["Predicted Delta P (Pa)"]
            videos = df_raw["Video Name"] if "Video Name" in df_raw.columns else range(len(df_raw))
            materials = df_raw["Material"] if "Material" in df_raw.columns else ["Unknown"] * len(df_raw)

            plt.figure(figsize=(16, 7))

            # Subplot 1: Scatter Plot (Actual vs Predicted)
            plt.subplot(1, 2, 1)
            unique_mats = materials.unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_mats)))
            markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'P']

            for i, mat in enumerate(unique_mats):
                mask = (materials == mat)
                plt.scatter(actual[mask], predicted[mask],
                            color=colors[i],
                            marker=markers[i % len(markers)],
                            label=mat,
                            edgecolor='k', alpha=0.8, s=60)

            plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2, label="Perfect Prediction")
            plt.xlabel("Actual Delta P (Pa)")
            plt.ylabel("Predicted Delta P (Pa)")
            plt.title("Actual vs Predicted Pressure Drop")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Subplot 2: Bar Chart (Comparison)
            plt.subplot(1, 2, 2)
            x = np.arange(len(actual))
            width = 0.35

            # We use material-based colors
            bar_colors = []
            for mat in materials:
                idx = np.where(unique_mats == mat)[0][0]
                bar_colors.append(colors[idx])

            # Actual bars: Material color, lower alpha
            plt.bar(x - width/2, actual, width, label='Actual', color=bar_colors, alpha=0.4, edgecolor='k')
            # Predicted bars: Material color, higher alpha
            plt.bar(x + width/2, predicted, width, label='Predicted', color=bar_colors, alpha=0.9, edgecolor='k')

            plt.xticks(x, videos, rotation=45, ha='right')
            plt.ylabel("Pressure Drop (Pa)")
            plt.title("Comparison per Video (Colored by Material)")

            # Combined legend
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w', label=mat,
                                      markerfacecolor=colors[i], markersize=10)
                               for i, mat in enumerate(unique_mats)]

            legend_elements.append(Line2D([0], [0], color='gray', lw=4, alpha=0.4, label='Actual'))
            legend_elements.append(Line2D([0], [0], color='gray', lw=4, alpha=0.9, label='Predicted'))

            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
            plt.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plot_path = "data/ml_data/prediction_comparison.jpg"
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Comparison plot saved to: {plot_path}")
        else:
            print("\nWARNING: 'Delta P (Pa)' column not found in raw_data.xlsx. Cannot generate comparison plots.")

    except Exception as e:
        print(f"Batch prediction failed: {e}")
