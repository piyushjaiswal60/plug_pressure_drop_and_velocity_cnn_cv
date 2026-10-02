"""
01_data_cleaning.py
Cleans the raw_data.xlsx file by filtering for physically relevant features
and preparing the dataset for Machine Learning.
"""

import pandas as pd
import os

def clean_data(input_file="raw_data.xlsx", output_file="cleaned_data.xlsx"):
    print("=== ML Phase 1: Data Cleaning ===\n")

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run the pipeline first.")
        return

    # Load the raw data
    df = pd.read_excel(input_file)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns.")

    # Define the physically relevant columns
    features = {
        "General": ["Video Name", "Material", "Pipe Diameter (mm)", "Discharge (L/min)"],
        "Vision": [
            "Initial Height (mm)", "Final Height (mm)", "Transit Time (s)",
            "Avg Velocity (m/s)", "Plug Length (mm)", "Max Dynamic Height (mm)",
            "Front Angle (deg)"
        ],
        "Physics": [
            "U_sf", "Ar", "U_par", "U_plu", "mu_e", "K",
            "epsilon", "alpha", "part_a", "B"
        ],
        "Target": ["Delta P (Pa)"]
    }

    # Flatten the list of all columns we want to keep
    all_keep_cols = []
    for category in features.values():
        all_keep_cols.extend(category)

    # Filter the dataframe to keep only relevant columns
    # Use intersection to avoid errors if some columns are missing
    existing_cols = [col for col in all_keep_cols if col in df.columns]
    df_cleaned = df[existing_cols].copy()

    # 1. Handling Missing Values
    # If the target (Delta P) is missing, the row is useless for training
    if "Delta P (Pa)" in df_cleaned.columns:
        df_cleaned = df_cleaned.dropna(subset=["Delta P (Pa)"])

    # Fill other numeric gaps with the median
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())

    # 2. Material Encoding
    # We keep the original Material column for reference in cleaned_data.xlsx,
    # but we'll do the One-Hot Encoding during the training phase.

    print(f"Cleaning complete. Final shape: {df_cleaned.shape}")

    # Save to Excel
    df_cleaned.to_excel(output_file, index=False)
    print(f"Cleaned data saved to: {output_file}")
    print("\n--- Sample of Cleaned Data ---")
    print(df_cleaned.head())

if __name__ == "__main__":
    clean_data()
