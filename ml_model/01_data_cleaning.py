"""
ml_model/01_data_cleaning.py
Transforms raw experimental data into a lean, ML-ready dataset by removing redundant/derived features.
"""

import pandas as pd
import numpy as np
import os

def clean_data(input_file="raw_data.xlsx", output_file="cleaned_data.xlsx"):
    print(f"Loading raw data from {input_file}...")
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"Error loading file: {e}")
        return False

    # 1. Define Core Physical Drivers
    # We keep only the independent measurements to avoid multi-collinearity
    setup_cols = ['Pipe Diameter (mm)', 'Discharge (L/min)']
    vision_cols = [
        'Initial Height (mm)',
        'Final Height (mm)',
        'Max Dynamic Height (mm)',
        'Plug Length (mm)',
        'Avg Velocity (m/s)',
        'Front Angle (deg)'
    ]
    categorical_col = 'Material'
    target_col = 'Delta P (Pa)'

    all_keep_cols = [categorical_col] + setup_cols + vision_cols + [target_col]

    # Filter for only the columns we need
    # Use intersection to avoid errors if some columns are missing
    available_cols = [col for col in all_keep_cols if col in df.columns]
    df = df[available_cols].copy()

    print(f"Selected {len(available_cols)} physical features.")

    # 2. Clean Categorical Data
    if categorical_col in df.columns:
        df[categorical_col] = df[categorical_col].astype(str).str.strip().str.lower()
        print(f"Cleaned {categorical_col} values.")

    # 3. Handle Missing Target Values
    if target_col in df.columns:
        initial_count = len(df)
        df = df.dropna(subset=[target_col])
        print(f"Dropped {initial_count - len(df)} rows with missing target {target_col}.")
    else:
        print(f"Error: Target column {target_col} not found in data!")
        return False

    # 4. Numeric Imputation (Median)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled missing values in {col} with median: {median_val:.4f}")

    # 5. Save Cleaned Dataset
    try:
        df.to_excel(output_file, index=False)
        print(f"\nSuccess! Cleaned data saved to {output_file}")
        print("\nCleaned Data Preview:")
        print(df.head())
        return True
    except Exception as e:
        print(f"Error saving cleaned data: {e}")
        return False

if __name__ == "__main__":
    clean_data()
