"""
pipeline_main.py
Automates the full process from video analysis to pressure drop prediction.
Connects the results of 04.py (Unified Analysis) to 05_sir_formula.py.
Exports comprehensive raw data to an Excel file.
"""

import os
import json
import importlib
import subprocess
import numpy as np
import pandas as pd
import re

def parse_physics_output(stdout):
    """
    Parses the stdout from 05_sir_formula.py to extract numerical values.
    """
    data = {}
    # Patterns for the variables in 05_sir_formula.py
    patterns = {
        "U_sf": r"Superficial Air Velocity \(U_sf\): ([\d\.]+)",
        "Ar": r"Archimedes Number \(Ar\): ([\d\.]+)",
        "U_par": r"Particle Velocity \(U_par\): ([\d\.]+)",
        "U_plu": r"Plug Velocity \(U_plu\): ([\d\.]+)",
        "mu_e": r"Effective Friction Tangent \(mu_e\): ([\d\.]+)",
        "K": r"Stress Transmission Ratio \(K\): ([\d\.]+)",
        "epsilon": r"Plug Void Fraction \(epsilon\): ([\d\.]+)",
        "alpha": r"Stationary Layer Area Fraction \(alpha\): ([\d\.]+)",
        "part_a": r"Integral Parameter A component \(part_a\): ([\d\.]+)",
        "B": r"Momentum & Resistance Parameter B: ([\d\.]+)",
        "Delta_P": r"Total Pressure Drop \(Delta P\): ([\d\.]+)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if match:
            data[key] = float(match.group(1))
        else:
            data[key] = None

    return data

def save_to_excel(data_row, filename="raw_data.xlsx"):
    """
    Appends a row of data to the raw_data.xlsx file.
    """
    df_new = pd.DataFrame([data_row])

    if os.path.exists(filename):
        try:
            df_existing = pd.read_excel(filename)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception as e:
            print(f"Error reading existing excel file: {e}. Creating new one.")
            df_final = df_new
    else:
        df_final = df_new

    df_final.to_excel(filename, index=False)
    print(f"Data successfully saved to {filename}")

def run_main_pipeline():
    print("=== Pipeline 2: Analysis & Pressure Prediction ===\n")

    # 1. Load Calibration Config
    try:
        with open("calibration_config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: calibration_config.json not found. Please run pipeline_setup.py first.")
        return

    # 2. Perform Unified Analysis (from 04.py)
    print("Step 1: Extracting metrics from video using 04.py...")
    try:
        unified_module = importlib.import_module("04")
        video_path = "C:/Users/piyus/OneDrive/Desktop/pfinal/plastic_bead/300 lpm.avi"
        roi = config.get('roi')

        # Get video properties for the call
        video_utils = importlib.import_module("02_video_utils")
        props = video_utils.get_video_properties(video_path)

        # Run the analysis
        results = unified_module.run_unified_analysis(video_path, roi, config, props)
        if not results:
            print("Error: Analysis failed to produce results.")
            return
    except Exception as e:
        print(f"Error running 04.py: {e}")
        return

    # 3. Prepare inputs for 05_sir_formula.py
    extracted_data = {
        "pipe_diameter_mm": config.get('pipe_diameter_mm'),
        "discharge_lpm": config.get('discharge_lpm'),
        "plug_length_m": results['length_mm'] / 1000.0,
        "front_angle_deg": results['front_angle_deg'],
        "final_height_mm": results['final_height_mm']
    }

    print("\n--- Extracted Parameters for Pressure Calculation ---")
    for key, val in extracted_data.items():
        print(f"{key}: {val}")
    print("---------------------------------------------------\n")

    # 4. User Verification and Manual Override
    confirm = input("Are these values correct, or do you need to feed any manually? (y/n): ").strip().lower()
    if confirm == 'n':
        print("\nPlease enter the correct values (leave blank to keep extracted value):")
        for key in extracted_data:
            val = input(f"Enter {key} [{extracted_data[key]}]: ").strip()
            if val:
                extracted_data[key] = float(val)

    for key in extracted_data:
        if extracted_data[key] is None or extracted_data[key] == 0:
            val = input(f"CRITICAL: {key} is missing. Please enter it now: ").strip()
            extracted_data[key] = float(val)

    # Show full 04.py metrics
    print("\n=============================================")
    print("         DETAILED ANALYSIS REPORT (04.py)     ")
    print("=============================================")
    print(f"Initial Height:      {results.get('initial_height_mm', 0):>7.2f} mm")
    print(f"Final Height:        {results.get('final_height_mm', 0):>7.2f} mm")
    print(f"Entry Frame:         {results.get('entry_frame', 'N/A'):>7}")
    print(f"Exit Frame:          {results.get('exit_frame', 'N/A'):>7}")
    print(f"Transit Time:        {results.get('transit_time_sec', 0):>7.4f} s")
    print(f"Avg Velocity:        {results.get('velocity_mps', 0):>7.4f} m/s")
    print(f"Plug Length:         {results.get('length_mm', 0):>7.2f} mm")
    print(f"Max Dynamic Height:  {results.get('max_height_mm', 0):>7.2f} mm")
    print(f"Front Angle:         {results.get('front_angle_deg', 0):>7.2f} degrees")
    print(f"Edge Slope:          {results.get('edge_slope', 0):>7.4f}")
    print("=============================================\n")

    # 5. Execute 05_sir_formula.py
    print("\nStep 2: Calculating Pressure Drop using 05_sir_formula.py...")
    material = input("Enter material (plastic bead, potash, zeolite): ").strip().lower()

    input_string = "\n".join([
        material,
        str(extracted_data["pipe_diameter_mm"]),
        str(extracted_data["discharge_lpm"]),
        str(extracted_data["plug_length_m"]),
        str(extracted_data["front_angle_deg"]),
        str(extracted_data["final_height_mm"])
    ]) + "\n"

    physics_results = {}
    try:
        process = subprocess.Popen(
            ["python", "05_sir_formula.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=input_string)

        if stdout:
            print("\n=== FINAL RESULTS FROM PHYSICS MODEL ===")
            print(stdout)
            physics_results = parse_physics_output(stdout)
        if stderr:
            print(f"Errors during physics calculation: {stderr}")

    except Exception as e:
        print(f"Error executing 05_sir_formula.py: {e}")

    # --- EXCEL EXPORT SECTION ---
    save_confirm = input("\nWould you like to save all these results to the raw data Excel file? (y/n): ").strip().lower()
    if save_confirm == 'y':
        # Construct a flat dictionary for the Excel row
        # Separate ROI and points as requested: "Every value in new column"
        roi = config.get('roi', [0,0,0,0])
        p1 = config.get('p1', [0,0])
        p2 = config.get('p2', [0,0])

        data_row = {
            "Video Name": os.path.basename(video_path),
            "Material": material,
            "FPS": props.get('fps'),
            "p1_x": p1[0],
            "p1_y": p1[1],
            "p2_x": p2[0],
            "p2_y": p2[1],
            "dist_mm": config.get('dist_mm'),
            "roi_x": roi[0],
            "roi_y": roi[1],
            "roi_w": roi[2],
            "roi_h": roi[3],
            "px_to_mm": config.get('px_to_mm'),
            "Pipe Diameter (mm)": extracted_data["pipe_diameter_mm"],
            "Discharge (L/min)": extracted_data["discharge_lpm"],
            "Initial Height (mm)": results.get('initial_height_mm'),
            "Final Height (mm)": results.get('final_height_mm'),
            "Entry Frame": results.get('entry_frame'),
            "Exit Frame": results.get('exit_frame'),
            "Transit Time (s)": results.get('transit_time_sec'),
            "Avg Velocity (m/s)": results.get('velocity_mps'),
            "Plug Length (mm)": results.get('length_mm'),
            "Max Dynamic Height (mm)": results.get('max_height_mm'),
            "Front Angle (deg)": results.get('front_angle_deg'),
            "Edge Slope": results.get('edge_slope'),
            "U_sf": physics_results.get("U_sf"),
            "Ar": physics_results.get("Ar"),
            "U_par": physics_results.get("U_par"),
            "U_plu": physics_results.get("U_plu"),
            "mu_e": physics_results.get("mu_e"),
            "K": physics_results.get("K"),
            "epsilon": physics_results.get("epsilon"),
            "alpha": physics_results.get("alpha"),
            "part_a": physics_results.get("part_a"),
            "B": physics_results.get("B"),
            "Delta P (Pa)": physics_results.get("Delta_P"),
        }
        save_to_excel(data_row)

    print("\n=== Pipeline 2 Complete ===\n")

if __name__ == "__main__":
    run_main_pipeline()
