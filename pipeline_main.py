"""
pipeline_main.py
Automates the full process from video analysis to pressure drop prediction.
Connects the results of 04.py (Unified Analysis) to 05_sir_formula.py.
"""

import os
import json
import importlib
import subprocess
import numpy as np

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
    # Mapping 04.py outputs and config to 05.py requirements
    # 05.py expects: Material, Pipe Dia, Discharge, Plug Length, Front Angle, Stationary height at end

    # Extracted values
    extracted_data = {
        "pipe_diameter_mm": config.get('pipe_diameter_mm'),
        "discharge_lpm": config.get('discharge_lpm'),
        "plug_length_m": results['length_mm'] / 1000.0,
        "front_angle_deg": results['front_angle_deg'],
        "final_height_mm": results['final_height_mm'] # Stationary height at end frame
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

    # Final check for missing critical values (like diameter)
    for key in extracted_data:
        if extracted_data[key] is None or extracted_data[key] == 0:
            val = input(f"CRITICAL: {key} is missing. Please enter it now: ").strip()
            extracted_data[key] = float(val)

    # --- ADDITION: Show full 04.py metrics before proceeding to physics ---
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

    # 5. Execute 05_sir_formula.py via Subprocess
    # Since 05.py uses input(), we feed the values into stdin.
    # Expected inputs in 05.py:
    # 1. Material, 2. Pipe Dia, 3. Discharge, 4. Plug Length, 5. Front Angle, 6. Height

    print("\nStep 2: Calculating Pressure Drop using 05_sir_formula.py...")

    # Material choice - we assume 'plastic bead' based on current tests,
    # but we'll ask the user to be safe.
    material = input("Enter material (plastic bead, potash, zeolite): ").strip().lower()

    # Construct the input string (one value per line)
    input_string = "\n".join([
        material,
        str(extracted_data["pipe_diameter_mm"]),
        str(extracted_data["discharge_lpm"]),
        str(extracted_data["plug_length_m"]),
        str(extracted_data["front_angle_deg"]),
        str(extracted_data["final_height_mm"])
    ]) + "\n"

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
        if stderr:
            print(f"Errors during physics calculation: {stderr}")

    except Exception as e:
        print(f"Error executing 05_sir_formula.py: {e}")

    print("\n=== Pipeline 2 Complete ===\n")

if __name__ == "__main__":
    run_main_pipeline()
