"""
pipeline_setup.py
Automates the initial setup: ROI selection, spatial calibration, and video property extraction.
Saves all necessary constants into calibration_config.json.
"""

import json
import os
import importlib

def run_setup():
    print("=== Pipeline 1: Setup & Calibration ===\n")

    # 1. Run Interactive Calibrator
    # Note: 00_interactive_calibrator usually has a main() or logic that opens a window.
    # We import it to ensure the user goes through the ROI selection process.
    print("Step 1: Starting Interactive ROI Selection...")
    try:
        calib_ui = importlib.import_module("00_interactive_calibrator")
        if hasattr(calib_ui, 'main'):
            calib_ui.main()
        else:
            # If there's no main(), we assume the script runs logic on import or
            # we just notify the user to run it manually if it's purely interactive.
            print("Please ensure you have run 00_interactive_calibrator.py and saved your ROI.")
    except Exception as e:
        print(f"Error during ROI selection: {e}")

    # 2. Perform Spatial Calibration
    print("\nStep 2: Calculating Spatial Calibration...")
    try:
        calib_logic = importlib.import_module("01_calibration")
        # We assume 01_calibration has a function to calculate the ratio
        # If not, we implement the math here based on the known 01_calibration logic.
        with open("calibration_config.json", "r") as f:
            config = json.load(f)

        p1 = np.array(config['p1'])
        p2 = np.array(config['p2'])
        dist_mm = config['dist_mm']
        px_to_mm = dist_mm / np.linalg.norm(p1 - p2)

        # Update config with the calculated ratio and diameter
        config['px_to_mm'] = px_to_mm
        # Diameter is often the ROI height or a specific value from 01_calibration
        # For now, we save what we have and add placeholders for the user to check
        config['pipe_diameter_mm'] = config.get('pipe_diameter_mm', 0.0)

        with open("calibration_config.json", "w") as f:
            json.dump(config, f, indent=4)
        print(f"Calibration updated. Scale: {px_to_mm:.4f} mm/px")
    except Exception as e:
        print(f"Error during calibration: {e}")

    # 3. Extract Video Properties
    print("\nStep 3: Extracting Video Properties...")
    try:
        video_utils = importlib.import_module("02_video_utils")
        # Use the target video
        video_path = "C:/Users/piyus/OneDrive/Desktop/pfinal/plastic_bead/300 lpm.avi"
        discharge = video_utils.extract_discharge_rate(os.path.basename(video_path))

        # Save discharge to config
        with open("calibration_config.json", "r") as f:
            config = json.load(f)

        config['discharge_lpm'] = discharge

        with open("calibration_config.json", "w") as f:
            json.dump(config, f, indent=4)
        print(f"Discharge rate {discharge} L/min saved to config.")
    except Exception as e:
        print(f"Error extracting video properties: {e}")

    print("\n=== Setup Pipeline Complete ===\n")

if __name__ == "__main__":
    import numpy as np
    run_setup()
