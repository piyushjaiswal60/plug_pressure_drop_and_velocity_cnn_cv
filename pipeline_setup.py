"""
pipeline_setup.py
Automates the initial setup: ROI selection, spatial calibration, and video property extraction.
Saves all necessary constants into calibration_config.json.
"""

import json
import os
import importlib
import numpy as np

def run_setup():
    print("=== Pipeline 1: Setup & Calibration ===\n")

    # 1. Run Interactive Calibrator
    print("Step 1: Starting Interactive ROI Selection...")
    try:
        calib_ui = importlib.import_module("00_interactive_calibrator")

        # Use the actual folder where videos are stored
        video_folder = "C:/Users/piyus/OneDrive/Desktop/pfinal/video"
        import glob
        videos = glob.glob(os.path.join(video_folder, "*.*"))

        if videos:
            video_path = videos[0]
            print(f"Using video: {video_path}")
            calib_ui.run_calibration(video_path)
        else:
            print(f"Error: No videos found in {video_folder}")
            return
    except Exception as e:
        print(f"Error during ROI selection: {e}")

    # 2. Perform Spatial Calibration
    print("\nStep 2: Calculating Spatial Calibration...")
    try:
        with open("calibration_config.json", "r") as f:
            config = json.load(f)

        p1 = np.array(config['p1'])
        p2 = np.array(config['p2'])
        dist_mm = config['dist_mm']

        # Correct Calculation: distance in mm / distance in pixels
        pixel_dist = np.linalg.norm(p1 - p2)
        px_to_mm = dist_mm / pixel_dist if pixel_dist != 0 else 0.0

        # Diameter Calculation: ROI height (roi[3]) * px_to_mm
        roi = config.get('roi', [0, 0, 0, 0])
        d_px = roi[3]
        pipe_diameter_mm = d_px * px_to_mm

        # Update config
        config['mm_to_px'] = px_to_mm
        config['pipe_diameter_mm'] = pipe_diameter_mm

        with open("calibration_config.json", "w") as f:
            json.dump(config, f, indent=4)
        print(f"Calibration updated. Scale: {px_to_mm:.4f} mm/px")
        print(f"Calculated Pipe Diameter: {pipe_diameter_mm:.2f} mm")
    except Exception as e:
        print(f"Error during calibration: {e}")

    # 3. Extract Video Properties
    print("\nStep 3: Extracting Video Properties...")
    try:
        video_utils = importlib.import_module("02_video_utils")
        # Use the first video found
        video_folder = "C:/Users/piyus/OneDrive/Desktop/pfinal/video"
        import glob
        videos = glob.glob(os.path.join(video_folder, "*.*"))
        if videos:
            video_path = videos[0]
            discharge = video_utils.extract_discharge_rate(os.path.basename(video_path))

            with open("calibration_config.json", "r") as f:
                config = json.load(f)

            config['discharge_lpm'] = discharge

            with open("calibration_config.json", "w") as f:
                json.dump(config, f, indent=4)
            print(f"Discharge rate {discharge} L/min saved to config.")
        else:
            print("Error: No video found for discharge extraction.")
    except Exception as e:
        print(f"Error extracting video properties: {e}")

    print("\n=== Setup Pipeline Complete ===\n")
    print("Please check calibration_config.json for the final values.\n")

if __name__ == "__main__":
    run_setup()
