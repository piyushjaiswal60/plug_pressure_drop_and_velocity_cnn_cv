"""
03c_plug_length.py
Calculates the physical length, velocity, and maximum height of a moving plug using the Virtual Gate method.
Combines U-Net masking and Optical Flow to distinguish moving plugs from stationary beads.
"""

import numpy as np
import cv2
import importlib
import json
import os

def calculate_px_to_mm(config):
    """
    Calculates the pixels-to-mm scale factor using calibration points.
    """
    p1 = np.array(config['p1'])
    p2 = np.array(config['p2'])
    dist_mm = config['dist_mm']

    pixel_dist = np.linalg.norm(p1 - p2)
    if pixel_dist == 0:
        return 0.0

    return dist_mm / pixel_dist

def calculate_plug_metrics(video_path, roi, calibration_config, video_props):
    """
    Tracks a plug passing a virtual gate to calculate its length, velocity, and max height.
    """
    # 1. Initialize Modules
    masker_module = importlib.import_module("03a_unet_masker")
    masker = masker_module.PlugMasker()

    flow_module = importlib.import_module("03b_flow_engine")
    engine = flow_module.RAFTFlowEngine()

    px_to_mm = calculate_px_to_mm(calibration_config)
    fps = video_props['fps']

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # 2. Define Virtual Gate
    gate_x = roi[0] + roi[2] // 2

    entry_frame = None
    exit_frame = None
    velocities = []

    consecutive_frames = 0
    min_consecutive_frames = 25

    max_height = 0

    frame_idx = 0
    ret, prev_frame = cap.read()

    print(f"Analyzing video for moving plug transit (Gate X: {gate_x}, Scale: {px_to_mm:.4f} mm/px)...")

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        mask = masker.create_mask(curr_frame, roi=roi)
        flow = engine.compute_flow(prev_frame, curr_frame, roi=roi, mask=mask)

        rel_gate_x = (gate_x - roi[0])

        if 0 <= rel_gate_x < flow.shape[1]:
            gate_velocities = flow[:, rel_gate_x, :]
            magnitudes = np.sqrt(gate_velocities[..., 0]**2 + gate_velocities[..., 1]**2)

            moving_pixels = magnitudes > 0.5
            current_height = np.sum(moving_pixels)
            moving_plug_present = current_height > 0

            if moving_plug_present:
                if entry_frame is None:
                    consecutive_frames += 1
                    if consecutive_frames >= min_consecutive_frames:
                        entry_frame = frame_idx - (min_consecutive_frames - 1)
                        print(f"Moving plug confirmed entering at frame {entry_frame}")

                if entry_frame is not None:
                    max_height = max(max_height, current_height)

                    valid_v = magnitudes[moving_pixels]
                    current_vel = np.median(valid_v)
                    velocities.append(current_vel)

                    if current_height < 0.8 * max_height:
                        exit_frame = frame_idx
                        print(f"Bulk plug exit detected at frame {exit_frame} (Height: {current_height}, Max Height: {max_height})")
                        break 
            else:
                if entry_frame is None:
                    consecutive_frames = 0

        prev_frame = curr_frame
        frame_idx += 1

    cap.release()

    if entry_frame is None or exit_frame is None:
        print("No moving plug detected passing the gate.")
        return None

    # 3. Final Calculations
    avg_v_px_frame = np.mean(velocities)
    velocity_mps = (avg_v_px_frame * px_to_mm * fps) / 1000.0
    transit_time = (exit_frame - entry_frame) / fps
    length_mm = velocity_mps * transit_time * 1000.0
    
    # Convert max height from pixels to millimeters
    max_height_mm = max_height * px_to_mm

    return {
        "length_mm": length_mm,
        "velocity_mps": velocity_mps,
        "entry_frame": entry_frame,
        "exit_frame": exit_frame,
        "transit_time_sec": transit_time,
        "max_height_mm": max_height_mm
    }

if __name__ == "__main__":
    try:
        with open("calibration_config.json", "r") as f:
            config = json.load(f)
            test_roi = config['roi']
            test_calib = config
    except FileNotFoundError:
        print("Error: calibration_config.json not found. Please run 00_interactive_calibrator.py first.")
        exit()

    video_utils = importlib.import_module("02_video_utils")
    # FIX: Dynamically find the video in the video folder
    video_folder = "C:/Users/piyus/OneDrive/Desktop/pfinal/video"
    import glob
    videos = glob.glob(os.path.join(video_folder, "*.*"))

    if not videos:
        print(f"Error: No videos found in {video_folder}")
        exit()

    test_video = videos[0]
    print(f"Using video: {test_video}")

    props = video_utils.get_video_properties(test_video)
    test_props = props

    results = calculate_plug_metrics(test_video, test_roi, test_calib, test_props)

    if results:
        print("\n--- PLUG METRICS RESULTS ---")
        print(f"Entry Frame: {results['entry_frame']}")
        print(f"Exit Frame:  {results['exit_frame']}")
        print(f"Transit Time: {results['transit_time_sec']:.4f} s")
        print(f"Avg Velocity: {results['velocity_mps']:.4f} m/s")
        print(f"Plug Length:  {results['length_mm']:.4f} mm")
        print(f"Max Height:   {results['max_height_mm']:.4f} mm")
        print("----------------------------\n")
    else:
        print("No moving plug detected in the video.")