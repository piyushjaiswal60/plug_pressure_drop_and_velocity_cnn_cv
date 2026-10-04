"""
03d_height_analyzer.py
Lightweight extraction of plug height at the first and last frames of the video.
Uses a median-based approach to ensure robust height measurement without noise outliers.
"""

import numpy as np
import cv2
import importlib
import json
import os

def get_robust_median_height(mask, px_to_mm):
    """
    Calculates the height of the plug using the median of all vertical slices.
    This avoids outliers from noise or small gaps.
    """
    # Find all pixels that are part of the plug
    coords = np.column_stack(np.where(mask == 255))

    if coords.size == 0:
        return 0.0

    # coords is [row, col] -> [y, x]
    unique_x = np.unique(coords[:, 1])
    heights = []

    for x in unique_x:
        col_pixels = coords[coords[:, 1] == x]
        y_min = np.min(col_pixels[:, 0])
        y_max = np.max(col_pixels[:, 0])
        heights.append(y_max - y_min + 1)

    if not heights:
        return 0.0

    # Use MEDIAN height across all columns to ignore noise spikes and gaps
    robust_height_px = np.median(heights)

    return float(robust_height_px * px_to_mm)

def analyze_stationary_heights(video_path, roi, calibration_config):
    """
    Extracts plug height only from the first and last frames.
    """
    masker_module = importlib.import_module("03a_unet_masker")
    masker = masker_module.PlugMasker()

    # Calculate scale from config
    p1 = np.array(calibration_config['p1'])
    p2 = np.array(calibration_config['p2'])
    dist_mm = calibration_config['dist_mm']
    px_to_mm = dist_mm / np.linalg.norm(p1 - p2) if np.linalg.norm(p1 - p2) != 0 else 0.0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 1. First Frame Height
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame_first = cap.read()
    initial_height = 0.0
    if ret:
        mask_first = masker.create_mask(frame_first, roi=roi)
        initial_height = get_robust_median_height(mask_first, px_to_mm)

    # 2. Last Frame Height
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame_last = cap.read()
    final_height = 0.0
    if ret:
        mask_last = masker.create_mask(frame_last, roi=roi)
        final_height = get_robust_median_height(mask_last, px_to_mm)

    cap.release()

    return {
        "initial_height_mm": initial_height,
        "final_height_mm": final_height
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

    # FIX: Dynamically find the video in the video folder
    video_folder = "C:/Users/piyus/OneDrive/Desktop/pfinal/video"
    import glob
    videos = glob.glob(os.path.join(video_folder, "*.*"))

    if not videos:
        print(f"Error: No videos found in {video_folder}")
        exit()

    test_video = videos[0]
    print(f"Using video: {test_video}")

    results = analyze_stationary_heights(test_video, test_roi, test_calib)

    if results:
        print("\n--- STATIONARY PLUG HEIGHT ANALYSIS ---")
        print(f"First Frame Height: {results['initial_height_mm']:.4f} mm")
        print(f"Last Frame Height:  {results['final_height_mm']:.4f} mm")
        print("---------------------------------------\n")
