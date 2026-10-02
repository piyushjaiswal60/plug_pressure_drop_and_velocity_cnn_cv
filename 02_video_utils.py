"""
02_video_utils.py
Helper functions for handling video files and frame extraction.
"""

import cv2
import os

def get_video_properties(video_path):
    """
    Returns fixed properties for the videos.
    As per project requirements: FPS is set to 500.
    """
    props = {"fps": 500.0, "total_frames": None}
    print(f"Video Properties for {os.path.basename(video_path)}: FPS = {props['fps']}")
    return props

def extract_discharge_rate(filename):
    """
    Extracts discharge rate from filename (e.g., '300 lpm.avi' -> 300.0)
    Discharge is in Liters per Minute (LPM).
    """
    import re
    # Look for any number followed by 'lpm' or 'Lmin'
    match = re.search(r'(\d+\.?\d*)\s*(lpm|Lmin)', filename, re.IGNORECASE)
    if match:
        rate = float(match.group(1))
        print(f"Extracted Discharge Rate for {filename}: {rate} L/min")
        return rate

    print(f"Warning: Could not extract discharge rate from filename: {filename}")
    return None

def get_video_files(directory):
    """
    Lists all video files in the directory.
    """
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(valid_extensions)]
    print(f"Found {len(files)} video files in {directory}")
    return files

if __name__ == "__main__":
    # Test block to verify extraction
    import glob
    video_folder = "C:/Users/piyus/OneDrive/Desktop/pfinal/video"
    videos = glob.glob(os.path.join(video_folder, "*.*"))

    if videos:
        test_video = videos[0]
        print(f"Testing with: {test_video}")
        get_video_properties(test_video)
        extract_discharge_rate(os.path.basename(test_video))
    else:
        print("No videos found for testing.")
