import cv2
import json
import os
import numpy as np
import importlib

def verify_masking():
    # 1. Load ROI from calibration
    try:
        with open("calibration_config.json", "r") as f:
            config = json.load(f)
            roi = config['roi']
    except FileNotFoundError:
        print("Error: calibration_config.json not found. Run calibrator first.")
        return

    # 2. Load frames with a gap to increase detectable motion
    video_folder = "C:/Users/piyus/OneDrive/Desktop/pfinal/video"
    import glob
    videos = glob.glob(os.path.join(video_folder, "*.*"))

    if not videos:
        print(f"Error: No videos found in {video_folder}")
        return

    video_path = videos[0]
    print(f"Using video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    # Frame 1
    ret1, frame1 = cap.read()

    # Skip 5 frames to make motion more obvious (Frame 1 vs Frame 6)
    for _ in range(5):
        cap.grab()

    # Frame 6
    ret2, frame2 = cap.read()
    cap.release()

    if not ret1 or not ret2:
        print("Error: Could not read frames.")
        return

    # Import using importlib to avoid SyntaxError with leading zeros
    masker_module = importlib.import_module("03a_unet_masker")
    masker = masker_module.PlugMasker()

    # 3. Create mask
    mask = masker.create_mask(frame2, roi=roi)

    # 4. Visualization
    x, y, w, h = roi
    cropped_orig = frame2[y:y+h, x:x+w]

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    comparison = np.hstack((cropped_orig, mask_bgr))

    cv2.imwrite("mask_verification.jpg", comparison)
    print("Success! Saved comparison to mask_verification.jpg")
    print("Using frame-skip (1 vs 6) and lower threshold to detect slow motion.")

if __name__ == "__main__":
    verify_masking()
