import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cv2
import json
import numpy as np
import importlib
import matplotlib.pyplot as plt

def verify_flow():
    # FIX: Use dynamic video path to match pipeline_main and pipeline_setup
    video_folder = "C:/Users/piyus/OneDrive/Desktop/pfinal/video"
    import glob
    videos = glob.glob(os.path.join(video_folder, "*.*"))

    if not videos:
        print(f"Error: No videos found in {video_folder}")
        return

    video_path = os.path.normpath(videos[0])
    print(f"Analyzing video: {os.path.basename(video_path)}")

    try:
        with open("calibration_config.json", "r") as f:
            config = json.load(f)
            roi = config['roi']
    except FileNotFoundError:
        print("Error: calibration_config.json not found.")
        return


    try:
        target_frame = int(input("Enter the frame number you want to analyze (e.g., 100): "))
    except ValueError:
        print("Invalid frame number. Please enter an integer.")
        return

    cap = cv2.VideoCapture(video_path)
    
    # Read consecutive frames (reverted step_size to maintain stable flow tracking)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()
    cap.release()

    if not ret1 or not ret2:
        print(f"Error: Could not read frames.")
        return

    masker_module = importlib.import_module("03a_unet_masker")
    masker = masker_module.PlugMasker()

    flow_module = importlib.import_module("03b_flow_engine")
    engine = flow_module.RAFTFlowEngine()

    mask = masker.create_mask(frame1, roi=roi)
    flow = engine.compute_flow(frame1, frame2, roi=roi, mask=mask)

    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    mag = magnitude

    print("\n--- VELOCITY STATISTICS REPORT ---")
    print(f"Frame analyzed: {target_frame}")
    print(f"Min Velocity: {np.min(mag):.4f} px/frame")
    print(f"Max Velocity: {np.max(mag):.4f} px/frame")
    print(f"Median Velocity: {np.median(mag[mask > 127]) if np.any(mask > 127) else 0.0:.4f} px/frame")
    print(f"Mean Velocity: {np.mean(mag[mask > 127]) if np.any(mask > 127) else 0.0:.4f} px/frame")
    print(f"95th Percentile (Top 5%): {np.percentile(mag[mask > 127], 95) if np.any(mask > 127) else 0.0:.4f} px/frame")
    print("----------------------------------\n")

    max_val = np.percentile(magnitude, 95)
    if max_val == 0: max_val = 1.0
    magnitude_norm = np.clip(magnitude / max_val * 255, 0, 255).astype(np.uint8)
    
    heatmap_bgr = cv2.applyColorMap(magnitude_norm, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    x, y, w, h = roi
    cropped = frame1[y:y+h, x:x+w]
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    
    overlay = cropped_rgb.copy()
    active_pixels = mask > 127
    
    overlay[active_pixels] = cv2.addWeighted(cropped_rgb, 0.6, heatmap_rgb, 0.4, 0)[active_pixels]

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cropped_rgb)
    plt.title(f"Original Frame {target_frame}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask_rgb)
    plt.title(f"U-Net Mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"Strict Masked Velocity Heatmap")
    plt.axis('off')

    plt.savefig("flow_verification.jpg")
    print("Success! Saved 'flow_verification.jpg'.")

if __name__ == "__main__":
    verify_flow()