import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cv2
import json
import numpy as np
import importlib
import matplotlib.pyplot as plt

def generate_flow_heatmap(video_path, roi, target_frame):
    """
    Generates a velocity heatmap for a specific frame.
    Called by pipeline_main.py or verify_flow() below.
    """
    if target_frame is None:
        print("Warning: Target frame is None. Cannot generate flow heatmap.")
        return False

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()
    cap.release()

    if not ret1 or not ret2:
        print(f"Error: Could not read frames around {target_frame}.")
        return False

    masker_module = importlib.import_module("03a_unet_masker")
    masker = masker_module.PlugMasker()

    flow_module = importlib.import_module("03b_flow_engine")
    engine = flow_module.RAFTFlowEngine()

    mask = masker.create_mask(frame1, roi=roi)
    flow = engine.compute_flow(frame1, frame2, roi=roi, mask=mask)

    if flow is None:
        return False

    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

    max_val = np.percentile(magnitude, 95)
    if max_val == 0: max_val = 1.0
    magnitude_norm = np.clip(magnitude / max_val * 255, 0, 255).astype(np.uint8)

    heatmap_bgr = cv2.applyColorMap(magnitude_norm, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    x, y, w, h = roi
    x, y = max(0, x), max(0, y)
    w = max(1, min(w, frame1.shape[1] - x))
    h = max(1, min(h, frame1.shape[0] - y))
    
    cropped = frame1[y:y+h, x:x+w]
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    overlay = cropped_rgb.copy()
    active_pixels = mask > 127
    
    if overlay.shape[:2] == heatmap_rgb.shape[:2] == active_pixels.shape:
        overlay[active_pixels] = cv2.addWeighted(cropped_rgb, 0.6, heatmap_rgb, 0.4, 0)[active_pixels]

    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.title(f"Velocity Heatmap - Frame {target_frame}")
    plt.axis('off')
    plt.savefig("flow_verification.jpg")
    plt.close()
    return True

def verify_flow():
    """
    Standalone interactive mode.
    """
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

    if generate_flow_heatmap(video_path, roi, target_frame):
        print("Success! Saved 'flow_verification.jpg'.")

if __name__ == "__main__":
    verify_flow()