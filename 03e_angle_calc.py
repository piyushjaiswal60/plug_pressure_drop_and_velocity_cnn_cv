"""
03e_angle_calc.py
Calculates the front angle of the moving plug during its ascent.
Uses U-Net masks and Optical Flow to ensure only the moving plug's front edge is analyzed,
ignoring stationary beads at the bottom of the pipe.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import cv2
import importlib
import json
import matplotlib.pyplot as plt

def calculate_px_to_mm(config):
    p1 = np.array(config['p1'])
    p2 = np.array(config['p2'])
    dist_mm = config['dist_mm']
    pixel_dist = np.linalg.norm(p1 - p2)
    return dist_mm / pixel_dist if pixel_dist != 0 else 0.0

def get_front_edge_points(flow, mask):
    """
    Extracts the true front edge of the moving plug.
    Isolates the moving mass from the stationary bed and dynamically targets
    the correct face based on the bulk flow direction.
    """
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    
    # 1. Isolate the solid moving plug (moving > 0.5 AND belongs to plug mask)
    moving_mask = ((mag > 0.5) & (mask > 127)).astype(np.uint8)
    
    # Clean up internal gaps to create a solid moving mass
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    solid_moving = cv2.morphologyEx(moving_mask, cv2.MORPH_CLOSE, kernel)
    solid_moving = cv2.morphologyEx(solid_moving, cv2.MORPH_OPEN, kernel)
    
    # 2. Find the largest connected component (the main moving wave)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(solid_moving, connectivity=8)
    if num_labels <= 1:
        return np.array([]), np.zeros_like(mask, dtype=bool)
        
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    plug_blob = (labels == largest_label)
    
    # 3. Determine flow direction to find the true front face
    u_flow = flow[..., 0]
    bulk_u = np.median(u_flow[plug_blob])
    
    front_edge = []
    h_roi = plug_blob.shape[0]
    
    for y in range(h_roi):
        row = plug_blob[y, :]
        if np.any(row):
            if bulk_u > 0:
                # Moving Left-to-Right -> Front is the Rightmost edge
                x_edge = np.where(row)[0][-1]
            else:
                # Moving Right-to-Left -> Front is the Leftmost edge
                x_edge = np.where(row)[0][0]
            
            front_edge.append((x_edge, y))
            
    return np.array(front_edge), plug_blob

def calculate_front_angle(video_path, roi, calibration_config, video_props):
    """
    Calculates the front angle of the moving plug during its ascent.
    """
    masker_module = importlib.import_module("03a_unet_masker")
    masker = masker_module.PlugMasker()

    flow_module = importlib.import_module("03b_flow_engine")
    engine = flow_module.RAFTFlowEngine()

    px_to_mm = calculate_px_to_mm(calibration_config)
    fps = video_props['fps']

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # 1. SCAN FOR ASCENT
    gate_x = roi[0] + roi[2] // 2
    entry_frame = None
    consecutive_frames = 0
    min_consecutive_frames = 25

    frame_idx = 0
    ret, prev_frame = cap.read()

    print(f"Scanning for plug ascent at Gate X: {gate_x}...")
    while True:
        ret, curr_frame = cap.read()
        if not ret: break

        mask = masker.create_mask(curr_frame, roi=roi)
        flow = engine.compute_flow(prev_frame, curr_frame, roi=roi, mask=mask)

        rel_gate_x = (gate_x - roi[0])
        if 0 <= rel_gate_x < flow.shape[1]:
            gate_velocities = flow[:, rel_gate_x, :]
            gate_mag = np.sqrt(gate_velocities[..., 0]**2 + gate_velocities[..., 1]**2)
            
            current_height = np.sum(gate_mag > 0.5)
            
            if current_height > 0:
                consecutive_frames += 1
                if consecutive_frames >= min_consecutive_frames and entry_frame is None:
                    entry_frame = frame_idx - (min_consecutive_frames - 1)
                    print(f"Ascent confirmed entering at frame {entry_frame}")
                    break
            else:
                if entry_frame is None: 
                    consecutive_frames = 0

        prev_frame = curr_frame
        frame_idx += 1

    if entry_frame is None:
        print("No moving plug ascent detected.")
        cap.release()
        return None

    # 2. ANALYZE FRONT ANGLE
    print(f"Analyzing front angle over 5 frames starting from frame {entry_frame}...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, entry_frame)

    last_mask = None
    last_frame_rgb = None
    last_points = None
    last_blob = None

    ret, prev_frame = cap.read()

    for i in range(5):
        ret, curr_frame = cap.read()
        if not ret: break

        mask = masker.create_mask(curr_frame, roi=roi)
        flow = engine.compute_flow(prev_frame, curr_frame, roi=roi, mask=mask)

        points, blob = get_front_edge_points(flow, mask)
        
        if points.size > 0:
            last_points = points
            last_blob = blob

        last_mask = mask
        last_frame_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
        prev_frame = curr_frame

    cap.release()

    if last_points is None or last_points.size == 0:
        print("Could not extract moving front edge points.")
        return None

    # GUARANTEE MATCH: Calculate angle directly from the final frame's pixel fit
    px_X = last_points[:, 0]
    px_Y = last_points[:, 1]
    
    fit_m_px, fit_c_px = np.polyfit(px_Y, px_X, 1)
    
    final_slope = 1.0 / fit_m_px if fit_m_px != 0 else 9999.0
    final_angle = abs(np.degrees(np.arctan(final_slope)))

    # 3. VISUALIZATION
    if last_mask is not None and last_frame_rgb is not None:
        plt.figure(figsize=(12, 8))
        plt.imshow(last_frame_rgb)

        x_roi, y_roi, w_roi, h_roi = roi
        full_overlay = np.zeros((last_frame_rgb.shape[0], last_frame_rgb.shape[1], 4))
        
        roi_overlay = np.zeros((h_roi, w_roi, 4))
        
        # Color stationary plug red
        stationary = (last_mask > 127) & (~last_blob)
        roi_overlay[stationary] = [1, 0, 0, 0.3] 
        
        # Color the moving plug (wave) yellow
        roi_overlay[last_blob] = [1, 1, 0, 0.4] 
        
        full_overlay[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi] = roi_overlay
        
        plt.imshow(full_overlay)

        plt.scatter(px_X + x_roi, px_Y + y_roi, color='blue', s=15, label='Detected Moving Face')

        # Fit line in pixel space for plotting
        y_range = np.linspace(px_Y.min(), px_Y.max(), 100)
        x_range = fit_m_px * y_range + fit_c_px

        plt.plot(x_range + x_roi, y_range + y_roi, color='lime', linewidth=3, label=f'Angle: {final_angle:.2f}°')
        plt.legend(loc='upper right')
        plt.title(f"Front Edge Fit (Final Frame of Ascent Sequence)")
        plt.savefig("angle_verification.jpg")
        print("Visualization saved to 'angle_verification.jpg'")

    return {
        "front_angle_deg": final_angle,
        "entry_frame": entry_frame,
        "slope": final_slope
    }

if __name__ == "__main__":
    try:
        with open("calibration_config.json", "r") as f:
            config = json.load(f)
            test_roi = config['roi']
            test_calib = config
    except FileNotFoundError:
        print("Error: calibration_config.json not found.")
        exit()

    video_utils = importlib.import_module("02_video_utils")
    
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

    results = calculate_front_angle(test_video, test_roi, test_calib, test_props)

    if results:
        print("\n--- PLUG FRONT ANGLE RESULTS ---")
        print(f"Entry Frame: {results['entry_frame']}")
        print(f"Front Angle: {results['front_angle_deg']:.2f} degrees")
        print(f"Edge Slope:  {results['slope']:.4f}")
        print("--------------------------------\n")
    else:
        print("Failed to calculate front angle.")