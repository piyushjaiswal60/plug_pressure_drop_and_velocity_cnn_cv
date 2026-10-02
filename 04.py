"""
04_unified_pipeline.py
A highly optimized, single-pass pipeline that runs U-Net masking and Optical Flow 
once per frame to simultaneously extract stationary heights, transit metrics, 
length, velocity, and the dynamic front angle of the moving plug.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import cv2
import importlib
import json
import matplotlib.pyplot as plt

# ==========================================
# HELPER FUNCTIONS 
# ==========================================
def calculate_px_to_mm(config):
    p1 = np.array(config['p1'])
    p2 = np.array(config['p2'])
    dist_mm = config['dist_mm']
    pixel_dist = np.linalg.norm(p1 - p2)
    return dist_mm / pixel_dist if pixel_dist != 0 else 0.0

def get_robust_median_height(mask, px_to_mm):
    coords = np.column_stack(np.where(mask == 255))
    if coords.size == 0: return 0.0
    
    unique_x = np.unique(coords[:, 1])
    heights = [np.max(coords[coords[:, 1] == x][:, 0]) - np.min(coords[coords[:, 1] == x][:, 0]) + 1 for x in unique_x]
    
    return float(np.median(heights) * px_to_mm) if heights else 0.0

def get_front_edge_points(flow, mask):
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    moving_mask = ((mag > 0.5) & (mask > 127)).astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    solid_moving = cv2.morphologyEx(moving_mask, cv2.MORPH_CLOSE, kernel)
    solid_moving = cv2.morphologyEx(solid_moving, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(solid_moving, connectivity=8)
    if num_labels <= 1:
        return np.array([]), np.zeros_like(mask, dtype=bool)
        
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    plug_blob = (labels == largest_label)
    
    u_flow = flow[..., 0]
    bulk_u = np.median(u_flow[plug_blob])
    
    front_edge = []
    h_roi = plug_blob.shape[0]
    
    for y in range(h_roi):
        row = plug_blob[y, :]
        if np.any(row):
            x_edge = np.where(row)[0][-1] if bulk_u > 0 else np.where(row)[0][0]
            front_edge.append((x_edge, y))
            
    return np.array(front_edge), plug_blob


# ==========================================
# MASTER UNIFIED LOOP
# ==========================================
def run_unified_analysis(video_path, roi, calibration_config, video_props):
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

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gate_x = roi[0] + roi[2] // 2

    # --- Initial State Variables ---
    initial_height_mm = 0.0
    final_height_mm = 0.0
    
    entry_frame = None
    exit_frame = None
    velocities = []
    consecutive_frames = 0
    min_consecutive_frames = 25
    max_height_pixels = 0
    
    angles = []
    slopes = []
    last_mask = None
    last_frame_rgb = None
    last_points = None
    last_blob = None

    # Temporary buffer to retroactively calculate the angle on the correct first 5 frames
    potential_entry_buffer = []

    print(f"\nStarting Unified Pipeline (Gate X: {gate_x}, Scale: {px_to_mm:.4f} mm/px)...")
    print("Processing video stream...")

    # FIX: Initialize frame index exactly as the standalone scripts do to ensure identical output frames
    frame_idx = 0
    
    ret, prev_frame = cap.read()
    if ret:
        first_mask = masker.create_mask(prev_frame, roi=roi)
        initial_height_mm = get_robust_median_height(first_mask, px_to_mm)
    
    # Single Pass Main Loop
    while True:
        ret, curr_frame = cap.read()
        if not ret: break

        mask = masker.create_mask(curr_frame, roi=roi)
        flow = engine.compute_flow(prev_frame, curr_frame, roi=roi, mask=mask)

        # --- TRANSIT, LENGTH & ANGLE TRACKING ---
        if exit_frame is None:
            rel_gate_x = (gate_x - roi[0])
            if 0 <= rel_gate_x < flow.shape[1]:
                gate_velocities = flow[:, rel_gate_x, :]
                gate_mag = np.sqrt(gate_velocities[..., 0]**2 + gate_velocities[..., 1]**2)

                moving_pixels = gate_mag > 0.5
                current_height = np.sum(moving_pixels)
                
                if current_height > 0:
                    
                    if entry_frame is None:
                        # We are building up to 25 frames. Save the geometry in the buffer just in case.
                        points, blob = get_front_edge_points(flow, mask)
                        potential_entry_buffer.append({
                            'points': points,
                            'blob': blob,
                            'mask': mask,
                            'frame_rgb': cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
                        })
                        
                        consecutive_frames += 1
                        
                        if consecutive_frames >= min_consecutive_frames:
                            entry_frame = frame_idx - (min_consecutive_frames - 1)
                            print(f" -> Ascent confirmed entering at frame {entry_frame}")

                            # FIX: Ascent is confirmed! Now retroactively extract the angles 
                            # strictly from the first 5 frames in our buffer.
                            for i in range(min(5, len(potential_entry_buffer))):
                                data = potential_entry_buffer[i]
                                pts = data['points']
                                if pts.size > 0:
                                    X, Y = pts[:, 0] * px_to_mm, pts[:, 1] * px_to_mm
                                    fit_m, fit_c = np.polyfit(Y, X, 1) 
                                    slope = 1.0 / fit_m if fit_m != 0 else 9999.0
                                    angles.append(abs(np.degrees(np.arctan(slope))))
                                    slopes.append(slope)
                                    
                                    last_points = pts
                                    last_blob = data['blob']
                                    last_mask = data['mask']
                                    last_frame_rgb = data['frame_rgb']

                            # Clear buffer to save memory for the rest of the video
                            potential_entry_buffer.clear()
                    
                    if entry_frame is not None:
                        # Keep tracking the bulk flow metrics exactly as 03c does
                        max_height_pixels = max(max_height_pixels, current_height)
                        velocities.append(np.median(gate_mag[moving_pixels]))

                        if current_height < 0.9 * max_height_pixels:
                            exit_frame = frame_idx
                            print(f" -> Bulk plug exit detected at frame {exit_frame}")
                else:
                    if entry_frame is None: 
                        consecutive_frames = 0
                        potential_entry_buffer.clear()

        if exit_frame is not None and entry_frame is not None:
             print(" -> Core metrics gathered. Skipping to end for final height calculation...")
             break

        prev_frame = curr_frame
        frame_idx += 1

    # --- FINAL STATIONARY HEIGHT ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame_last = cap.read()
    if ret:
        mask_last = masker.create_mask(frame_last, roi=roi)
        final_height_mm = get_robust_median_height(mask_last, px_to_mm)

    cap.release()

    # ==========================================
    # DATA AGGREGATION & VISUALIZATION
    # ==========================================
    if entry_frame is None or exit_frame is None:
        print("\nError: Plug transit was not fully captured in this video.")
        return None

    avg_v_px_frame = np.mean(velocities) if velocities else 0
    velocity_mps = (avg_v_px_frame * px_to_mm * fps) / 1000.0
    transit_time = (exit_frame - entry_frame) / fps
    length_mm = velocity_mps * transit_time * 1000.0
    max_height_mm = max_height_pixels * px_to_mm
    
    final_angle = np.median(angles) if angles else 0.0
    final_slope = np.median(slopes) if slopes else 0.0

    if last_mask is not None and last_frame_rgb is not None and last_points is not None:
        plt.figure(figsize=(12, 8))
        plt.imshow(last_frame_rgb)

        x_roi, y_roi, w_roi, h_roi = roi
        full_overlay = np.zeros((last_frame_rgb.shape[0], last_frame_rgb.shape[1], 4))
        roi_overlay = np.zeros((h_roi, w_roi, 4))
        
        stationary = (last_mask > 127) & (~last_blob)
        roi_overlay[stationary] = [1, 0, 0, 0.3] 
        roi_overlay[last_blob] = [1, 1, 0, 0.4]  
        
        full_overlay[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi] = roi_overlay
        plt.imshow(full_overlay)

        px_X, px_Y = last_points[:, 0], last_points[:, 1]
        plt.scatter(px_X + x_roi, px_Y + y_roi, color='blue', s=15, label='Detected Moving Face')

        fit_m_px, fit_c_px = np.polyfit(px_Y, px_X, 1)
        y_range = np.linspace(px_Y.min(), px_Y.max(), 100)
        x_range = fit_m_px * y_range + fit_c_px

        plt.plot(x_range + x_roi, y_range + y_roi, color='lime', linewidth=3, label=f'Angle: {final_angle:.2f}°')
        plt.legend(loc='upper right')
        plt.title(f"Unified Analysis - Front Edge Fit (Frame {entry_frame})")
        plt.savefig("angle_verification.jpg")
        print(" -> Visualization saved to 'angle_verification.jpg'")

    return {
        "initial_height_mm": initial_height_mm,
        "final_height_mm": final_height_mm,
        "entry_frame": entry_frame,
        "exit_frame": exit_frame,
        "transit_time_sec": transit_time,
        "velocity_mps": velocity_mps,
        "length_mm": length_mm,
        "max_height_mm": max_height_mm,
        "front_angle_deg": final_angle,
        "edge_slope": final_slope
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
    test_video = "C:/Users/piyus/OneDrive/Desktop/pfinal/plastic_bead/300 lpm.avi"

    if not os.path.exists(test_video):
        print(f"Error: Video file not found at {test_video}")
        exit()

    props = video_utils.get_video_properties(test_video)
    
    results = run_unified_analysis(test_video, test_roi, test_calib, props)

    if results:
        print("\n=============================================")
        print("         UNIFIED PLUG METRICS REPORT         ")
        print("=============================================")
        print("[ STATIONARY HEIGHTS ]")
        print(f"  First Frame Height:  {results['initial_height_mm']:>7.2f} mm")
        print(f"  Last Frame Height:   {results['final_height_mm']:>7.2f} mm\n")
        
        print("[ TRANSIT METRICS ]")
        print(f"  Entry Frame:         {results['entry_frame']:>7}")
        print(f"  Exit Frame:          {results['exit_frame']:>7}")
        print(f"  Transit Time:        {results['transit_time_sec']:>7.4f} s")
        print(f"  Avg Velocity:        {results['velocity_mps']:>7.4f} m/s")
        print(f"  Plug Length:         {results['length_mm']:>7.2f} mm")
        print(f"  Max Dynamic Height:  {results['max_height_mm']:>7.2f} mm\n")
        
        print("[ FRONT EDGE GEOMETRY ]")
        print(f"  Front Angle:         {results['front_angle_deg']:>7.2f} degrees")
        print(f"  Edge Slope:          {results['edge_slope']:>7.4f}")
        print("=============================================\n")