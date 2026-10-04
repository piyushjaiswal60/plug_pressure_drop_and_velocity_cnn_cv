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

def calculate_px_to_mm(config):
    p1 = np.array(config.get('p1', [0,0]))
    p2 = np.array(config.get('p2', [0,0]))
    dist_mm = config.get('dist_mm', 0)
    pixel_dist = np.linalg.norm(p1 - p2)
    return dist_mm / pixel_dist if pixel_dist != 0 else 0.0

def get_robust_median_height(mask, px_to_mm):
    coords = np.column_stack(np.where(mask == 255))
    if coords.size == 0: return 0.0
    
    unique_x = np.unique(coords[:, 1])
    heights = [np.max(coords[coords[:, 1] == x][:, 0]) - np.min(coords[coords[:, 1] == x][:, 0]) + 1 for x in unique_x]
    
    return float(np.median(heights) * px_to_mm) if heights else 0.0

def get_front_edge_points(flow, mask):
    if mask is None or flow is None or mask.size == 0:
        return np.array([]), np.zeros((1,1), dtype=bool)

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

def run_unified_analysis(video_path, roi, calibration_config, video_props):
    if not roi or len(roi) != 4 or roi[2] <= 0 or roi[3] <= 0:
        print("Error: Invalid ROI. Please run pipeline_setup.py again.")
        return None

    masker_module = importlib.import_module("03a_unet_masker")
    masker = masker_module.PlugMasker()

    flow_module = importlib.import_module("03b_flow_engine")
    engine = flow_module.RAFTFlowEngine()

    px_to_mm = calculate_px_to_mm(calibration_config)
    fps = video_props.get('fps', 500.0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gate_x = roi[0] + roi[2] // 2

    initial_height_mm = 0.0
    final_height_mm = 0.0
    entry_frame = None
    exit_frame = None
    velocities = []
    consecutive_frames = 0
    min_consecutive_frames = 25
    max_height_pixels = 0
    
    final_angle = 0.0
    final_slope = 0.0
    last_mask = None
    last_frame_rgb = None
    last_points = None
    last_blob = None
    potential_entry_buffer = []

    print(f"\nStarting Unified Pipeline (Gate X: {gate_x}, Scale: {px_to_mm:.4f} mm/px)...")
    print("Processing video stream...")

    frame_idx = 0
    ret, prev_frame = cap.read()
    if ret:
        first_mask = masker.create_mask(prev_frame, roi=roi)
        initial_height_mm = get_robust_median_height(first_mask, px_to_mm)
    
    while True:
        ret, curr_frame = cap.read()
        if not ret: break

        mask = masker.create_mask(curr_frame, roi=roi)
        flow = engine.compute_flow(prev_frame, curr_frame, roi=roi, mask=mask)

        # Safely handle flow generation failure
        if flow is None:
            prev_frame = curr_frame
            frame_idx += 1
            continue

        if exit_frame is None:
            rel_gate_x = (gate_x - roi[0])
            if 0 <= rel_gate_x < flow.shape[1]:
                gate_velocities = flow[:, rel_gate_x, :]
                gate_mag = np.sqrt(gate_velocities[..., 0]**2 + gate_velocities[..., 1]**2)

                moving_pixels = gate_mag > 0.5
                current_height = np.sum(moving_pixels)
                moving_plug_present = current_height > 0
                
                if moving_plug_present:
                    if entry_frame is None:
                        points, blob = get_front_edge_points(flow, mask)
                        if points.size > 0:
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

                            limit = min(5, len(potential_entry_buffer))
                            if limit > 0:
                                data = potential_entry_buffer[limit - 1]
                                last_points = data['points']
                                last_blob = data['blob']
                                last_mask = data['mask']
                                last_frame_rgb = data['frame_rgb']

                                if last_points is not None and last_points.size > 0:
                                    px_X = last_points[:, 0]
                                    px_Y = last_points[:, 1]
                                    if len(np.unique(px_Y)) > 1:
                                        fit_m_px, fit_c_px = np.polyfit(px_Y, px_X, 1)
                                        final_slope = 1.0 / fit_m_px if fit_m_px != 0 else 9999.0
                                        final_angle = abs(np.degrees(np.arctan(final_slope)))

                            potential_entry_buffer.clear()
                    
                    if entry_frame is not None:
                        max_height_pixels = max(max_height_pixels, current_height)
                        valid_v = gate_mag[moving_pixels]
                        if valid_v.size > 0:
                            velocities.append(np.median(valid_v))

                        if current_height < 0.8 * max_height_pixels:
                            exit_frame = frame_idx
                            print(f" -> Bulk plug exit detected at frame {exit_frame} (Height: {current_height}, Max Height: {max_height_pixels})")
                            break 
                else:
                    if entry_frame is None: 
                        consecutive_frames = 0
                        potential_entry_buffer.clear()

        prev_frame = curr_frame
        frame_idx += 1

    print(" -> Core metrics gathered. Skipping to end for final height calculation...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame_last = cap.read()
    if ret:
        mask_last = masker.create_mask(frame_last, roi=roi)
        final_height_mm = get_robust_median_height(mask_last, px_to_mm)

    cap.release()

    if entry_frame is None or exit_frame is None:
        print("\nError: Plug transit was not fully captured in this video.")
        return None

    avg_v_px_frame = np.mean(velocities) if velocities else 0
    velocity_mps = (avg_v_px_frame * px_to_mm * fps) / 1000.0
    transit_time = (exit_frame - entry_frame) / fps
    length_mm = velocity_mps * transit_time * 1000.0
    max_height_mm = max_height_pixels * px_to_mm
    
    if last_mask is not None and last_frame_rgb is not None and last_points is not None and last_points.size > 0:
        plt.figure(figsize=(12, 8))
        plt.imshow(last_frame_rgb)

        x_roi, y_roi, w_roi, h_roi = roi
        h_mask, w_mask = last_mask.shape
        full_overlay = np.zeros((last_frame_rgb.shape[0], last_frame_rgb.shape[1], 4))
        
        # Safely build ROI overlay
        roi_overlay = np.zeros((h_mask, w_mask, 4))
        stationary = (last_mask > 127) & (~last_blob)
        roi_overlay[stationary] = [1, 0, 0, 0.3] 
        roi_overlay[last_blob] = [1, 1, 0, 0.4]  
        
        full_overlay[y_roi:y_roi+h_mask, x_roi:x_roi+w_mask] = roi_overlay
        plt.imshow(full_overlay)

        px_X, px_Y = last_points[:, 0], last_points[:, 1]
        plt.scatter(px_X + x_roi, px_Y + y_roi, color='blue', s=15, label='Detected Moving Face')

        if len(np.unique(px_Y)) > 1:
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