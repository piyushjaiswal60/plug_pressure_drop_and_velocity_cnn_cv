"""
03b_flow_engine.py
Implements a robust Optical Flow engine using the Farneback method.
Provides precise velocity vectors for every pixel in the ROI.
"""

import numpy as np
import cv2

class RAFTFlowEngine:
    def __init__(self, model_type='small', device=None):
        print("Initializing Robust Farneback Optical Flow Engine (High-Speed Profile)...")
        self.device = device

    def compute_flow(self, frame1, frame2, roi=None, mask=None):
        if roi:
            x, y, w, h = roi
            x, y = max(0, x), max(0, y)
            w = max(1, min(w, frame1.shape[1] - x))
            h = max(1, min(h, frame1.shape[0] - y))
            
            f1 = frame1[y:y+h, x:x+w]
            f2 = frame2[y:y+h, x:x+w]
            
            if mask is not None:
                if mask.shape == frame1.shape[:2]:
                    mask = mask[y:y+h, x:x+w]
        else:
            f1, f2 = frame1, frame2

        if f1.size == 0 or f2.size == 0 or f1.shape[0] == 0 or f1.shape[1] == 0:
            return None

        prev_gray = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        prev_gray = clahe.apply(prev_gray)
        next_gray = clahe.apply(next_gray)

        # HIGH-SPEED OPTIMIZATION: 
        # Increased levels (8) and winsize (65) to track large frame-to-frame 
        # displacements in fast-moving plugs without fragmenting the velocity field.
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None,
            pyr_scale=0.5, levels=8, winsize=65,
            iterations=5, poly_n=7, poly_sigma=1.5,
            flags=0
        )

        if mask is not None:
            if mask.shape[:2] != flow.shape[:2]:
                mask = cv2.resize(mask, (flow.shape[1], flow.shape[0]))
            
            mask_bool = (mask > 127)
            flow[~mask_bool] = 0
        else:
            u_median = np.median(flow[..., 0])
            v_median = np.median(flow[..., 1])
            flow[..., 0] -= u_median
            flow[..., 1] -= v_median

        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        threshold = 0.2

        if mask is not None:
            flow[(mag < threshold) | (~mask_bool)] = 0
        else:
            flow[mag < threshold] = 0

        return flow

    def get_median_velocity(self, flow, mask=None):
        u = flow[:, :, 0]
        v = flow[:, :, 1]

        if mask is not None:
            mask_bool = (mask > 127)
            u_vals = u[mask_bool]
            v_vals = v[mask_bool]
        else:
            u_vals = u.flatten()
            v_vals = v.flatten()

        if len(u_vals) == 0:
            return 0.0, 0.0

        return np.median(u_vals), np.median(v_vals)