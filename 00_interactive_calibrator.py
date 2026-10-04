"""
00_interactive_calibrator.py
An interactive tool to calibrate the pixel-to-mm ratio and define the Pipe ROI.
"""

import cv2
import json
import os
import glob

# Global variables
clicked_points = []
roi_rect = None # [x, y, w, h]
img = None

def click_event(event, x, y, flags, param):
    global clicked_points, img
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Point recorded: ({x}, {y})")
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Calibration Tool", img)

def run_calibration(video_path):
    global img, clicked_points, roi_rect
    clicked_points = []
    roi_rect = None

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read video frame from {video_path}")
        return

    img = frame.copy()
    h, w = img.shape[:2]

    # FIX: Ensure window fits on standard screens by setting WINDOW_NORMAL
    cv2.namedWindow("Calibration Tool", cv2.WINDOW_NORMAL)
    
    # Calculate a comfortable window size (max height ~ 800 px)
    target_height = 800
    if h > target_height:
        target_width = int(w * (target_height / h))
        cv2.resizeWindow("Calibration Tool", target_width, target_height)
    else:
        cv2.resizeWindow("Calibration Tool", w, h)

    cv2.setMouseCallback("Calibration Tool", click_event)

    print("\n--- INTERACTIVE CALIBRATION GUIDE ---")
    print("1. Click TWO marks on the scale (e.g., 0mm and 10mm)")
    print("2. Press 'R' to start drawing the ROI rectangle over the pipe")
    print("3. Press 'S' to save and exit, or 'Q' to quit.")
    print("------------------------------------\n")

    while True:
        cv2.imshow("Calibration Tool", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r') or key == ord('R'):
            print("ROI Mode: Click and drag a rectangle over the pipe. Press ENTER or SPACE to confirm.")
            # selectROI uses the existing window, so UI scaling remains intact
            roi = cv2.selectROI("Calibration Tool", img, fromCenter=False, showCrosshair=True)
            roi_rect = roi # [x, y, w, h]
            print(f"ROI selected: {roi_rect}")

        elif key == ord('s') or key == ord('S'):
            if len(clicked_points) >= 2 and roi_rect is not None and roi_rect[2] > 0:
                d_px = roi_rect[3]

                config = {
                    "p1": clicked_points[0],
                    "p2": clicked_points[1],
                    "dist_mm": 10.0,
                    "roi": roi_rect,
                    "d_px": d_px
                }

                try:
                    user_input = input("Enter the actual distance between the scale points in mm (default 10.0): ")
                    dist_mm = float(user_input) if user_input.strip() else 10.0
                    config["dist_mm"] = dist_mm
                except EOFError:
                    print("Warning: Could not read input. Using default 10.0mm.")
                    config["dist_mm"] = 10.0
                except ValueError:
                    print("Invalid input. Using default 10.0mm.")
                    config["dist_mm"] = 10.0

                with open("calibration_config.json", "w") as f:
                    json.dump(config, f, indent=4)

                print("\nCalibration saved to calibration_config.json!")
                break
            else:
                print("Error: Please provide at least 2 scale points and 1 valid ROI rectangle before saving.")

        elif key == ord('q') or key == ord('Q'):
            print("Calibration cancelled by user.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_folder = "C:/Users/piyus/OneDrive/Desktop/pfinal/video"
    import glob
    videos = glob.glob(os.path.join(video_folder, "*.*"))

    if videos:
        video_path = videos[0]
        print(f"Using video for calibration: {video_path}")
        run_calibration(video_path)
    else:
        print(f"Error: No videos found in {video_folder}")