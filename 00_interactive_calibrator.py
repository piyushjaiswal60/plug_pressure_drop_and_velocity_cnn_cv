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

def click_event(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Point recorded: ({x}, {y})")
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Calibration Tool", img)

def run_calibration(video_path):
    global img
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read video frame from {video_path}")
        return

    img = frame.copy()
    cv2.namedWindow("Calibration Tool")
    cv2.setMouseCallback("Calibration Tool", click_event)

    print("\n--- INTERACTIVE CALIBRATION GUIDE ---")
    print("1. Click TWO marks on the scale (e.g., 0mm and 10mm)")
    print("2. Press 'R' to start drawing the ROI rectangle over the pipe")
    print("3. Press 'S' to save and exit, or 'Q' to quit.")
    print("------------------------------------\n")

    while True:
        cv2.imshow("Calibration Tool", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            print("ROI Mode: Click and drag a rectangle over the pipe.")
            roi = cv2.selectROI("Calibration Tool", img, fromCenter=False, showCrosshair=True)
            roi_rect = roi # [x, y, w, h]
            print(f"ROI selected: {roi_rect}")

        elif key == ord('s'):
            if len(clicked_points) >= 2 and roi_rect is not None:
                d_px = roi_rect[3]

                config = {
                    "p1": clicked_points[0],
                    "p2": clicked_points[1],
                    "dist_mm": 10.0,
                    "roi": roi_rect,
                    "d_px": d_px
                }

                # Using a simple input here, but if it fails in CLI,
                # you might need to edit the JSON directly
                try:
                    dist_mm = float(input("Enter the actual distance between the scale points in mm (e.g. 10.0): "))
                    config["dist_mm"] = dist_mm
                except EOFError:
                    print("Warning: Could not read input. Using default 10.0mm.")
                    config["dist_mm"] = 10.0

                with open("calibration_config.json", "w") as f:
                    json.dump(config, f)

                print("\nCalibration saved to calibration_config.json!")
                break
            else:
                print("Error: Please provide 2 scale points and 1 ROI rectangle.")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Attempt to find a video automatically in the videos folder
    video_folder = "C:/Users/piyus/OneDrive/Desktop/pfinal/plastic_bead"
    videos = glob.glob(os.path.join(video_folder, "*.*"))

    if videos:
        video_path = videos[0]
        print(f"Using video for calibration: {video_path}")
        run_calibration(video_path)
    else:
        print(f"Error: No videos found in {video_folder}")
