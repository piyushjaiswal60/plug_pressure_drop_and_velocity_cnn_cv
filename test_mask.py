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

    # 2. Load video
    video_folder = "C:/Users/piyus/OneDrive/Desktop/pfinal/video"
    import glob
    videos = glob.glob(os.path.join(video_folder, "*.*"))

    if not videos:
        print(f"Error: No videos found in {video_folder}")
        return

    video_path = videos[0]
    print(f"Using video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    # Import U-Net Masker
    masker_module = importlib.import_module("03a_unet_masker")
    masker = masker_module.PlugMasker()

    while True:
        try:
            frame_num = input("\nEnter frame number to see mask (or 'q' to quit): ").strip()
            if frame_num.lower() == 'q':
                break

            frame_idx = int(frame_num)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                print(f"Error: Could not read frame {frame_idx}.")
                continue

            mask = masker.create_mask(frame, roi=roi)

            # Visualization
            x, y, w, h = roi
            cropped_orig = frame[y:y+h, x:x+w]
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            comparison = np.hstack((cropped_orig, mask_bgr))

            # Add frame number text to the image
            text = f"Frame: {frame_idx}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            filename = "mask_verification.jpg"
            cv2.imwrite(filename, comparison)
            print(f"Success! Saved mask for frame {frame_idx} to {filename}")

        except ValueError:
            print("Invalid input. Please enter an integer frame number.")
        except Exception as e:
            print(f"An error occurred: {e}")

    cap.release()

if __name__ == "__main__":
    verify_masking()
