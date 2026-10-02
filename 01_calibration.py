"""
01_calibration.py
Handles the spatial calibration of the video:
1. Pixel-to-mm ratio calculation using the physical scale.
2. Extraction of pipe inner diameter (D).
"""

import json
import numpy as np

class VideoCalibrator:
    def __init__(self):
        self.px_to_mm = None
        self.pipe_diameter = None

    def calibrate_scale(self, frame, p1, p2, real_distance_mm):
        """
        Calibrates the pixel-to-mm ratio.
        p1, p2: Coordinates (x, y) of two points on the scale.
        real_distance_mm: Known physical distance between those points in mm.
        """
        pixel_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        self.px_to_mm = real_distance_mm / pixel_dist
        return self.px_to_mm

    def calibrate_pipe_diameter(self, frame, d_pixels):
        """
        Sets the pipe inner diameter based on pixel measurement.
        """
        if self.px_to_mm is None:
            raise ValueError("Must calibrate scale before pipe diameter.")

        self.pipe_diameter = d_pixels * self.px_to_mm
        return self.pipe_diameter

    def get_mm(self, pixels):
        """Convert pixel distance to mm."""
        if self.px_to_mm is None:
            return pixels
        return pixels * self.px_to_mm

    def print_summary(self):
        """Prints the calculated calibration values."""
        print("\n--- Calibration Summary ---")
        print(f"Pixel-to-mm Ratio: {self.px_to_mm:.4f} mm/px")
        print(f"Pipe Diameter (D): {self.pipe_diameter:.2f} mm")
        print("---------------------------\n")

if __name__ == "__main__":
    # This allows you to run 'python 01_calibration.py' to see current values
    try:
        with open("calibration_config.json", "r") as f:
            config = json.load(f)

        cal = VideoCalibrator()
        cal.calibrate_scale(None, config['p1'], config['p2'], config['dist_mm'])
        cal.calibrate_pipe_diameter(None, config['d_px'])
        cal.print_summary()
    except FileNotFoundError:
        print("Error: calibration_config.json not found. Please run 00_interactive_calibrator.py first.")
    except Exception as e:
        print(f"An error occurred: {e}")
