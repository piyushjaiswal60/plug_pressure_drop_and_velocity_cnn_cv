"""
03a_unet_masker.py
Implements a U-Net based plug segmentation masker.
Loads a pre-trained model to create binary masks of the plug in video frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

# --- Architecture (Must match tr.py exactly) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = DoubleConv(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv5 = DoubleConv(64, 32)
        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)

        u1 = self.up1(c3)
        diffY = c2.size()[2] - u1.size()[2]
        diffX = c2.size()[3] - u1.size()[3]
        u1 = F.pad(u1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        concat1 = torch.cat([u1, c2], dim=1)
        c4 = self.conv4(concat1)

        u2 = self.up2(c4)
        diffY = c1.size()[2] - u2.size()[2]
        diffX = c1.size()[3] - u2.size()[3]
        u2 = F.pad(u2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        concat2 = torch.cat([u2, c1], dim=1)
        c5 = self.conv5(concat2)

        return torch.sigmoid(self.out_conv(c5))

class PlugMasker:
    def __init__(self, weights_path='best_plug_segmentation_unet.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing U-Net Masker on {self.device}...")

        self.model = UNet().to(self.device)
        try:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.model.eval()
            print(f"Model weights loaded successfully from {weights_path}")
        except FileNotFoundError:
            print(f"Error: Model weights not found at {weights_path}")
            raise

    def create_mask(self, frame, roi=None):
        """
        Processes a BGR frame and returns a binary mask of the plug.
        roi: (x, y, w, h) if provided, only that region is processed.
        """
        # 1. ROI Crop
        if roi:
            x, y, w, h = roi
            cropped = frame[y:y+h, x:x+w]
        else:
            cropped = frame

        # 2. Pre-process for U-Net (Gray -> Resize -> Normalize)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (256, 256))
        normalized = resized / 255.0

        # Convert to torch tensor [1, 1, 256, 256]
        input_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        # 3. Inference
        with torch.no_grad():
            prediction = self.model(input_tensor)
            # Convert sigmoid output to binary mask (threshold 0.5)
            mask = (prediction > 0.5).float()

        # 4. Post-process
        mask_np = mask.squeeze().cpu().numpy()
        mask_resized = cv2.resize(mask_np, (cropped.shape[1], cropped.shape[0]))

        # Scale to 0-255 for OpenCV compatibility
        binary_mask = (mask_resized * 255).astype(np.uint8)

        return binary_mask