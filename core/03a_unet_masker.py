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

# --- Architecture (Synchronized with tr.py) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class FlexibleUNet(nn.Module):
    def __init__(self, base_c, depth, dropout_rate):
        super().__init__()
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        # Encoder
        in_c = 1
        for i in range(depth):
            out_c = base_c * (2 ** i)
            self.downs.append(DoubleConv(in_c, out_c, dropout_rate))
            self.pools.append(nn.MaxPool2d(2))
            in_c = out_c

        # Bottleneck
        self.bottleneck = DoubleConv(in_c, in_c * 2, dropout_rate)

        # Decoder
        in_c = in_c * 2
        for i in range(depth - 1, -1, -1):
            out_c = base_c * (2 ** i)
            self.ups.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2))
            self.up_convs.append(DoubleConv(in_c, out_c, dropout_rate))
            in_c = out_c

        self.out_conv = nn.Conv2d(base_c, 1, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(len(self.ups)):
            x = self.ups[idx](x)
            skip_x = skip_connections[idx]

            diffY = skip_x.size()[2] - x.size()[2]
            diffX = skip_x.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

            x = torch.cat([skip_x, x], dim=1)
            x = self.up_convs[idx](x)

        return torch.sigmoid(self.out_conv(x))

class PlugMasker:
    def __init__(self, weights_path='models/unet/best_optimized_unet.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Optimized U-Net Masker on {self.device}...")

        # Best architecture from tuning_results_advanced.csv:
        # Channels: 64, Depth: 5, Dropout: 0.3
        self.model = FlexibleUNet(base_c=64, depth=5, dropout_rate=0.3).to(self.device)
        try:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.model.eval()
            print(f"Optimized model weights loaded successfully from {weights_path}")
        except FileNotFoundError:
            print(f"Error: Model weights not found at {weights_path}")
            raise
        except Exception as e:
            print(f"Error loading model weights: {e}")
            raise

    def create_mask(self, frame, roi=None):
        if roi:
            x, y, w, h = roi
            # Ensure ROI stays strictly within frame dimensions
            x, y = max(0, x), max(0, y)
            w = max(1, min(w, frame.shape[1] - x))
            h = max(1, min(h, frame.shape[0] - y))
            cropped = frame[y:y+h, x:x+w]
        else:
            cropped = frame

        if cropped.size == 0:
            return np.zeros((256, 256), dtype=np.uint8)

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (256, 256))
        normalized = resized / 255.0

        input_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor)
            mask = (prediction > 0.5).float()

        mask_np = mask.squeeze().cpu().numpy()
        mask_resized = cv2.resize(mask_np, (cropped.shape[1], cropped.shape[0]))

        binary_mask = (mask_resized * 255).astype(np.uint8)
        return binary_mask
