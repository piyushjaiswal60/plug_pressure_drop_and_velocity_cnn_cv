import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- 1. Configuration ---
IMG_HEIGHT = 256
IMG_WIDTH = 256
EPOCHS = 50
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = 'data/new_images_cvat_upload_dataset' 
MASK_DIR = 'data/new_mask_cavt'   

# --- 2. Custom Dataset Loader ---
class PlugDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = [f for f in sorted(os.listdir(image_dir)) if f.endswith(('.png', '.jpg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        mask_path = os.path.join(self.mask_dir, img_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
        mask = np.where(mask > 127, 1.0, 0.0)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# --- 3. U-Net Architecture ---
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

# --- 4. Training Loop ---
if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    
    if not os.path.exists(IMAGE_DIR) or not os.path.exists(MASK_DIR):
        print("Error: Could not find 'data/images' or 'data/masks'. Please create them and add your data.")
        exit()

    dataset = PlugDataset(IMAGE_DIR, MASK_DIR)
    print(f"Found {len(dataset)} images for training.")
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = UNet().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for images, masks in dataloader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {epoch_loss/len(dataloader):.4f}")
        
    print("Saving trained model...")
    torch.save(model.state_dict(), '_unet.pth')
    print("Training complete! You can now run your prediction script.")