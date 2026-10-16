import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.ndimage import distance_transform_edt

# --- 1. Configuration Grid ---
IMG_HEIGHT = 256
IMG_WIDTH = 256
EPOCHS = 50
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = 'dataunettr/new_images_cvat_upload_dataset' 
MASK_DIR = 'dataunettr/new_mask_cavt'   

HYPERPARAMETERS = {
    "base_channels": [16, 32, 64],
    "learning_rates": [1e-3, 1e-4]
}

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

# --- 3. Dynamic U-Net Architecture ---
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

class DynamicUNet(nn.Module):
    def __init__(self, base_c):
        super().__init__()
        self.conv1 = DoubleConv(1, base_c)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(base_c, base_c * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(base_c * 2, base_c * 4)
        
        self.up1 = nn.ConvTranspose2d(base_c * 4, base_c * 2, 2, stride=2)
        self.conv4 = DoubleConv(base_c * 4, base_c * 2)
        
        self.up2 = nn.ConvTranspose2d(base_c * 2, base_c, 2, stride=2)
        self.conv5 = DoubleConv(base_c * 2, base_c)
        
        self.out_conv = nn.Conv2d(base_c, 1, 1)

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

# --- 4. Metric Calculations ---
def calculate_metrics(preds, labels, threshold=0.5):
    preds_b = (preds > threshold).float()
    
    # IoU and Dice
    intersection = (preds_b * labels).sum()
    union = preds_b.sum() + labels.sum() - intersection
    iou = (intersection / (union + 1e-6)).item()
    dice = (2 * intersection / (preds_b.sum() + labels.sum() + 1e-6)).item()
    
    # Boundary MAE via Euclidean Distance Transform
    preds_np = preds_b.squeeze(1).cpu().numpy()
    labels_np = labels.squeeze(1).cpu().numpy()
    mae_list = []
    
    for i in range(preds_np.shape[0]):
        pred_mask = (preds_np[i] * 255).astype(np.uint8)
        label_mask = (labels_np[i] * 255).astype(np.uint8)
        
        pred_edge = cv2.Canny(pred_mask, 100, 200)
        label_edge = cv2.Canny(label_mask, 100, 200)
        
        if pred_edge.sum() == 0 and label_edge.sum() == 0:
            mae_list.append(0.0)
        elif pred_edge.sum() == 0 or label_edge.sum() == 0:
            mae_list.append(IMG_HEIGHT) # Severe penalty for completely missing boundaries
        else:
            # Create distance map from true boundary
            dt = distance_transform_edt(label_edge == 0)
            # Average distance at predicted boundary pixels
            mae = dt[pred_edge > 0].mean()
            mae_list.append(mae)
            
    b_mae = np.mean(mae_list) if mae_list else 0.0
    return iou, dice, b_mae

# --- 5. Tuning Pipeline ---
if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    
    full_dataset = PlugDataset(IMAGE_DIR, MASK_DIR)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Total Images: {len(full_dataset)} | Training: {train_size} | Validation: {val_size}")

    results = []
    best_overall_iou = 0.0

    trial = 1
    total_trials = len(HYPERPARAMETERS["base_channels"]) * len(HYPERPARAMETERS["learning_rates"])

    for channels in HYPERPARAMETERS["base_channels"]:
        for lr in HYPERPARAMETERS["learning_rates"]:
                
            print(f"\n--- Trial {trial}/{total_trials} | Channels: {channels} | LR: {lr} ---")
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            model = DynamicUNet(base_c=channels).to(DEVICE)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            best_trial_iou = 0.0
            best_trial_dice = 0.0
            best_trial_mae = float('inf')
            
            for epoch in range(EPOCHS):
                model.train()
                for images, masks in train_loader:
                    images, masks = images.to(DEVICE), masks.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()
                
                model.eval()
                val_iou_sum, val_dice_sum, val_mae_sum = 0, 0, 0
                with torch.no_grad():
                    for images, masks in val_loader:
                        images, masks = images.to(DEVICE), masks.to(DEVICE)
                        outputs = model(images)
                        
                        iou, dice, mae = calculate_metrics(outputs, masks)
                        val_iou_sum += iou
                        val_dice_sum += dice
                        val_mae_sum += mae
                        
                avg_iou = val_iou_sum / len(val_loader)
                avg_dice = val_dice_sum / len(val_loader)
                avg_mae = val_mae_sum / len(val_loader)
                
                if avg_iou > best_trial_iou:
                    best_trial_iou = avg_iou
                    best_trial_dice = avg_dice
                    best_trial_mae = avg_mae
                    
                    if best_trial_iou > best_overall_iou:
                        best_overall_iou = best_trial_iou
                        torch.save(model.state_dict(), 'best_optimized_unet.pth')

            print(f"Trial Completed. Best Val IoU: {best_trial_iou:.4f} | Dice: {best_trial_dice:.4f} | B-MAE: {best_trial_mae:.2f} px")
            
            results.append({
                "Channels": channels,
                "Learning Rate": lr,
                "Val IoU": best_trial_iou,
                "Val Dice": best_trial_dice,
                "Boundary MAE (px)": best_trial_mae
            })
            trial += 1

    # --- 6. Results Export ---
    print("\n================ HYPERPARAMETER TUNING COMPLETE ================")
    df = pd.DataFrame(results).sort_values(by="Val IoU", ascending=False)
    print(df.to_string(index=False))
    df.to_csv("tuning_results.csv", index=False)
    
    print(f"\nThe absolute best model achieved an IoU of {best_overall_iou:.4f}.")
    print("This optimal model has been saved as 'best_optimized_unet.pth'.")