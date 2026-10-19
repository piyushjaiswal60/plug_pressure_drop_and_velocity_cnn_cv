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
import random
import itertools

# --- 1. Configuration ---
IMG_HEIGHT = 256
IMG_WIDTH = 256
EPOCHS = 100
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = 'dataunettr/new_images_cvat_upload_dataset' 
MASK_DIR = 'dataunettr/new_mask_cavt'   

# Hyperparameter Grid (2 x 2 x 2 x 2 = 16 combinations)
HYPERPARAMETERS = {
    "base_channels": [32, 64],       
    "depths": [3, 5],                
    "dropouts": [0.3, 0.5],          
    "learning_rate": [1e-3, 1e-4]          
}

# --- 2. Advanced Loss Function (BCE + Dice) ---
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        smooth = 1e-6
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1.0 - ((2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))
        return bce_loss + dice_loss

# --- 3. Dataset Loader with Aggressive Augmentation ---
class PlugDataset(Dataset):
    def __init__(self, image_dir, mask_dir, is_train=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        self.images = [f for f in sorted(os.listdir(image_dir)) if f.endswith(('.png', '.jpg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        
        mask_path = os.path.join(self.mask_dir, img_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
        mask = np.where(mask > 127, 1.0, 0.0).astype(np.float32)
        image = image / 255.0

        if self.is_train:
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
                
            # Random Brightness & Contrast Shift
            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2)  
                beta = random.uniform(-0.1, 0.1)  
                image = np.clip(image * alpha + beta, 0, 1)
                
            # Random Gaussian Noise
            if random.random() > 0.5:
                noise = np.random.normal(0, 0.03, image.shape)
                image = np.clip(image + noise, 0, 1)

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# --- 4. Flexible U-Net Architecture ---
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

# --- 5. Metric Calculations ---
def calculate_metrics(preds, labels, threshold=0.5):
    preds_b = (preds > threshold).float()
    
    intersection = (preds_b * labels).sum()
    union = preds_b.sum() + labels.sum() - intersection
    iou = (intersection / (union + 1e-6)).item()
    dice = (2 * intersection / (preds_b.sum() + labels.sum() + 1e-6)).item()
    
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
            mae_list.append(IMG_HEIGHT) 
        else:
            dt = distance_transform_edt(label_edge == 0)
            mae = dt[pred_edge > 0].mean()
            mae_list.append(mae)
            
    b_mae = np.mean(mae_list) if mae_list else 0.0
    return iou, dice, b_mae

# --- 6. Tuning Pipeline ---
if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    
    full_dataset = PlugDataset(IMAGE_DIR, MASK_DIR, is_train=False) 
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset.is_train = True 

    print(f"Total Images: {len(full_dataset)} | Training: {train_size} | Validation: {val_size}")

    # Generate all combinations of hyperparameters
    keys, values = zip(*HYPERPARAMETERS.items())
    grid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    best_overall_iou = 0.0
    total_trials = len(grid_combinations)

    for trial, config in enumerate(grid_combinations, 1):
        c_base = config["base_channels"]
        c_depth = config["depths"]
        c_drop = config["dropouts"]
        c_lr = config["learning_rate"]
        
        print(f"\n--- Trial {trial}/{total_trials} | Channels: {c_base} | Depth: {c_depth} | Dropout: {c_drop} | LR: {c_lr} ---")
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = FlexibleUNet(base_c=c_base, depth=c_depth, dropout_rate=c_drop).to(DEVICE)
        criterion = BCEDiceLoss()
        optimizer = optim.Adam(model.parameters(), lr=c_lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        
        best_trial_iou = 0.0
        best_trial_dice = 0.0
        best_trial_mae = float('inf')
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            
            for images, masks in train_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
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
                    
            avg_train_loss = train_loss / len(train_loader)
            avg_iou = val_iou_sum / len(val_loader)
            avg_dice = val_dice_sum / len(val_loader)
            avg_mae = val_mae_sum / len(val_loader)
            
            print(f"  Epoch {epoch+1:03d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val IoU: {avg_iou:.4f} | Val Dice: {avg_dice:.4f} | B-MAE: {avg_mae:.2f} px")
            
            scheduler.step(avg_iou)
            
            if avg_iou > best_trial_iou:
                best_trial_iou = avg_iou
                best_trial_dice = avg_dice
                best_trial_mae = avg_mae
                
                if best_trial_iou > best_overall_iou:
                    best_overall_iou = best_trial_iou
                    torch.save(model.state_dict(), 'best_optimized_unet.pth')

        print(f"Trial {trial} Completed. Best Val IoU: {best_trial_iou:.4f} | Dice: {best_trial_dice:.4f} | B-MAE: {best_trial_mae:.2f} px")
        
        results.append({
            "Channels": c_base,
            "Depth": c_depth,
            "Dropout": c_drop,
            "Learning Rate": c_lr,
            "Val IoU": best_trial_iou,
            "Val Dice": best_trial_dice,
            "Boundary MAE (px)": best_trial_mae
        })

        # Save intermediate results after each trial completes
        df = pd.DataFrame(results).sort_values(by="Val IoU", ascending=False)
        df.to_csv("tuning_results_advanced.csv", index=False)
        print("--> Progress saved to tuning_results_advanced.csv")

    # --- 7. Results Export ---
    print("\n================ HYPERPARAMETER TUNING COMPLETE ================")
    print(df.to_string(index=False))
    
    print(f"\nThe absolute best model achieved an IoU of {best_overall_iou:.4f}.")
    print("This optimal model has been saved as 'best_optimized_unet.pth'.")