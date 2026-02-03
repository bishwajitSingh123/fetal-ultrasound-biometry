# ==========================================
# TRAINER.PY - Hypothesis 1: Baseline U-Net
# Fetal Landmark Detection (BPD & OFD)
# Author: Bishwajit Prasad Singh
# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def create_gaussian_heatmap(size, center, sigma=10):
    """
    Generate Gaussian heatmap for landmark localization.
    
    Args:
        size: (H, W) - Output heatmap size
        center: (x, y) - Landmark center in image coordinates
        sigma: Gaussian kernel standard deviation
    
    Returns:
        heatmap: 2D numpy array with Gaussian peak at center
    """
    H, W = size
    
    # Clamp center coordinates inside image bounds
    cx = min(max(center[0], 0), W - 1)
    cy = min(max(center[1], 0), H - 1)
    
    # Create coordinate grid
    grid_y, grid_x = np.mgrid[0:H, 0:W]
    
    # Calculate squared distance from center
    dist_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
    
    # Generate Gaussian heatmap
    heatmap = np.exp(-dist_sq / (2 * sigma ** 2))
    heatmap = heatmap / (heatmap.max() + 1e-8)  # Normalize
    
    # Threshold small values to zero
    heatmap[heatmap < 0.01] = 0
    
    return heatmap


# ==========================================
# DATASET CLASS
# ==========================================

class FetalLandmarkDataset(Dataset):
    """
    Dataset for fetal ultrasound landmark detection with heatmap targets.
    
    Landmarks (4 points):
    - OFD-1, OFD-2: Occipitofrontal diameter endpoints
    - BPD-1, BPD-2: Biparietal diameter endpoints
    """
    
    def __init__(self, df, img_dir, target_size=(256, 256), sigma=10):
        """
        Args:
            df: DataFrame with columns [image_name, ofd_1_x, ofd_1_y, ..., bpd_2_y]
            img_dir: Path to images folder
            target_size: Resize images to (H, W)
            sigma: Gaussian heatmap sigma
        """
        self.df = df
        self.img_dir = img_dir
        self.target_size = target_size
        self.sigma = sigma
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        h, w, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract landmarks (CSV order: ofd1_x, ofd1_y, ofd2_x, ofd2_y, bpd1_x, bpd1_y, bpd2_x, bpd2_y)
        raw_landmarks = self.df.iloc[idx, 1:9].values.astype('float32').reshape(-1, 2)
        
        # Scale landmarks to target size
        scale_x = self.target_size[1] / w
        scale_y = self.target_size[0] / h
        
        scaled_landmarks = []
        for (x, y) in raw_landmarks:
            scaled_landmarks.append([x * scale_x, y * scale_y])
        
        # Resize image
        image_resized = cv2.resize(image, self.target_size)
        image_tensor = torch.tensor(image_resized).permute(2, 0, 1).float() / 255.0
        
        # Generate 4 heatmaps (one per landmark)
        heatmaps = []
        for point in scaled_landmarks:
            hm = create_gaussian_heatmap(self.target_size, point, self.sigma)
            heatmaps.append(hm)
        
        heatmaps_tensor = torch.tensor(np.array(heatmaps)).float()
        
        return image_tensor, heatmaps_tensor


# ==========================================
# MODEL ARCHITECTURE
# ==========================================

def double_conv(in_channels, out_channels):
    """Double convolution block: Conv-BN-ReLU-Conv-BN-ReLU"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    """
    U-Net architecture for heatmap regression.
    
    Encoder: 3 -> 64 -> 128 -> 256 -> 512
    Decoder: 512 -> 256 -> 128 -> 64 -> 4 (heatmaps)
    """
    
    def __init__(self, n_class=4):
        super().__init__()
        
        # Encoder (Downsampling)
        self.dconv_down1 = double_conv(3, 64)      # 256x256
        self.dconv_down2 = double_conv(64, 128)    # 128x128
        self.dconv_down3 = double_conv(128, 256)   # 64x64
        self.dconv_down4 = double_conv(256, 512)   # 32x32 (bottleneck)
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Decoder (Upsampling)
        self.dconv_up3 = double_conv(512 + 256, 256)  # 512 from bottleneck + 256 from encoder
        self.dconv_up2 = double_conv(256 + 128, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        # Output layer
        self.conv_last = nn.Conv2d(64, n_class, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        conv1 = self.dconv_down1(x)    # [B, 64, 256, 256]
        x = self.maxpool(conv1)         # [B, 64, 128, 128]
        
        conv2 = self.dconv_down2(x)    # [B, 128, 128, 128]
        x = self.maxpool(conv2)         # [B, 128, 64, 64]
        
        conv3 = self.dconv_down3(x)    # [B, 256, 64, 64]
        x = self.maxpool(conv3)         # [B, 256, 32, 32]
        
        x = self.dconv_down4(x)        # [B, 512, 32, 32] <- Bottleneck
        
        # Decoder
        x = self.upsample(x)            # [B, 512, 64, 64]
        x = torch.cat([x, conv3], dim=1)  # [B, 768, 64, 64]
        x = self.dconv_up3(x)          # [B, 256, 64, 64]
        
        x = self.upsample(x)            # [B, 256, 128, 128]
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)          # [B, 128, 128, 128]
        
        x = self.upsample(x)            # [B, 128, 256, 256]
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)          # [B, 64, 256, 256]
        
        out = self.conv_last(x)        # [B, 4, 256, 256]
        
        return out


# ==========================================
# TRAINING FUNCTIONS
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    
    for images, heatmaps in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, heatmaps)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, heatmaps in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, heatmaps)
            
            running_loss += loss.item()
    
    return running_loss / len(loader)


# ==========================================
# MAIN TRAINING CODE
# ==========================================

if __name__ == '__main__':
    
    print("=" * 60)
    print("🔥 TRAINING - HYPOTHESIS 1: BASELINE U-NET")
    print("=" * 60)
    
    # ===== CONFIGURE PATHS (MODIFY THESE) =====
    csv_path = "path/to/role_challenge_dataset_ground_truth.csv"
    image_folder_path = "path/to/images/"
    save_dir = "path/to/save/models/"
    
    # ===== LOAD DATA =====
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} images")
    
    # ===== TRAIN/VAL/TEST SPLIT =====
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # ===== CREATE DATASETS =====
    train_ds = FetalLandmarkDataset(train_df.reset_index(drop=True), image_folder_path)
    val_ds = FetalLandmarkDataset(val_df.reset_index(drop=True), image_folder_path)
    
    # ===== CREATE DATALOADERS =====
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)
    
    # ===== SETUP DEVICE =====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # ===== INITIALIZE MODEL =====
    model = UNet(n_class=4).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ===== LOSS & OPTIMIZER =====
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # ===== TRAINING CONFIGURATION =====
    num_epochs = 50
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    start_time = time.time()
    
    # ===== TRAINING LOOP =====
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 
                      os.path.join(save_dir, 'hypothesis_1_best.pth'))
            print(f"  ✅ Best model saved!")
    
    elapsed = time.time() - start_time
    
    # ===== SAVE FULL CHECKPOINT =====
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'epoch': num_epochs
    }, os.path.join(save_dir, 'hypothesis_1_full_saved_model.pth'))
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Training time: {elapsed/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Model saved to: {save_dir}")
    print("=" * 60)
    
    # ===== PLOT TRAINING CURVES =====
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Curves - Hypothesis 1: Baseline U-Net', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    print("\n📊 Training curves saved!")
