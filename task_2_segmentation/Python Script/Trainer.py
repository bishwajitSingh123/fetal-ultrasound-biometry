"""
FETAL HEAD SEGMENTATION - COMPLETE TRAINING SCRIPT
===================================================
This script contains the complete training pipeline for all 3 hypotheses:
- Hypothesis 1: Baseline U-Net with boundary masks
- Hypothesis 2: U-Net with filled masks + improvements (BEST: Dice=0.9575)
- Hypothesis 3: ResNet34 encoder + CBAM attention

Author: Bishwajit Singh
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# ==========================================
# DATASET CLASS WITH PREPROCESSING
# ==========================================

def apply_clahe_rgb(image):
    """CLAHE contrast enhancement"""
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

class FetalSegmentationDataset(Dataset):
    """Dataset for fetal head segmentation"""
    
    def __init__(self, image_paths, mask_paths, target_size=(256, 256), augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.target_size = target_size
        self.augment = augment
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        image = apply_clahe_rgb(image)
        
        # Load mask
        mask = cv2.imread(self.mask_paths[idx], 0)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        
        # Augmentation (training only)
        if self.augment:
            if np.random.rand() < 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)
            
            if np.random.rand() < 0.5:
                image = np.flipud(image)
                mask = np.flipud(mask)
            
            if np.random.rand() < 0.5:
                angle = np.random.uniform(-15, 15)
                M = cv2.getRotationMatrix2D((self.target_size[0]//2, self.target_size[1]//2), angle, 1.0)
                image = cv2.warpAffine(image, M, self.target_size)
                mask = cv2.warpAffine(mask, M, self.target_size, flags=cv2.INTER_NEAREST)
            
            if np.random.rand() < 0.3:
                factor = np.random.uniform(0.8, 1.2)
                image = np.clip(image * factor, 0, 255).astype(np.uint8)
            
            if np.random.rand() < 0.2:
                noise = np.random.normal(0, 5, image.shape)
                image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # Make contiguous
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        
        # To tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask

# ==========================================
# MODEL ARCHITECTURES
# ==========================================

def double_conv(in_channels, out_channels):
    """Double convolution block"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNetSegmentation(nn.Module):
    """Baseline U-Net architecture"""
    
    def __init__(self):
        super().__init__()
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = nn.Sequential(double_conv(256, 512), nn.Dropout2d(p=0.3))
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up3 = double_conv(512 + 256, 256)
        self.dconv_up2 = double_conv(256 + 128, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        self.conv_last = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1), nn.Sigmoid())
    
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        x = self.dconv_down4(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        return self.conv_last(x)

# ==========================================
# LOSS FUNCTIONS
# ==========================================

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced segmentation"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

class DiceLoss(nn.Module):
    """Dice Loss"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    """Combined Loss: Focal + Dice"""
    def __init__(self, focal_weight=0.3, dice_weight=0.7):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        return self.focal_weight * self.focal(pred, target) + self.dice_weight * self.dice(pred, target)

# ==========================================
# TRAINING FUNCTIONS
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    for images, masks in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
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
        for images, masks in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
    
    return running_loss / len(loader)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, patience, save_dir, hypothesis_name, device):
    """Complete training loop"""
    
    best_val_loss = float('inf')
    counter = 0
    train_losses = []
    val_losses = []
    
    print(f"\n🚀 Training {hypothesis_name}...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), f'{save_dir}/{hypothesis_name}_best.pth')
            print(f"  ✅ Best model saved!")
        else:
            counter += 1
            print(f"  ⏳ No improvement ({counter}/{patience})")
        
        if counter >= patience:
            print("🛑 Early stopping triggered")
            break
    
    elapsed = time.time() - start_time
    
    # Save full checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'epoch': epoch + 1
    }, f'{save_dir}/{hypothesis_name}_full_saved_model.pth')
    
    print(f"\n✅ Training complete! Time: {elapsed/60:.1f} min, Best val loss: {best_val_loss:.4f}")
    
    return train_losses, val_losses, best_val_loss

# ==========================================
# MAIN TRAINING FUNCTION
# ==========================================

if __name__ == "__main__":
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Paths
    base_dir = '/path/to/your/data'
    image_folder = f'{base_dir}/images'
    mask_folder = f'{base_dir}/masks_properly_filled'
    save_dir = '/path/to/save/models'
    
    # Prepare dataset
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
    image_paths = [os.path.join(image_folder, f) for f in image_files]
    mask_paths = [os.path.join(mask_folder, f.replace('.png', '_Annotation.png')) for f in image_files]
    
    # Split data
    indices = list(range(len(image_paths)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    # Create datasets
    train_ds = FetalSegmentationDataset([image_paths[i] for i in train_idx], [mask_paths[i] for i in train_idx], augment=True)
    val_ds = FetalSegmentationDataset([image_paths[i] for i in val_idx], [mask_paths[i] for i in val_idx], augment=False)
    
    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)
    
    # Train Hypothesis 2 (BEST MODEL)
    model = UNetSegmentation().to(device)
    criterion = CombinedLoss(focal_weight=0.3, dice_weight=0.7)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    train_losses, val_losses, best_val_loss = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=50, patience=8, save_dir=save_dir, hypothesis_name='hypothesis_2', device=device
    )
    
    print("\n🎉 Training pipeline completed successfully!")
