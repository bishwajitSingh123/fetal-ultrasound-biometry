
"""
================================================================
HYPOTHESIS 3: RESNET-UNET TRAINER
================================================================
Author: Bishwajit Prasad Singh
Date: January 2026
Task: Fetal BPD & OFD Landmark Detection with Advanced Architecture
================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import math
import argparse


# ==========================================
# GAUSSIAN HEATMAP GENERATION
# ==========================================

def create_gaussian_heatmap(size, center, sigma=10):
    """Generate Gaussian heatmap for landmark"""
    H, W = size
    cx = min(max(center[0], 0), W - 1)
    cy = min(max(center[1], 0), H - 1)
    
    grid_y, grid_x = np.mgrid[0:H, 0:W]
    dist_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
    heatmap = np.exp(-dist_sq / (2 * sigma ** 2))
    heatmap = heatmap / (heatmap.max() + 1e-8)
    heatmap[heatmap < 0.01] = 0
    
    return heatmap


# ==========================================
# DATASET CLASS
# ==========================================

class FetalLandmarkDataset(Dataset):
    """Dataset for fetal landmark detection"""
    
    def __init__(self, df, img_dir, target_size=(256, 256), sigma=10):
        self.df = df
        self.img_dir = img_dir
        self.target_size = target_size
        self.sigma = sigma
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        h, w, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        raw_landmarks = self.df.iloc[idx, 1:9].values.astype('float32').reshape(-1, 2)
        
        scale_x = self.target_size[1] / w
        scale_y = self.target_size[0] / h
        
        scaled_landmarks = []
        for (x, y) in raw_landmarks:
            scaled_landmarks.append([x * scale_x, y * scale_y])
        
        image_resized = cv2.resize(image, self.target_size)
        image_tensor = torch.tensor(image_resized).permute(2, 0, 1).float() / 255.0
        
        heatmaps = []
        for point in scaled_landmarks:
            hm = create_gaussian_heatmap(self.target_size, point, self.sigma)
            heatmaps.append(hm)
        
        heatmaps_tensor = torch.tensor(np.array(heatmaps)).float()
        
        return image_tensor, heatmaps_tensor


# ==========================================
# ATTENTION MODULES
# ==========================================

class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                             padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention.expand_as(x)


# ==========================================
# RESNET-UNET MODEL
# ==========================================

class ResNetUNet(nn.Module):
    """ResNet-34 + U-Net Decoder with Attention"""
    
    def __init__(self, n_class=4, pretrained=True):
        super(ResNetUNet, self).__init__()
        
        resnet = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.att1 = ChannelAttention(64)
        self.att2 = ChannelAttention(128)
        self.att3 = ChannelAttention(256)
        
        self.decoder4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.decoder3 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.spatial_att = SpatialAttention()
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_class, kernel_size=1)
        )
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        d4 = self.decoder4(e4)
        d4 = self.upsample(d4)
        e3_att = self.att3(e3)
        d4 = torch.cat([d4, e3_att], dim=1)
        
        d3 = self.decoder3(d4)
        d3 = self.upsample(d3)
        e2_att = self.att2(e2)
        d3 = torch.cat([d3, e2_att], dim=1)
        
        d2 = self.decoder2(d3)
        d2 = self.upsample(d2)
        e1_att = self.att1(e1)
        d2 = torch.cat([d2, e1_att], dim=1)
        
        d1 = self.decoder1(d2)
        d1 = self.upsample(d1)
        d1 = self.upsample(d1)
        
        d1 = self.spatial_att(d1)
        out = self.final_conv(d1)
        
        return out


# ==========================================
# ADAPTIVE WING LOSS
# ==========================================

class AdaptiveWingLoss(nn.Module):
    """Adaptive Wing Loss for Robust Landmark Detection"""
    
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
    
    def forward(self, pred, target):
        delta = (target - pred).abs()
        
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, 
                                              self.alpha - target))) *             (self.alpha - target) * torch.pow(self.theta / self.epsilon, 
                                              self.alpha - target - 1) *             (1 / self.epsilon)
        
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(
            self.theta / self.epsilon, self.alpha - target))
        
        loss = torch.where(
            delta < self.theta,
            self.omega * torch.log(1 + torch.pow(delta / self.epsilon, 
                                                  self.alpha - target)),
            A * delta - C
        )
        
        return loss.mean()


# ==========================================
# TRAINING FUNCTIONS
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train one epoch"""
    model.train()
    running_loss = 0.0
    
    for images, heatmaps in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, heatmaps)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, heatmaps)
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


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """Cosine Annealing with Warmup"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) /                       (self.total_epochs - self.warmup_epochs)
            return [self.min_lr + (base_lr - self.min_lr) * 
                   0.5 * (1 + math.cos(math.pi * progress)) 
                   for base_lr in self.base_lrs]


# ==========================================
# MAIN TRAINING
# ==========================================

def main(args):
    print("=" * 60)
    print("🔥 HYPOTHESIS 3: RESNET-UNET TRAINING")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} images")
    
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    train_ds = FetalLandmarkDataset(train_df.reset_index(drop=True), args.image_folder_path)
    val_ds = FetalLandmarkDataset(val_df.reset_index(drop=True), args.image_folder_path)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    model = ResNetUNet(n_class=4, pretrained=True).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    criterion = AdaptiveWingLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineWarmupScheduler(optimizer, args.warmup_epochs, args.epochs)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    print(f"Loss: Adaptive Wing Loss")
    print(f"Optimizer: AdamW (lr={args.lr})")
    print(f"Epochs: {args.epochs}")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    learning_rates = []
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        print(f"Epoch [{epoch+1:02d}/{args.epochs}] | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"LR: {current_lr:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_save_path)
            print(f"  ✅ Best model saved!")
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Best val loss: {best_val_loss:.4f}")
    
    checkpoint_path = args.model_save_path.replace('.pth', '_checkpoint.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'best_val_loss': best_val_loss,
        'epoch': args.epochs,
    }, checkpoint_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(train_losses, label='Train Loss', linewidth=2, color='#3498db')
    axes[0].plot(val_losses, label='Val Loss', linewidth=2, color='#e74c3c')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training Curves - Hypothesis 3', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(learning_rates, color='#2ecc71', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(args.plot_save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved: {args.plot_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hypothesis 3 Training')
    
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--image_folder_path', type=str, required=True)
    parser.add_argument('--model_save_path', type=str, default='hypothesis_3_resnet_unet.pth')
    parser.add_argument('--plot_save_path', type=str, default='hypothesis_3_training_curves.png')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    
    args = parser.parse_args()
    main(args)
