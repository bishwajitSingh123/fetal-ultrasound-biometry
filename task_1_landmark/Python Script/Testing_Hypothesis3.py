
"""
================================================================
HYPOTHESIS 3: RESNET-UNET TESTER
================================================================
Author: Bishwajit Prasad Singh
Date: January 2026
Task: Evaluate trained model on test set

Metrics: Pixel error, visualization, error distribution
================================================================
"""

import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.model_selection import train_test_split
import argparse


# ==========================================
# HEATMAP UTILITIES
# ==========================================

def create_gaussian_heatmap(size, center, sigma=10):
    """Generate Gaussian heatmap"""
    H, W = size
    cx = min(max(center[0], 0), W - 1)
    cy = min(max(center[1], 0), H - 1)
    
    grid_y, grid_x = np.mgrid[0:H, 0:W]
    dist_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
    heatmap = np.exp(-dist_sq / (2 * sigma ** 2))
    heatmap = heatmap / (heatmap.max() + 1e-8)
    heatmap[heatmap < 0.01] = 0
    
    return heatmap


def heatmap_to_coord(heatmap):
    """Extract coordinate from heatmap"""
    h, w = heatmap.shape
    idx = heatmap.argmax()
    y, x = idx // w, idx % w
    return x, y


# ==========================================
# DATASET
# ==========================================

class FetalLandmarkDataset(torch.utils.data.Dataset):
    """Test dataset"""
    
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
# MODEL ARCHITECTURE (same as training)
# ==========================================

class SpatialAttention(nn.Module):
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


class ResNetUNet(nn.Module):
    def __init__(self, n_class=4, pretrained=False):
        super(ResNetUNet, self).__init__()
        
        resnet = models.resnet34(weights=None)
        
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
# VISUALIZATION
# ==========================================

def visualize_predictions(model, dataset, indices, device, save_path):
    """Visualize predictions for given indices"""
    model.eval()
    
    n_samples = len(indices)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        img_tensor, gt_heatmaps = dataset[idx]
        
        with torch.no_grad():
            img_batch = img_tensor.unsqueeze(0).to(device)
            pred_heatmaps = model(img_batch).cpu().squeeze(0)
        
        gt_coords = [heatmap_to_coord(gt_heatmaps[j].numpy()) for j in range(4)]
        pred_coords = [heatmap_to_coord(pred_heatmaps[j].numpy()) for j in range(4)]
        
        errors = [np.sqrt((g[0]-p[0])**2 + (g[1]-p[1])**2) 
                  for g, p in zip(gt_coords, pred_coords)]
        mean_error = np.mean(errors)
        
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        # Ground truth
        axes[i, 0].imshow(img_np)
        for j, (x, y) in enumerate(gt_coords):
            circle = Circle((x, y), radius=3, color='lime', fill=True)
            axes[i, 0].add_patch(circle)
            axes[i, 0].text(x+5, y-5, f'L{j+1}', color='lime', fontweight='bold')
        axes[i, 0].set_title('Ground Truth', fontweight='bold')
        axes[i, 0].axis('off')
        
        # Predictions
        axes[i, 1].imshow(img_np)
        for j, (x, y) in enumerate(pred_coords):
            circle = Circle((x, y), radius=3, color='red', fill=True)
            axes[i, 1].add_patch(circle)
            axes[i, 1].text(x+5, y-5, f'L{j+1}', color='red', fontweight='bold')
        axes[i, 1].set_title(f'Prediction (Error: {mean_error:.2f}px)', fontweight='bold')
        axes[i, 1].axis('off')
        
        # Overlay
        axes[i, 2].imshow(img_np)
        for j, ((gx, gy), (px, py)) in enumerate(zip(gt_coords, pred_coords)):
            circle_gt = Circle((gx, gy), radius=3, color='lime', fill=True, alpha=0.7)
            axes[i, 2].add_patch(circle_gt)
            circle_pred = Circle((px, py), radius=3, color='red', fill=True, alpha=0.7)
            axes[i, 2].add_patch(circle_pred)
            axes[i, 2].plot([gx, px], [gy, py], 'yellow', linewidth=2, alpha=0.5)
        axes[i, 2].set_title('Overlay (Green=GT, Red=Pred)', fontweight='bold')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Predictions saved: {save_path}")


def plot_error_distribution(all_errors, save_path):
    """Plot error distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(all_errors, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(all_errors), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(all_errors):.2f}px')
    axes[0].axvline(np.median(all_errors), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(all_errors):.2f}px')
    axes[0].set_xlabel('Error (pixels)', fontweight='bold')
    axes[0].set_ylabel('Frequency', fontweight='bold')
    axes[0].set_title('Error Distribution', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Boxplot
    axes[1].boxplot(all_errors, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1].set_ylabel('Error (pixels)', fontweight='bold')
    axes[1].set_title('Error Boxplot', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Error distribution saved: {save_path}")


# ==========================================
# MAIN TESTING
# ==========================================

def main(args):
    print("=" * 60)
    print("🎯 HYPOTHESIS 3: MODEL TESTING")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    df = pd.read_csv(args.csv_path)
    _, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    _, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    test_ds = FetalLandmarkDataset(test_df.reset_index(drop=True), args.image_folder_path)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)
    
    print(f"Test set: {len(test_ds)} images")
    
    # Load model
    model = ResNetUNet(n_class=4, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Model loaded successfully")
    
    # Evaluate
    print("\nEvaluating on test set...")
    all_errors = []
    landmark_errors = [[], [], [], []]
    
    with torch.no_grad():
        for img_tensor, gt_heatmaps in tqdm(test_loader):
            img_tensor = img_tensor.to(device)
            pred_heatmaps = model(img_tensor).cpu().squeeze(0)
            
            gt_coords = [heatmap_to_coord(gt_heatmaps[0][i].numpy()) for i in range(4)]
            pred_coords = [heatmap_to_coord(pred_heatmaps[i].numpy()) for i in range(4)]
            
            errors = [np.sqrt((g[0]-p[0])**2 + (g[1]-p[1])**2) 
                      for g, p in zip(gt_coords, pred_coords)]
            
            all_errors.extend(errors)
            for i, err in enumerate(errors):
                landmark_errors[i].append(err)
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    print(f"Total landmarks: {len(all_errors)}")
    print(f"Mean Error: {np.mean(all_errors):.2f} ± {np.std(all_errors):.2f} pixels")
    print(f"Median Error: {np.median(all_errors):.2f} pixels")
    print(f"Min Error: {np.min(all_errors):.2f} pixels")
    print(f"Max Error: {np.max(all_errors):.2f} pixels")
    print(f"95th Percentile: {np.percentile(all_errors, 95):.2f} pixels")
    
    print("\nPer-Landmark Errors:")
    landmark_names = ['OFD-1', 'OFD-2', 'BPD-1', 'BPD-2']
    for i, name in enumerate(landmark_names):
        mean_err = np.mean(landmark_errors[i])
        median_err = np.median(landmark_errors[i])
        print(f"  {name}: Mean={mean_err:.2f}px, Median={median_err:.2f}px")
    
    # Visualize predictions
    print("\nGenerating visualizations...")
    sample_indices = [0, 10, 20, 30, 40, 50]
    visualize_predictions(model, test_ds, sample_indices, device, args.pred_save_path)
    
    # Plot error distribution
    plot_error_distribution(all_errors, args.error_save_path)
    
    print("\n" + "=" * 60)
    print("✅ TESTING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hypothesis 3 Testing')
    
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--image_folder_path', type=str, required=True, help='Path to images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--pred_save_path', type=str, default='hypothesis_3_predictions_visualization.png')
    parser.add_argument('--error_save_path', type=str, default='hypothesis_3_error_distribution.png')
    
    args = parser.parse_args()
    main(args)
