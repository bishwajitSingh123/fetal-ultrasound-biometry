# ==========================================
# TESTER.PY - Hypothesis 1: Baseline U-Net
# Fetal Landmark Detection - Inference & Evaluation
# Author: Bishwajit Singh
# ==========================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ==========================================
# COPY HELPER FUNCTIONS & CLASSES FROM TRAINER
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


class UNet(nn.Module):
    """U-Net for heatmap regression"""
    
    def __init__(self, n_class=4):
        super().__init__()
        
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dconv_up3 = double_conv(512 + 256, 256)
        self.dconv_up2 = double_conv(256 + 128, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, kernel_size=1)
    
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
        
        out = self.conv_last(x)
        
        return out


# ==========================================
# INFERENCE HELPER FUNCTIONS
# ==========================================

def extract_landmarks_from_heatmaps(heatmaps):
    """
    Extract (x, y) coordinates from heatmap peaks.
    
    Args:
        heatmaps: Tensor or numpy array [4, H, W]
    
    Returns:
        landmarks: List of (x, y) tuples
    """
    landmarks = []
    for i in range(4):
        hm = heatmaps[i].cpu().numpy() if torch.is_tensor(heatmaps[i]) else heatmaps[i]
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        landmarks.append((x, y))
    return landmarks


def calculate_distance_error(pred_landmarks, true_landmarks):
    """
    Calculate Euclidean distance error between predicted and true landmarks.
    
    Args:
        pred_landmarks: List of (x, y) predictions
        true_landmarks: List of (x, y) ground truths
    
    Returns:
        errors: List of pixel errors
    """
    errors = []
    for (px, py), (tx, ty) in zip(pred_landmarks, true_landmarks):
        error = np.sqrt((px - tx)**2 + (py - ty)**2)
        errors.append(error)
    return errors


def visualize_predictions(model, test_ds, device, num_samples=6, save_path='predictions.png'):
    """
    Visualize model predictions on test samples.
    
    Creates a grid showing:
    - Column 1: Ground truth landmarks
    - Column 2: Predicted landmarks
    - Column 3: Overlay with error
    """
    model.eval()
    
    test_indices = np.random.choice(len(test_ds), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples*4))
    
    for row, idx in enumerate(test_indices):
        img_t, hm_true = test_ds[idx]
        img_batch = img_t.unsqueeze(0).to(device)
        
        with torch.no_grad():
            hm_pred = model(img_batch).squeeze(0).cpu()
        
        pred_landmarks = extract_landmarks_from_heatmaps(hm_pred)
        true_landmarks = extract_landmarks_from_heatmaps(hm_true)
        errors = calculate_distance_error(pred_landmarks, true_landmarks)
        
        img_np = img_t.permute(1, 2, 0).numpy()
        
        # Column 1: Ground Truth
        axes[row, 0].imshow(img_np)
        for i, (x, y) in enumerate(true_landmarks):
            color = 'red' if i < 2 else 'blue'  # OFD=red, BPD=blue
            axes[row, 0].scatter(x, y, c=color, s=150, marker='o',
                                edgecolors='white', linewidths=2)
        axes[row, 0].set_title("Ground Truth", fontsize=12, fontweight='bold')
        axes[row, 0].axis('off')
        
        # Column 2: Prediction
        axes[row, 1].imshow(img_np)
        for i, (x, y) in enumerate(pred_landmarks):
            color = 'red' if i < 2 else 'blue'
            axes[row, 1].scatter(x, y, c=color, s=150, marker='x', linewidths=3)
        axes[row, 1].set_title("Prediction", fontsize=12, fontweight='bold')
        axes[row, 1].axis('off')
        
        # Column 3: Overlay
        axes[row, 2].imshow(img_np)
        for i, (x, y) in enumerate(true_landmarks):
            axes[row, 2].scatter(x, y, c='green', s=150, marker='o',
                                alpha=0.6, edgecolors='white', linewidths=2,
                                label='True' if i==0 else '')
        for i, (x, y) in enumerate(pred_landmarks):
            axes[row, 2].scatter(x, y, c='red', s=150, marker='x', linewidths=3,
                                label='Pred' if i==0 else '')
        axes[row, 2].set_title(f"Error: {np.mean(errors):.2f}px", 
                              fontsize=12, fontweight='bold')
        axes[row, 2].axis('off')
        if row == 0:
            axes[row, 2].legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Predictions saved: {save_path}")


def plot_error_distribution(all_errors, save_path='error_distribution.png'):
    """Plot error distribution histogram and per-landmark boxplot"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(all_errors.flatten(), bins=30, color='steelblue', 
                 edgecolor='black', alpha=0.7)
    axes[0].axvline(all_errors.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {all_errors.mean():.2f}px')
    axes[0].axvline(np.median(all_errors), color='green', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(all_errors):.2f}px')
    axes[0].set_xlabel('Pixel Error', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Error Distribution (All Landmarks)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)
    
    # Boxplot
    landmark_names = ['OFD-1', 'OFD-2', 'BPD-1', 'BPD-2']
    axes[1].boxplot([all_errors[:, i] for i in range(4)], labels=landmark_names)
    axes[1].set_ylabel('Pixel Error', fontsize=12)
    axes[1].set_title('Error Distribution per Landmark', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Error distribution saved: {save_path}")


# ==========================================
# MAIN TESTING CODE
# ==========================================

if __name__ == '__main__':
    
    print("=" * 60)
    print("🧪 TESTING - HYPOTHESIS 1: BASELINE U-NET")
    print("=" * 60)
    
    # ===== CONFIGURE PATHS (MODIFY THESE) =====
    csv_path = "path/to/role_challenge_dataset_ground_truth.csv"
    image_folder_path = "path/to/images/"
    model_path = "path/to/hypothesis_1_full_saved_model.pth"
    results_dir = "path/to/save/results/"
    
    # ===== LOAD DATA =====
    df = pd.read_csv(csv_path)
    
    # Split to get test set (same split as training)
    _, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    _, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"\nTest set: {len(test_df)} images")
    
    # ===== CREATE DATASET =====
    test_ds = FetalLandmarkDataset(test_df.reset_index(drop=True), image_folder_path)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    
    # ===== LOAD MODEL =====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = UNet(n_class=4).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ Model loaded successfully!")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")

# ===== EVALUATE ON TEST SET =====
print("\n" + "=" * 60)
print("EVALUATING ON TEST SET")
print("=" * 60)

all_errors = []

with torch.no_grad():
    for img_t, hm_true in tqdm(test_loader, desc="Testing"):
        img_t = img_t.to(device)
        hm_pred = model(img_t).squeeze(0).cpu()
        hm_true = hm_true.squeeze(0)
        
        pred_landmarks = extract_landmarks_from_heatmaps(hm_pred)
        true_landmarks = extract_landmarks_from_heatmaps(hm_true)
        errors = calculate_distance_error(pred_landmarks, true_landmarks)
        all_errors.append(errors)

all_errors = np.array(all_errors)

# ===== PRINT RESULTS =====
print("\n" + "=" * 60)
print("TEST SET RESULTS")
print("=" * 60)
print(f"\nPer-Landmark Mean Error (± std):")
print(f"  OFD-1: {all_errors[:, 0].mean():.2f} ± {all_errors[:, 0].std():.2f} px")
print(f"  OFD-2: {all_errors[:, 1].mean():.2f} ± {all_errors[:, 1].std():.2f} px")
print(f"  BPD-1: {all_errors[:, 2].mean():.2f} ± {all_errors[:, 2].std():.2f} px")
print(f"  BPD-2: {all_errors[:, 3].mean():.2f} ± {all_errors[:, 3].std():.2f} px")

print(f"\nOverall Statistics:")
print(f"  Mean Error: {all_errors.mean():.2f} ± {all_errors.std():.2f} px")
print(f"  Median Error: {np.median(all_errors):.2f} px")
print(f"  Max Error: {all_errors.max():.2f} px")
print(f"  Min Error: {all_errors.min():.2f} px")
print("=" * 60)

# ===== VISUALIZE PREDICTIONS =====
print("\n📊 Generating visualizations...")

visualize_predictions(
    model, test_ds, device, num_samples=6,
    save_path=os.path.join(results_dir, 'predictions_visualization.png')
)

plot_error_distribution(
    all_errors,
    save_path=os.path.join(results_dir, 'error_distribution.png')
)

print("\n" + "=" * 60)
print("✅ TESTING COMPLETE!")
print("=" * 60)
print(f"Results saved to: {results_dir}")
print("=" * 60)

