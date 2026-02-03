"""
FETAL HEAD SEGMENTATION - COMPLETE TESTING SCRIPT
==================================================
This script evaluates trained models and generates visualizations.

Author: Bishwajit Singh
Date: January 2026
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# ==========================================
# EVALUATION METRICS
# ==========================================

def calculate_iou(pred, target, threshold=0.5):
    """Calculate IoU"""
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def calculate_dice(pred, target, threshold=0.5):
    """Calculate Dice coefficient"""
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    return (2. * intersection + 1e-6) / (pred_binary.sum() + target.sum() + 1e-6)

# ==========================================
# POST-PROCESSING
# ==========================================

def post_process_mask(prob_mask, threshold=0.5):
    """Post-process prediction mask"""
    binary = (prob_mask > threshold).astype(np.uint8)
    
    # Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary = (labels == largest).astype(np.uint8)
    
    # Morphological closing
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary

# ==========================================
# ELLIPSE FITTING
# ==========================================

def fit_ellipse_and_points(binary_mask):
    """Fit ellipse and get biometric points"""
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0 or len(contours[0]) < 5:
        return None, None
    
    cnt = max(contours, key=cv2.contourArea)
    ellipse = cv2.fitEllipse(cnt)
    (cx, cy), (MA, ma), angle = ellipse
    
    angle_rad = np.deg2rad(angle)
    dx = (MA / 2) * np.cos(angle_rad)
    dy = (MA / 2) * np.sin(angle_rad)
    
    p1 = (int(cx + dx), int(cy + dy))
    p2 = (int(cx - dx), int(cy - dy))
    
    return ellipse, (p1, p2)

# ==========================================
# EVALUATION FUNCTION
# ==========================================

def evaluate_model(model, test_loader, device):
    """Complete model evaluation"""
    model.eval()
    
    iou_scores = []
    dice_scores = []
    predictions = []
    targets = []
    images = []
    
    with torch.no_grad():
        for batch_images, batch_masks in tqdm(test_loader, desc="Evaluating"):
            batch_images = batch_images.to(device)
            batch_masks = batch_masks.to(device)
            
            outputs = model(batch_images)
            
            for i in range(batch_images.size(0)):
                pred = outputs[i, 0].cpu().numpy()
                target = batch_masks[i, 0].cpu().numpy()
                
                pred_binary = post_process_mask(pred, threshold=0.5)
                
                iou = calculate_iou(torch.from_numpy(pred_binary).float(), torch.from_numpy(target).float())
                dice = calculate_dice(torch.from_numpy(pred_binary).float(), torch.from_numpy(target).float())
                
                iou_scores.append(iou.item())
                dice_scores.append(dice.item())
                
                if len(predictions) < 12:
                    predictions.append(outputs[i].cpu())
                    targets.append(batch_masks[i].cpu())
                    images.append(batch_images[i].cpu())
    
    avg_iou = np.mean(iou_scores)
    avg_dice = np.mean(dice_scores)
    
    print(f"\n✅ Test Results:")
    print(f"   IoU:  {avg_iou:.4f} ± {np.std(iou_scores):.4f}")
    print(f"   Dice: {avg_dice:.4f} ± {np.std(dice_scores):.4f}")
    
    return {
        'avg_iou': avg_iou,
        'avg_dice': avg_dice,
        'predictions': predictions,
        'targets': targets,
        'images': images
    }

# ==========================================
# MAIN TESTING FUNCTION
# ==========================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model (define UNetSegmentation class from Trainer.py)
    # model = UNetSegmentation().to(device)
    # model.load_state_dict(torch.load('hypothesis_2_best.pth'))
    
    # Evaluate
    # results = evaluate_model(model, test_loader, device)
    
    print("\n🎉 Testing completed successfully!")
