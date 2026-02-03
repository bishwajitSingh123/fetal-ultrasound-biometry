"""
UTILITY FUNCTIONS FOR FETAL HEAD SEGMENTATION
=============================================
Helper functions for data preprocessing, mask filling, and visualization.

Author: Bishwajit Singh
Date: January 2026
"""

import cv2
import numpy as np
import os
from tqdm import tqdm

# ==========================================
# MASK FILLING FUNCTIONS
# ==========================================

def fill_mask_boundaries(mask):
    """Convert boundary mask to filled mask"""
    mask = (mask > 127).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled, [largest_contour], -1, 1, thickness=cv2.FILLED)
        
        # Use convex hull if coverage is low
        if filled.sum() / filled.size < 0.20:
            hull = cv2.convexHull(largest_contour)
            filled = np.zeros_like(mask)
            cv2.drawContours(filled, [hull], -1, 1, thickness=cv2.FILLED)
    
    return filled

def preprocess_all_masks(mask_folder, output_folder):
    """Fill all boundary masks in a folder"""
    os.makedirs(output_folder, exist_ok=True)
    
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png')]
    
    print(f"Processing {len(mask_files)} masks...")
    
    for mask_file in tqdm(mask_files):
        mask = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE)
        filled = fill_mask_boundaries(mask)
        cv2.imwrite(os.path.join(output_folder, mask_file), filled * 255)
    
    print(f"✅ Saved filled masks to: {output_folder}")

# ==========================================
# VALIDATION FUNCTIONS
# ==========================================

def validate_dataset(image_dir, mask_dir, ext=".png"):
    """Validate image-mask pairs"""
    report = {
        "total_images": 0,
        "valid_pairs": 0,
        "missing_masks": [],
        "shape_mismatch": []
    }
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(ext)]
    report["total_images"] = len(image_files)
    
    for img_name in image_files:
        mask_name = img_name.replace(ext, "_Annotation" + ext)
        
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            report["missing_masks"].append(img_name)
            continue
        
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            continue
        
        if img.shape[:2] != mask.shape:
            report["shape_mismatch"].append((img_name, img.shape, mask.shape))
            continue
        
        report["valid_pairs"] += 1
    
    return report

if __name__ == "__main__":
    print("Utility functions loaded successfully!")
