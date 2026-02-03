# FETAL HEAD SEGMENTATION - FINAL SUBMISSION
> End-to-end deep learning pipeline for accurate fetal head segmentation from 2D ultrasound images

---

## 📋 PROJECT OVERVIEW

**Task:** Automated fetal head segmentation from 2D ultrasound images  
**Dataset:** 622 images (Train: 497, Validation: 62, Test: 63)  
**Annotation Type:** Binary segmentation masks  
**Best Performing Model:** Hypothesis 2 (Dice: 0.9575, IoU: 0.9229)

---

## 📁 FOLDER STRUCTURE
```
task_2_segmentation/
├── Model Weights/          ← Trained model checkpoints
├── Python Script/          ← Training & testing code
├── Assets/                 ← Visualizations & plots
├── Report/                 ← PDF report + detailed logs
└── README.md              ← This file
```


---

## 🏆 HYPOTHESES EVALUATED

### **Hypothesis 1: Baseline U-Net**
- **Issue Identified:** Boundary-only masks (~0.7% foreground coverage)
- **Performance:** Dice = 0.037
- **Outcome:** Failed due to incorrect ground truth representation
- **Key Learning:** Data quality is fundamental to model performance

---

### **Hypothesis 2: Improved U-Net** ⭐ **BEST MODEL**
- **Fix Applied:** Properly filled masks (25–28% coverage)
- **Techniques:** Focal + Dice loss, strong data augmentation
- **Performance:** Dice = 0.9575 (**+2488% improvement over H1**)
- **Conclusion:** Correct annotations unlocked model capability

---

### **Hypothesis 3: ResNet34 Encoder + CBAM Attention**
- **Approach:** Transfer learning with attention mechanism
- **Performance:** Dice = 0.940
- **Observation:** Marginal drop compared to H2
- **Conclusion:** H2 already near optimal for this dataset

---

## 🎯 CORE INSIGHT

**DATA QUALITY > ARCHITECTURE COMPLEXITY**

- Fixing annotations (H1 → H2): **+2488% Dice improvement**
- Increasing model complexity (H2 → H3): **–1.8% Dice change**

---

## 📊 FINAL RESULTS SUMMARY

| Model | Dice Score | IoU | Training Time |
|------|------------|-----|---------------|
| H1 | 0.037 | 0.019 | 10.1 min |
| **H2 (Best)** | **0.9575** | **0.9229** | 24.0 min |
| H3 | 0.940 | 0.890 | 9.1 min |

---

## 🚀 USAGE

### **Training**
```bash
python Python\ Script/Trainer.py

```

### **Testing:**
```python
python Python\ Script/Tester.py
```

### **Inference:**
```python
import torch
from Trainer import UNetSegmentation

model = UNetSegmentation()
model.load_state_dict(torch.load('Model Weights/hypothesis_Final_best.pth'))
model.eval()

```python
import torch
import cv2
import numpy as np
from Trainer import UNetSegmentation

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetSegmentation().to(device)
model.load_state_dict(
    torch.load("Model Weights/hypothesis_Final_best.pth", map_location=device)
)
model.eval()

# Read and preprocess image
img = cv2.imread("sample_ultrasound.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))
img = img / 255.0

img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    pred_mask = model(img_tensor)
    pred_mask = (pred_mask > 0.5).float()

# Convert to numpy
pred_mask = pred_mask.squeeze().cpu().numpy()

---

**Output:** Binary segmentation mask (256×256)  
**Post-processing:** Optional morphological smoothing / ellipse fitting for HC estimation


📦 INCLUDED FILES
Model Weights/

hypothesis_1_best.pth – Baseline U-Net

hypothesis_2_best.pth – Improved U-Net

hypothesis_3_best.pth – ResNet34 + CBAM

hypothesis_Final_best.pth ⭐ – Recommended for deployment

Additional checkpoints with training history

Python Script/

Trainer.py – Complete training pipeline

Tester.py – Evaluation and metric computation

Assets/utils.py – Helper and utility functions

README.md – Script-level documentation

Assets/

Training and validation curves

Prediction visualizations

Model comparison charts

Metric plots

Report/

Report.pdf ⭐ – Final submission report

FINAL_SUMMARY_REPORT.txt – Consolidated findings

hypothesis_2_detailed_report.txt – Best model analysis

🔑 KEY LEARNINGS

Validate annotations before training

Strong augmentation is critical for small medical datasets

Focal + Dice loss handles class imbalance effectively

Transfer learning offers diminishing returns when data quality is high

🎯 BIOMETRIC POINT EXTRACTION

Method: Ellipse fitting on predicted segmentation mask
Output: Two endpoints of the major axis
Clinical Relevance: Sufficient for estimating Head Circumference (HC) using ellipse approximation
```python
def fit_ellipse_and_points(binary_mask):
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cnt = max(contours, key=cv2.contourArea)
    ellipse = cv2.fitEllipse(cnt)

    (cx, cy), (MA, ma), angle = ellipse
    angle_rad = np.deg2rad(angle)

    dx = (MA / 2) * np.cos(angle_rad)
    dy = (MA / 2) * np.sin(angle_rad)

    p1 = (int(cx + dx), int(cy + dy))
    p2 = (int(cx - dx), int(cy - dy))

    return ellipse, (p1, p2)

```

---

## 📞 CONTACT

**Author:** Bishwajit Prasad Singh  
**Date:** January 2026  
**Task:** Fetal Head Segmentation Challenge

---

## 🙏 ACKNOWLEDGMENTS

- Dataset: Origin Medical AI Challenge
- Framework: PyTorch
- Inspiration: Clinical biometry automation

**Jai Shri Ram! 🙏**
