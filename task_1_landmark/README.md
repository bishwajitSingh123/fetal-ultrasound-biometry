================================================================
HYPOTHESIS 1: BASELINE U-NET FOR LANDMARK DETECTION
================================================================

Author: Bishwajit Singh
Date: January 2026
Task: Fetal BPD & OFD Landmark Detection (Part A)

================================================================
APPROACH
================================================================
- Model: Basic U-Net architecture
- Input: 256x256 RGB ultrasound images
- Output: 4 heatmaps (one per landmark point)
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam (lr=1e-3, ReduceLROnPlateau scheduler)
- Training: 50 epochs, batch size 8
- Data Split: 80% train (497), 10% val (62), 10% test (63)
- No data augmentation
- Gaussian heatmap sigma: 10

================================================================
RESULTS
================================================================
Training:
- Training Time: 23.3 minutes
- Best Validation Loss: 0.0027
- Final Train Loss: 0.0001

Test Set Performance (63 images):
- Overall Mean Error: 25.93 ± 32.45 pixels
- Median Error: 6.08 pixels
- Min Error: 0.44 pixels
- Max Error: 192.87 pixels

Per-Landmark Errors:
- OFD-1: Mean ~18px, Median ~10px
- OFD-2: Mean ~24px, Median ~15px  
- BPD-1: Mean ~30px, Median ~15px
- BPD-2: Mean ~12px, Median ~5px (BEST!)

================================================================
OBSERVATIONS
================================================================
Strengths:
+ Median error (6.08px) indicates good performance on majority of images
+ BPD-2 landmark consistently accurate (~5px median)
+ Fast convergence (50 epochs, 23 mins)
+ Model learned general pattern well

Weaknesses:
- Presence of outliers (some errors > 100px)
- OFD points less accurate than BPD points
- High variance in predictions (std: 32.45px)
- Possible overfitting (train: 0.0001 vs val: 0.0027)
- No data augmentation used

Root Causes:
- Image quality variations not handled
- No augmentation → poor generalization
- MSE loss → not robust to outliers
- Simple architecture → limited feature learning

================================================================
ARCHITECTURE DETAILS
================================================================
U-Net Encoder:
  Input (3, 256, 256)
  → Conv Block 1: 64 channels (256x256)
  → MaxPool → Conv Block 2: 128 channels (128x128)
  → MaxPool → Conv Block 3: 256 channels (64x64)
  → MaxPool → Bottleneck: 512 channels (32x32)

U-Net Decoder:
  Bottleneck (512, 32, 32)
  → Upsample + Skip → Conv Block: 256 channels (64x64)
  → Upsample + Skip → Conv Block: 128 channels (128x128)
  → Upsample + Skip → Conv Block: 64 channels (256x256)
  → Output Conv: 4 channels (heatmaps)

Total Parameters: 7,788,740
Each Conv Block: 2x (Conv2d + BatchNorm + ReLU)

================================================================
FILES INCLUDED
================================================================
1. Model Weights/
   - hypothesis_1_full_saved_model.pth (complete checkpoint)

2. Python Script/
   - Trainer.py (training script)
   - Tester.py (inference & evaluation)

3. Assets/
   - hypothesis_1_training_curves.png (loss curves)
   - predictions_visualization.png (6 sample predictions)
   - error_distribution.png (histogram + boxplot)

4. ReadMe.txt (this file)

================================================================
HOW TO RUN
================================================================

TRAINING:
---------
python Trainer.py

Required arguments:
- csv_path: Path to ground truth CSV
- image_folder_path: Path to images directory

Output:
- Trained model: hypothesis_1_full_saved_model.pth
- Training curves: hypothesis_1_training_curves.png

TESTING:
--------
python Tester.py

Required arguments:
- csv_path: Path to ground truth CSV
- image_folder_path: Path to images directory
- model_path: Path to saved model

Output:
- Test metrics printed to console
- predictions_visualization.png (sample predictions)
- error_distribution.png (error analysis)

Requirements:
- PyTorch >= 1.10
- torchvision
- opencv-python (cv2)
- pandas
- numpy
- matplotlib
- scikit-learn
- tqdm

================================================================
HYPOTHESIS 2 & 3 - PLANNED IMPROVEMENTS
================================================================

Based on Hypothesis 1 results, next experiments will focus on:

Hypothesis 2 - Data Augmentation:
1. Add random rotations (±15 degrees)
2. Random horizontal flips
3. Brightness/contrast adjustments
4. Elastic deformations
→ Expected: Reduce outliers, improve generalization

Hypothesis 3 - Advanced Architecture:
1. ResNet-34 backbone instead of vanilla CNN
2. Adaptive Wing Loss instead of MSE (robust to outliers)
3. Attention mechanisms for better feature focus
4. Deeper decoder with more skip connections
→ Expected: Better mean error, more robust predictions

================================================================
CONCLUSION
================================================================

Hypothesis 1 establishes a strong baseline with median error of 6.08px.
However, outliers and high variance indicate need for:
- Data augmentation (reduce overfitting)
- Better loss function (handle outliers)
- Deeper architecture (capture complex patterns)

These will be addressed in Hypotheses 2 and 3.

================================================================
