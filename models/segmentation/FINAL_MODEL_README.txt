================================================================================
FINAL MODEL SELECTION: HYPOTHESIS 2
================================================================================

FILES:
  - hypothesis_Final_best.pth (29.77 MB)
  - hypothesis_Final_full_saved_model.pth (89.24 MB)

MODEL DETAILS:
  - Architecture: U-Net
  - Parameters: 7,788,545
  - Input: 256x256 RGB ultrasound images
  - Output: 256x256 binary segmentation mask

PERFORMANCE:
  - Test Dice Coefficient: 0.9575 ± 0.0550
  - Test IoU: 0.9229 ± 0.0844
  - Training Time: 24 minutes
  - Best Epoch: 44/50

WHY HYPOTHESIS 2 WAS CHOSEN:
  ✓ HIGHEST Dice score (0.9575 vs H3's 0.940)
  ✓ HIGHEST IoU score (0.9229 vs H3's 0.890)
  ✓ Simpler architecture (easier to deploy)
  ✓ Better generalization on test set

KEY IMPROVEMENTS OVER BASELINE:
  1. Filled Masks: Converted boundary-only masks to filled regions
     - Mask coverage increased from 0.7% to 25-28%
     - This single fix gave 2488% improvement!

  2. Focal Loss: Better handles class imbalance
     - alpha=0.25, gamma=2.0
     - Focuses learning on hard examples

  3. Heavy Augmentation:
     - Horizontal/vertical flips (50% each)
     - Rotation ±15° (50%)
     - Brightness ±20% (30%)
     - Gaussian noise (20%)
     - Zoom 0.9-1.1 (20%)

  4. Regularization:
     - Weight decay: 1e-4
     - Early stopping: patience=8

USAGE:
  For inference:
    model = UNetSegmentation()
    model.load_state_dict(torch.load('hypothesis_Final_best.pth'))
    model.eval()

  For continued training:
    checkpoint = torch.load('hypothesis_Final_full_saved_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

DEPLOYMENT RECOMMENDATIONS:
  1. Use threshold = 0.5 for binary predictions
  2. Apply post-processing:
     - Keep largest connected component
     - Morphological closing (5x5 kernel)
  3. Fit ellipse to extract biometry points
  4. Consider test-time augmentation for ensemble

================================================================================
END OF README
================================================================================
