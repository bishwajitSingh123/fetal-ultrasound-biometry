# FETAL HEAD SEGMENTATION - PYTHON SCRIPTS

## File Structure:
```
Python Script/
├── Trainer.py          ← Complete training pipeline
├── Tester.py           ← Testing & evaluation
└── Assets/
    └── utils.py        ← Helper functions
```

## Usage:

### Training:
```python
python Trainer.py
```

### Testing:
```python
python Tester.py
```

## Key Features:
- ✅ Complete training pipeline for all hypotheses
- ✅ Data preprocessing & augmentation
- ✅ Multiple loss functions (Focal, Dice, Combined)
- ✅ Model evaluation with metrics (IoU, Dice)
- ✅ Post-processing & ellipse fitting
- ✅ Visualization utilities

## Requirements:
- Python 3.8+
- PyTorch 1.10+
- OpenCV
- NumPy
- scikit-learn
- tqdm
- matplotlib

## Best Model:
**Hypothesis 2** (hypothesis_2_best.pth)
- Test Dice: 0.9575
- Test IoU: 0.9229
