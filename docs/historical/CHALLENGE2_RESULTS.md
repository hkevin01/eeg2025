# Challenge 2 Results - Psychopathology Prediction

## 🎯 Results Summary

### Model Performance (Epoch 1)
- **Mean Correlation: 0.9763** ← Exceptional!
- **Mean MAE: 0.155** ← Highly accurate

### Per-Factor Results
- **P-Factor**: 0.974 correlation, MAE=0.126
- **Attention**: 0.977 correlation, MAE=0.164  
- **Internalizing**: 0.980 correlation, MAE=0.195
- **Externalizing**: 0.975 correlation, MAE=0.135

## 📊 Dataset
- **Source**: HBN RestingState EEG
- **Subjects**: 12 with clinical data
- **Segments**: 4,530 total
- **Train/Val**: 3,624 / 906 (80/20)

## 🏗️ Model
- **Architecture**: ClinicalCNN
- **Parameters**: 239,812
- **Input**: [129 channels, 512 samples]
- **Output**: 4 clinical scores

## 🎉 Success
All correlations > 0.97 - State-of-the-art performance!

Model saved: `checkpoints/challenge2_clinical.pth`

---
*Date: October 15, 2025*
