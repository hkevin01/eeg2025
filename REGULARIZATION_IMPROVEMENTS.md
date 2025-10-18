# Enhanced Regularization for EEG Challenge 1

**Date:** October 18, 2025  
**Objective:** Add L1, L2, and Dropout regularization to prevent overfitting

## 🎯 Problem: Overfitting

EEG models tend to overfit because:
- High-dimensional data (129 channels × 200 timepoints)
- Relatively small datasets (even with R1-R4: ~719 subjects)
- Individual variability in EEG signals
- Model can memorize training patterns instead of learning generalizable features

## ✅ Solutions Implemented

### 1. **Dropout Regularization** (Already had, enhanced)

**What it does:** Randomly drops neurons during training (sets to 0)  
**Benefits:** Forces network to learn robust features, prevents co-adaptation

**Implementation:**
```python
class CompactCNN(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        
        self.features = nn.Sequential(
            # Conv1
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_p * 0.6),  # 30% dropout
            
            # Conv2
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_p * 0.8),  # 40% dropout
            
            # Conv3
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_p),  # 50% dropout
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),  # 50% dropout
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_p * 0.8),  # 40% dropout
            nn.Linear(32, 1)
        )
```

**Dropout rates:**
- Conv layers: 30% → 40% → 50% (increasing depth)
- FC layers: 50% → 40% (decreasing towards output)
- **Total:** 5 dropout layers throughout network

### 2. **L2 Regularization (Ridge)** - Weight Decay

**What it does:** Penalizes large weights, encourages smaller weights  
**Benefits:** Prevents weights from growing too large, improves generalization

**Implementation:**
```python
# L2 regularization via AdamW's weight_decay
optimizer = optim.AdamW(
    model.parameters(), 
    lr=1e-3, 
    weight_decay=1e-4  # L2 penalty = 0.0001
)
```

**How it works:**
```
During each update:
weight = weight - lr * (gradient + weight_decay * weight)
                                    ^^^^^^^^^^^^^^^^
                                    L2 penalty term
```

**Effect:**
- Weights are pulled towards zero
- Model prefers simpler solutions
- Reduces overfitting to training data

### 3. **L1 Regularization (Lasso)** - NEW!

**What it does:** Penalizes absolute value of weights, promotes sparsity  
**Benefits:** Some weights go to exactly zero, automatic feature selection

**Implementation:**
```python
def train_model(model, train_loader, val_loader, epochs=50, 
                l1_lambda=1e-5, l2_lambda=1e-4):
    
    for data, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        
        # MSE loss
        mse_loss = criterion(outputs, labels)
        
        # L1 regularization (sum of absolute weights)
        l1_penalty = 0.0
        for param in model.parameters():
            l1_penalty += torch.sum(torch.abs(param))
        
        # Total loss = MSE + L1 penalty
        # (L2 is handled by weight_decay in optimizer)
        loss = mse_loss + l1_lambda * l1_penalty
        
        loss.backward()
        optimizer.step()
```

**L1 vs L2 comparison:**
```
L1 (Lasso):           |w|    →  Sparse weights (some exactly 0)
L2 (Ridge):           w²     →  Small weights (never exactly 0)
Elastic Net (L1+L2):  α|w| + β·w²  →  Best of both!
```

**Why use both?**
- L1: Feature selection (zeroes out unimportant channels/features)
- L2: Stability (prevents remaining weights from being too large)
- **Elastic Net** = L1 + L2 = Better generalization

### 4. **Gradient Clipping** (Already had)

**What it does:** Limits gradient magnitude to prevent exploding gradients  
**Benefits:** Stabilizes training, especially with strong regularization

**Implementation:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 📊 Regularization Hyperparameters

| Technique | Parameter | Value | Effect |
|-----------|-----------|-------|--------|
| **Dropout** | dropout_p | 0.5 | 50% neurons dropped during training |
| **L1 (Lasso)** | l1_lambda | 1e-5 | Weak L1 penalty for sparsity |
| **L2 (Ridge)** | l2_lambda | 1e-4 | Moderate L2 penalty (weight_decay) |
| **Gradient Clip** | max_norm | 1.0 | Prevents exploding gradients |
| **Batch Norm** | - | All conv layers | Normalizes activations |

## 🔍 How to Monitor Regularization

During training, you'll see:
```
Train NRMSE: 0.6543  |  L1 Penalty: 1.23e+03
Val NRMSE:   0.6821
```

**What to look for:**

1. **Train vs Val NRMSE:**
   - Train << Val → Overfitting (increase regularization)
   - Train ≈ Val → Good generalization ✅
   - Train >> Val → Underfitting (decrease regularization)

2. **L1 Penalty:**
   - Should decrease over time as weights get smaller
   - Typical range: 1e+03 to 1e+04
   - Very high (>1e+05) → too much regularization
   - Very low (<1e+02) → not enough sparsity

3. **Model weights:**
   ```python
   # Check weight sparsity after training
   total_params = 0
   zero_params = 0
   for param in model.parameters():
       total_params += param.numel()
       zero_params += (param.abs() < 1e-6).sum().item()
   
   sparsity = 100 * zero_params / total_params
   print(f"Sparsity: {sparsity:.2f}%")  # Should be 5-15% with L1
   ```

## 🎛️ Tuning Regularization Strength

If overfitting (train NRMSE much better than val):
```python
# Option 1: Increase dropout
model = CompactCNN(dropout_p=0.6)  # Instead of 0.5

# Option 2: Increase L1
train_model(..., l1_lambda=5e-5)  # Instead of 1e-5

# Option 3: Increase L2
train_model(..., l2_lambda=5e-4)  # Instead of 1e-4

# Option 4: Combination
model = CompactCNN(dropout_p=0.6)
train_model(..., l1_lambda=5e-5, l2_lambda=5e-4)
```

If underfitting (both train and val NRMSE high):
```python
# Option 1: Decrease dropout
model = CompactCNN(dropout_p=0.4)  # Instead of 0.5

# Option 2: Decrease L1
train_model(..., l1_lambda=1e-6)  # Instead of 1e-5

# Option 3: Decrease L2
train_model(..., l2_lambda=1e-5)  # Instead of 1e-4
```

## 📈 Expected Impact

**Baseline (minimal regularization):**
- Train NRMSE: 0.50
- Val NRMSE: 0.80
- Overfitting gap: 0.30 (bad!)

**With L1 + L2 + Dropout:**
- Train NRMSE: 0.65
- Val NRMSE: 0.70
- Overfitting gap: 0.05 (good!)

**Benefits:**
- ✅ Better generalization to validation/test data
- ✅ More robust predictions
- ✅ Reduced overfitting (train ≈ val)
- ✅ Sparser model (some weights exactly 0)
- ✅ Automatic feature selection

**Trade-offs:**
- ❌ Slightly worse training performance
- ❌ Slightly slower training (L1 computation)
- ❌ Need to tune hyperparameters

## 🧪 Quick Ablation Study

To understand each regularization's contribution:

```bash
# 1. No regularization (baseline)
model = CompactCNN(dropout_p=0.0)
train_model(..., l1_lambda=0.0, l2_lambda=0.0)

# 2. Dropout only
model = CompactCNN(dropout_p=0.5)
train_model(..., l1_lambda=0.0, l2_lambda=0.0)

# 3. Dropout + L2
model = CompactCNN(dropout_p=0.5)
train_model(..., l1_lambda=0.0, l2_lambda=1e-4)

# 4. Dropout + L2 + L1 (FULL - recommended)
model = CompactCNN(dropout_p=0.5)
train_model(..., l1_lambda=1e-5, l2_lambda=1e-4)
```

## 📚 Mathematical Background

### L1 Regularization (Lasso):
```
Loss = MSE + λ₁ ∑|wᵢ|

Where:
- λ₁ = l1_lambda (strength of L1 penalty)
- wᵢ = model weights
- |wᵢ| = absolute value of weights

Effect: Pushes weights towards exactly 0 (sparsity)
```

### L2 Regularization (Ridge):
```
Loss = MSE + λ₂ ∑wᵢ²

Where:
- λ₂ = l2_lambda (strength of L2 penalty)
- wᵢ = model weights
- wᵢ² = squared weights

Effect: Shrinks all weights towards 0 (smoothness)
```

### Elastic Net (L1 + L2):
```
Loss = MSE + λ₁ ∑|wᵢ| + λ₂ ∑wᵢ²

Combines benefits of both:
- L1: Sparsity (feature selection)
- L2: Stability (prevents overfitting)
```

## 🔗 References

- **Dropout:** Srivastava et al. (2014) - "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
- **L1/L2:** Zou & Hastie (2005) - "Regularization and variable selection via the elastic net"
- **AdamW:** Loshchilov & Hutter (2017) - "Decoupled Weight Decay Regularization"

## ✨ Summary

**Implemented regularization stack:**
1. ✅ **Dropout:** 0.3-0.5 throughout network (5 layers)
2. ✅ **L2 (Ridge):** weight_decay=1e-4 in optimizer
3. ✅ **L1 (Lasso):** l1_lambda=1e-5 in loss function
4. ✅ **Gradient clipping:** max_norm=1.0
5. ✅ **Batch normalization:** All conv layers

**Result:** **Elastic Net regularization** for best generalization! 🎯

---

**Next steps:**
1. Run training with new regularization
2. Monitor train vs val NRMSE gap
3. Tune hyperparameters if needed
4. Compare with baseline (no regularization)
