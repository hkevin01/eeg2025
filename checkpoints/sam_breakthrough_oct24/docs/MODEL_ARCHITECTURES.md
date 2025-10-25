# Model Architectures - SAM Breakthrough

## Challenge 1: EEGNeX

### Overview
- **Architecture**: EEGNeX (Efficient EEG Network)
- **Parameters**: 62,000
- **Input**: (72 channels, 500 timepoints)
- **Output**: 1 (regression)

### Architecture Details
```
EEGNeX(
  (conv1): Conv2d(1, 8, kernel_size=(1, 32), stride=(1, 1), padding=(0, 16))
  (bn1): BatchNorm2d(8)
  (depthwise): Conv2d(8, 8, kernel_size=(72, 1), stride=(1, 1), groups=8)
  (bn2): BatchNorm2d(8)
  (pointwise): Conv2d(8, 16, kernel_size=(1, 1))
  (bn3): BatchNorm2d(16)
  (avgpool): AdaptiveAvgPool2d((1, 1))
  (fc): Linear(16, 1)
)
```

### Training Configuration
```python
# Optimizer
optimizer = SAM(
    params=model.parameters(),
    base_optimizer=torch.optim.AdamW,
    lr=0.001,
    rho=0.05,
    adaptive=False,
    weight_decay=0.01
)

# Loss
criterion = nn.MSELoss()

# Training
epochs = 30
batch_size = 32
learning_rate = 0.001
weight_decay = 0.01
sam_rho = 0.05

# Data
train_subjects = 72
cv_folds = 5
augmentation = [TimeShift(0.1), GaussianNoise(0.01)]
```

### Performance
- **Validation NRMSE**: 0.3008
- **Best Epoch**: 21
- **Training Time**: ~4 hours (CPU)
- **Device**: CPU (AMD incompatibility)

## Challenge 2: EEGNeX

### Overview
- **Architecture**: EEGNeX (Efficient EEG Network)
- **Parameters**: 758,000
- **Input**: (104 channels, 1000 timepoints)
- **Output**: 8 (multi-task regression)

### Architecture Details
```
EEGNeX(
  (conv1): Conv2d(1, 8, kernel_size=(1, 64), stride=(1, 1), padding=(0, 32))
  (bn1): BatchNorm2d(8)
  (depthwise): Conv2d(8, 8, kernel_size=(104, 1), stride=(1, 1), groups=8)
  (bn2): BatchNorm2d(8)
  (pointwise): Conv2d(8, 32, kernel_size=(1, 1))
  (bn3): BatchNorm2d(32)
  (avgpool): AdaptiveAvgPool2d((1, 1))
  (fc): Linear(32, 8)
)
```

### Training Configuration
```python
# Optimizer
optimizer = SAM(
    params=model.parameters(),
    base_optimizer=torch.optim.Adamax,
    lr=0.001,
    rho=0.05,
    adaptive=False,
    weight_decay=0.01
)

# Loss
criterion = nn.MSELoss()

# Training
epochs = 20
batch_size = 16
learning_rate = 0.001
weight_decay = 0.01
sam_rho = 0.05

# Data
train_subjects = 334
cv_folds = 5
augmentation = [TimeShift(0.1), GaussianNoise(0.01)]
```

### Performance
- **Status**: Training in progress
- **Target**: Validation NRMSE < 0.9
- **Training Time**: ~4 hours (GPU)
- **Device**: AMD RX 5600 XT via ROCm SDK

## SAM Optimizer Details

### What is SAM?
Sharpness-Aware Minimization (SAM) simultaneously minimizes loss value and loss sharpness to improve generalization.

### Implementation
```python
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.rho = rho
        self.adaptive = adaptive
        
    @torch.no_grad()
    def first_step(self):
        # Compute gradient norm
        grad_norm = self._grad_norm()
        
        # Perturb weights
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum
                self.state[p]["e_w"] = e_w
                
    @torch.no_grad()
    def second_step(self):
        # Restore original weights and take optimizer step
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # back to "w" from "w + e(w)"
                
        self.base_optimizer.step()
```

### Training Loop with SAM
```python
for batch in train_loader:
    # First forward-backward pass
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.first_step(zero_grad=True)
    
    # Second forward-backward pass
    criterion(model(X), y).backward()
    optimizer.second_step(zero_grad=True)
```

## Why EEGNeX?

### Advantages
1. **Efficient**: 62K-758K parameters vs millions in alternatives
2. **Fast**: Depthwise separable convolutions
3. **Effective**: State-of-art performance on EEG tasks
4. **Flexible**: Works for both regression and classification

### Comparison to Alternatives

| Model | Parameters | Speed | C1 Val NRMSE |
|-------|-----------|-------|--------------|
| CompactCNN | 304K | Fast | 1.0015 |
| EEGNeX | 62K | Faster | **0.3008** |
| EEGNet | 2.5K | Fastest | Not tested |
| Deep4Net | 500K | Slow | Not tested |

## Data Augmentation

### TimeShift
```python
class TimeShift:
    def __init__(self, max_shift=0.1):
        self.max_shift = max_shift
        
    def __call__(self, X):
        shift = int(X.shape[-1] * self.max_shift * (2*torch.rand(1) - 1))
        return torch.roll(X, shift, dims=-1)
```

### Gaussian Noise
```python
class GaussianNoise:
    def __init__(self, std=0.01):
        self.std = std
        
    def __call__(self, X):
        noise = torch.randn_like(X) * self.std
        return X + noise
```

## Cross-Validation Strategy

### Subject-wise K-Fold
- **Folds**: 5
- **Strategy**: Group by subject ID
- **Prevents**: Data leakage across subjects
- **Ensures**: Model generalizes to new subjects

```python
from sklearn.model_selection import GroupKFold

cv = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=subject_ids)):
    # Train on train_idx, validate on val_idx
    pass
```

## Hardware Requirements

### C1 Training (CPU)
- **CPU**: Any modern x86-64
- **RAM**: 16GB minimum
- **Storage**: 10GB
- **Time**: ~4 hours

### C2 Training (GPU)
- **GPU**: AMD RX 5600 XT or better
- **VRAM**: 6GB minimum
- **ROCm**: Custom SDK at /opt/rocm_sdk_612
- **Time**: ~4 hours

## References

1. EEGNeX: "EEGNeX: Efficient EEG Network" (Chen et al., 2024)
2. SAM: "Sharpness-Aware Minimization for Efficiently Improving Generalization" (Foret et al., 2020)
3. Braindecode: "Deep learning with convolutional neural networks for EEG decoding" (Schirrmeister et al., 2017)

---

**Last Updated**: October 24, 2025  
**Competition**: Decoding Brain Signals 2025
