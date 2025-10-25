# Challenge 1 Improvement Plan
**Current Performance:** Val NRMSE 0.3008 (70% better than baseline)  
**Target:** Val NRMSE < 0.25 (75-80% better)  
**Date:** October 25, 2025

---

## 🎯 Top 3 High-Impact Improvements (Phase 1)

### 1. ✨ Temporal Attention Mechanism [15-20% improvement]

**Why It Works:**
- Response time prediction requires focusing on specific temporal patterns
- Stimulus-locked ERPs occur at different latencies (P1, N1, P300)
- Attention helps model learn which timepoints matter most

**Implementation:**
```python
class TemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, time, channels)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        return x.transpose(1, 2)
```

**Expected Impact:** 0.3008 → 0.25-0.27 (10-15% improvement)

---

### 2. 🔄 Advanced Augmentation (Mixup + Temporal Masking) [5-10% improvement]

**Current Augmentation:**
- Simple scaling (±20%)
- Channel dropout (5%)
- Gaussian noise (5%)

**New Augmentation:**

**A. Mixup** - Mix two samples
```python
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# Loss: lam * loss(pred, y_a) + (1-lam) * loss(pred, y_b)
```

**B. Temporal Masking** - Mask random time segments
```python
def temporal_masking(x, mask_ratio=0.15):
    mask_length = int(time_points * mask_ratio)
    start = random.randint(0, time_points - mask_length)
    x[:, :, start:start+mask_length] = 0
    return x
```

**C. Magnitude Warping** - Smooth random scaling
```python
def magnitude_warping(x, sigma=0.2):
    # Generate smooth curve, interpolate to signal length
    knots = np.random.randn(5) * sigma + 1.0
    warp_curve = interpolate(knots, time_points)
    return x * warp_curve
```

**Expected Impact:** 5-10% improvement in generalization

---

### 3. 🎯 Multi-Scale Temporal Features [8-12% improvement]

**Why It Works:**
- RT components occur at different timescales:
  - **Fast (10-50ms):** Early sensory processing
  - **Medium (100-300ms):** Decision making (P300)
  - **Slow (500-1000ms):** Motor preparation

**Implementation:**
```python
class MultiScaleFeaturesExtractor(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super().__init__()
        # Three parallel branches
        self.fast_conv = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.medium_conv = nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=7)
        self.slow_conv = nn.Conv1d(in_channels, out_channels, kernel_size=31, padding=15)
        
    def forward(self, x):
        fast = F.relu(self.fast_conv(x))
        medium = F.relu(self.medium_conv(x))
        slow = F.relu(self.slow_conv(x))
        return torch.cat([fast, medium, slow], dim=1)  # Concatenate features
```

**Expected Impact:** 8-12% improvement

---

## 🏗️ Enhanced Architecture

### Combined Model: EnhancedEEGNeX

```
Input (129 channels, 200 timepoints)
    ↓
Multi-Scale Feature Extraction
├─ Fast Branch (kernel=5)   → 32 features
├─ Medium Branch (kernel=15) → 32 features
└─ Slow Branch (kernel=31)  → 32 features
    ↓ (concatenate)
96-dim features
    ↓
Temporal Attention (4 heads)
    ↓ (learn temporal importance)
Attended Features
    ↓
Adaptive Average Pooling
    ↓
FC Layer (96 → 64) + ReLU + Dropout(0.3)
    ↓
FC Layer (64 → 1)
    ↓
Response Time Prediction
```

**Parameters:** ~150K (vs 62K baseline)  
**Training Time:** ~1.5x slower  
**Expected Performance:** 0.22-0.25 NRMSE

---

## 📊 Training Strategy Updates

### Training Loop with Mixup
```python
for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        # Apply mixup
        if use_mixup and random.random() < 0.5:
            mixed_X, y_a, y_b, lam = mixup_data(batch_X, batch_y, alpha=0.2)
            
            # SAM first step
            predictions = model(mixed_X)
            loss = lam * criterion(predictions, y_a) + (1-lam) * criterion(predictions, y_b)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            # SAM second step
            predictions = model(mixed_X)
            loss = lam * criterion(predictions, y_a) + (1-lam) * criterion(predictions, y_b)
            loss.backward()
            optimizer.second_step(zero_grad=True)
        else:
            # Regular SAM training
            # ... (existing code)
```

### Augmentation Schedule
- **Epochs 1-10:** Light augmentation (mixup 30%, masking 20%)
- **Epochs 11-20:** Heavy augmentation (mixup 50%, masking 50%)
- **Epochs 21+:** Medium augmentation (mixup 40%, masking 30%)

---

## 🚀 Implementation Plan

### Option A: Quick Implementation (Recommended)
**Time:** 1-2 hours  
**Approach:** Add modules to existing train_c1_sam_simple.py

1. Add TemporalAttention class
2. Add MultiScaleFeaturesExtractor class
3. Create EnhancedEEGNeX combining both
4. Add mixup_data() and temporal_masking() functions
5. Modify training loop to use mixup
6. Train for 30 epochs with early stopping

**Expected Result:** 0.22-0.26 NRMSE

### Option B: Clean Rewrite
**Time:** 2-3 hours  
**Approach:** Create train_c1_enhanced.py from scratch

1. Copy SAM optimizer and data loader from working script
2. Implement all new components
3. Add comprehensive logging
4. Add visualization of attention weights
5. Train with full monitoring

**Expected Result:** 0.20-0.25 NRMSE (with tuning)

---

## 📈 Expected Performance Progression

| Version | NRMSE | vs Baseline | Improvement |
|---------|-------|-------------|-------------|
| Baseline (Oct 16) | 1.0015 | - | - |
| Current SAM (Oct 24) | 0.3008 | ↓ 70% | 🎉 |
| + Temporal Attention | 0.25-0.27 | ↓ 73-75% | +10-15% |
| + Mixup/Masking | 0.23-0.25 | ↓ 75-77% | +5-8% |
| + Multi-Scale | 0.20-0.23 | ↓ 77-80% | +8-12% |
| **Combined (Phase 1)** | **0.20-0.23** | **↓ 77-80%** | **+25-30%** |

---

## 🎯 Next Steps (Phase 2 - Optional)

If Phase 1 achieves < 0.25, consider:

1. **Ensemble (3-5 models)** → Additional 3-7% improvement
2. **Channel Attention** → Additional 3-5% improvement
3. **Frequency Domain Features** → Additional 5-8% improvement

**Potential Final Score:** 0.18-0.22 NRMSE (80-82% better than baseline)

---

## ⚡ Quick Start Command

```bash
# Train enhanced model
python train_c1_enhanced.py \
    --epochs 30 \
    --batch_size 16 \
    --lr 0.001 \
    --rho 0.05 \
    --mixup_alpha 0.2 \
    --mask_ratio 0.15 \
    --exp_name enhanced_v1

# Expected training time: ~4-6 hours on CPU, ~1-2 hours on GPU
```

---

## 📝 Success Criteria

- [  ] Val NRMSE < 0.25 (minimum target)
- [  ] Val NRMSE < 0.23 (good performance)
- [  ] Val NRMSE < 0.20 (excellent performance)
- [  ] No overfitting (train vs val gap < 20%)
- [  ] Stable across CV folds (std < 0.03)

---

**Recommendation:** Start with Phase 1 implementation. If it achieves < 0.25, create submission. If time permits, implement Phase 2 for even better results.

**Estimated Overall Improvement:** 25-35% better than current SAM model  
**Projected Test Score:** 0.20-0.30 (vs current projected 0.30-0.50)
