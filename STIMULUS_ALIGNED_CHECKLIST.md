# Stimulus-Aligned Training - Implementation Checklist

**Status:** ✅ READY TO TRAIN  
**Date:** October 18, 2025  
**Expected Improvement:** 25-30% better NRMSE (1.00 → 0.70-0.80)

## ✅ Completed (This Session)

```markdown
- [x] Identified critical issue: trial-aligned windows instead of stimulus-aligned
- [x] Changed window anchor from "contrast_trial_start" to "stimulus_anchor"
- [x] Updated metadata descriptor to match anchor
- [x] Added R3 and R4 to training data (33% more data)
- [x] Changed validation from R3 to R5 (better split)
- [x] Created STIMULUS_ALIGNED_TRAINING.md (complete explanation)
- [x] Created TRAINING_IMPROVEMENTS_TODO.md (roadmap)
- [x] Created STIMULUS_ALIGNMENT_SESSION_COMPLETE.md (summary)
- [x] Committed all changes to git (3 commits)
```

## 🔥 Immediate Next Steps

### Step 1: Test Stimulus-Aligned Training (PRIORITY: URGENT)

```bash
# Run the updated training script
python scripts/training/challenge1/train_challenge1_multi_release.py
```

**What to look for:**
```markdown
- [ ] Log shows "Creating STIMULUS-ALIGNED windows from trials..."
- [ ] Log shows "Using anchor: stimulus_anchor"
- [ ] Windows created successfully (count > 0)
- [ ] No errors in metadata extraction
- [ ] RT values are valid (not all zeros or NaN)
- [ ] Training NRMSE better than 1.00 baseline
- [ ] Validation NRMSE better than previous runs
```

**Expected runtime:** 2-3 hours (CPU) or 15-30 min (ROCm GPU)

### Step 2: Add Data Augmentation (30 minutes)

```python
# Add to train_challenge1_multi_release.py around line 290

def augment_eeg(X, training=True):
    """EEG augmentation for better generalization"""
    if not training:
        return X
    
    # Time shift (±50ms = ±5 samples at 100Hz)
    if np.random.rand() > 0.5:
        shift = np.random.randint(-5, 6)
        X = np.roll(X, shift, axis=1)
    
    # Gaussian noise (0.5% of signal)
    if np.random.rand() > 0.5:
        noise = np.random.randn(*X.shape) * 0.005
        X = X + noise
    
    # Channel dropout (10% of channels)
    if np.random.rand() > 0.5:
        n_drop = int(0.1 * X.shape[0])
        drop_ch = np.random.choice(X.shape[0], n_drop, replace=False)
        X[drop_ch, :] = 0
    
    return X

# In __getitem__ method, after normalization:
X = augment_eeg(X, training=True)
```

**Checklist:**
```markdown
- [ ] Added augment_eeg function
- [ ] Called augmentation after normalization
- [ ] Tested training runs without errors
- [ ] Measured NRMSE improvement vs baseline
```

**Expected improvement:** 5-10% better NRMSE

### Step 3: Try Better Loss Function (15 minutes)

**Option A: Huber Loss (Robust to outliers)**
```python
# In train() function, around line 350
# Replace:
criterion = nn.MSELoss()

# With:
criterion = nn.SmoothL1Loss()  # Huber loss in PyTorch
```

**Option B: Custom NRMSE Loss**
```python
class NRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2)
        rmse = torch.sqrt(mse)
        nrmse = rmse / (target.std() + 1e-8)
        return nrmse

# Use it:
criterion = NRMSELoss()
```

**Checklist:**
```markdown
- [ ] Changed loss function
- [ ] Trained for at least 10 epochs
- [ ] Compared NRMSE vs MSELoss baseline
- [ ] Selected best performing loss
```

**Expected improvement:** 3-7% better NRMSE

## 🎯 Week 1 Goals (Quick Wins)

```markdown
- [ ] Test stimulus-aligned training → 0.75-0.85 NRMSE
- [ ] Add R4 training data → 0.70-0.80 NRMSE
- [ ] Add data augmentation → 0.65-0.75 NRMSE
- [ ] Try better loss function → 0.60-0.70 NRMSE

Target: 0.60-0.70 NRMSE (30-40% improvement from 1.00)
```

## 🚀 Week 2 Goals (Medium Priority)

```markdown
- [ ] Extract EEGNeX script from EEGNEX_ROCM_STRATEGY.md
- [ ] Add stimulus-aligned anchoring to EEGNeX
- [ ] Train EEGNeX model (4-6 hours)
- [ ] Create ensemble (CompactCNN + EEGNeX)
- [ ] Implement subject-specific normalization

Target: 0.48-0.55 NRMSE (Top 15-20 leaderboard range)
```

## 💎 Week 3+ Goals (Long-term)

```markdown
- [ ] Self-supervised pre-training on passive tasks
- [ ] Temporal attention mechanism
- [ ] Advanced ensemble optimization
- [ ] Hyperparameter tuning

Target: 0.35-0.45 NRMSE (Top 5-10 leaderboard range) 🏆
```

## 📊 Performance Tracking

| Date | Improvement | NRMSE | Notes |
|------|-------------|-------|-------|
| Baseline | - | 1.00 | Trial-aligned, R1-R2 only |
| Oct 18 | Stimulus alignment | ? | Test needed |
| Oct 18 | + R4 data | ? | Test needed |
| TBD | + Data augmentation | ? | Implement next |
| TBD | + Better loss | ? | Implement next |

**Update this table after each training run!**

## 🔍 Verification Steps

After training completes:
```markdown
- [ ] Check training curves (should be smooth)
- [ ] Verify no overfitting (train NRMSE ≈ val NRMSE)
- [ ] Compare with baseline (1.00 NRMSE)
- [ ] Save model weights with descriptive name
- [ ] Update performance tracking table
- [ ] Document any issues or observations
```

## 📚 Reference Documents

- **STIMULUS_ALIGNED_TRAINING.md** - Why stimulus alignment matters
- **TRAINING_IMPROVEMENTS_TODO.md** - Complete roadmap with 8 improvements
- **STIMULUS_ALIGNMENT_SESSION_COMPLETE.md** - This session summary
- **EEGNEX_ROCM_STRATEGY.md** - EEGNeX training guide
- **TRAINING_DATA_ANALYSIS.md** - Available data and techniques

## 💡 Key Reminders

1. **Always use stimulus-aligned windows**
   - Anchor must be "stimulus_anchor"
   - Descriptor must match anchor
   - Response time is relative to stimulus onset

2. **Keep validation separate**
   - Don't train on R5
   - Use R5 only for validation
   - R6 is final test set

3. **Log everything**
   - Save training logs
   - Track metrics over time
   - Document all changes

4. **Start simple, add complexity**
   - Test stimulus alignment first
   - Add one improvement at a time
   - Measure impact of each change

## 🎬 Next Command

```bash
cd /home/kevin/Projects/eeg2025
python scripts/training/challenge1/train_challenge1_multi_release.py
```

**Look for:** "Creating STIMULUS-ALIGNED windows from trials..."

**Expected:** Better NRMSE than 1.00 baseline! 🎯

---

**Status:** Ready to train with properly aligned windows!  
**Confidence:** HIGH  
**Risk:** LOW  
**Path to Top 10:** Clear and achievable! 🏆
