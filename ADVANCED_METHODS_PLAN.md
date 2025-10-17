# 🚀 Advanced Methods Implementation Plan

**Date:** October 16, 2025  
**Status:** Post-analysis, ready for Priority 1 improvements  
**Current Position:** #47 → Target: #15-20

---

## 🎯 Competition Constraints & Rules

### Allowed Methods
✅ Any ML/DL architecture (CNN, RNN, Transformer, etc.)
✅ Feature extraction from EEG signals
✅ Ensemble methods
✅ Domain adaptation techniques
✅ Cross-validation strategies
✅ Data augmentation (EEG-appropriate)
✅ Regularization techniques

### Constraints
⚠️ No external EEG datasets allowed
⚠️ No leakage between train/test splits
⚠️ Must use provided releases only (R1-R5)
⚠️ Inference time limit (likely ~30s per sample)
⚠️ Model size reasonable for submission

### Current Data Availability
- **Training:** R1, R2, R3 (available locally)
- **Validation:** R3 (can use any split strategy)
- **Test:** R4, R5 (held out by organizers)

---

## 📊 Prioritized Methods (Competition-Compliant)

### 🔴 Priority 1: Release-Aware Cross-Validation (TONIGHT - 3 hours)

**Status:** ✅ **APPROVED - HIGHEST IMPACT**

**Why This First:**
- Addresses root cause of 4x overfitting
- Uses all available data (R1+R2+R3)
- No new algorithms needed - just better data strategy
- Expected: 2.01 → 1.3-1.5 (30-40% improvement)

**Implementation:**
```python
# 3-Fold Strategy: Each fold holds out one release
Fold 1: Train R1+R2, Val R3
Fold 2: Train R1+R3, Val R2
Fold 3: Train R2+R3, Val R1

# Ensemble: Median of 3 predictions (robust to outliers)
```

**Files to Create:**
- `scripts/train_challenge1_grouped_cv.py`
- `scripts/train_challenge2_grouped_cv.py`
- `scripts/ensemble_predictions.py`

**Time:** 3 hours (1h implementation + 2h training)

---

### 🟠 Priority 2: Robust Loss Functions (TOMORROW AM - 1 hour)

**Status:** ✅ **APPROVED - MEDIUM IMPACT**

**Why This Helps:**
- Huber loss robust to outliers in RT labels
- Reduces impact of mislabeled trials
- Simple to implement, proven effective
- Expected: Additional 5-10% improvement

**Implementation:**
```python
# Replace MSE with Huber loss
def huber_loss(pred, target, delta=1.0):
    err = pred - target
    abs_err = err.abs()
    quad = torch.clamp(abs_err, max=delta)
    lin = abs_err - quad
    return (0.5 * quad**2 + delta * lin).mean()

# Add residual-based reweighting after warmup
# Downweight samples with large residuals
```

**Files to Modify:**
- Update loss function in training scripts
- Add residual reweighting after epoch 5

**Time:** 1 hour (implementation + quick retrain test)

---

### 🟠 Priority 3: Multi-Scale CNN Architecture (TOMORROW PM - 2 hours)

**Status:** ✅ **APPROVED - MEDIUM IMPACT**

**Why This Helps:**
- Captures EEG patterns at multiple time scales
- Different frequency bands (delta, theta, alpha, beta, gamma)
- Squeeze-and-Excitation for channel attention
- Expected: 10-15% improvement

**Implementation:**
```python
class MultiScaleTCN(nn.Module):
    """Multi-scale temporal CNN with SE attention"""
    def __init__(self, in_ch=129, hidden=64, ks=(5,15,45,125)):
        # Parallel branches with different kernel sizes
        # 5 samples = 20ms (gamma)
        # 15 samples = 60ms (beta/alpha)
        # 45 samples = 180ms (theta)
        # 125 samples = 500ms (delta)
        
        self.branches = nn.ModuleList([...])
        self.se = SEBlock(hidden * len(ks))
        self.head = nn.Sequential(...)
```

**Files to Create:**
- `src/models/multi_scale_cnn.py`
- `scripts/train_challenge1_multiscale.py`
- `scripts/train_challenge2_multiscale.py`

**Time:** 2 hours (1h implementation + 1h training test)

---

### 🟡 Priority 4: CORAL Domain Alignment (WEEKEND - 2 hours)

**Status:** ✅ **APPROVED - LOW-MEDIUM IMPACT**

**Why This Helps:**
- Aligns feature distributions across releases
- Reduces domain shift between R1/R2/R3 → R4/R5
- Lightweight addition to existing training
- Expected: 5-10% improvement

**Implementation:**
```python
def coral_loss(h_s, h_t):
    """Align covariance matrices of source/target features"""
    # Compute on penultimate layer features
    # Add to total loss: loss_total = loss_main + lambda_coral * coral_loss
```

**Files to Modify:**
- Add CORAL loss to training loops
- Extract penultimate features in forward pass
- Hyperparameter: lambda_coral = 1e-3 to 1e-2

**Time:** 2 hours (implementation + ablation test)

---

### 🟡 Priority 5: EEG-Specific Augmentations (WEEKEND - 2 hours)

**Status:** ⚠️ **CONDITIONAL APPROVAL**

**Why This Helps:**
- Increases effective training data
- Improves generalization
- Must be physiologically valid!

**Approved Augmentations:**
```python
# ✅ Safe augmentations:
- Gaussian noise (SNR-matched, σ=0.05-0.1)
- Time shifting (±50ms, maintains trial structure)
- Channel dropout (p=0.1-0.2, simulates bad electrodes)
- Amplitude scaling (0.8-1.2x, mild)

# ❌ Avoid:
- Random time warping (distorts ERP timing!)
- Frequency masking (breaks physiological bands)
- Mixup on raw signals (creates non-physiological patterns)
```

**Implementation:**
```python
class EEGAugmentation:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, x):
        if random.random() < self.p:
            # Apply one random augmentation
            aug = random.choice([
                self.add_noise,
                self.time_shift,
                self.channel_dropout,
                self.amplitude_scale
            ])
            return aug(x)
        return x
```

**Time:** 2 hours (implementation + validation)

---

### ⭕ Priority 6: Subject Normalization (IF TIME - 1 hour)

**Status:** ⚠️ **NEEDS VALIDATION**

**Why This Might Help:**
- EEG varies significantly across subjects
- Z-score per subject reduces inter-subject variance
- BUT: Risk of leakage if done wrong!

**Safe Implementation:**
```python
# ✅ TRAIN TIME:
# Compute per-subject stats ONLY from training samples
subject_stats = {}
for subj in train_subjects:
    subj_samples = train_data[train_data['subject'] == subj]
    subject_stats[subj] = {
        'mean': subj_samples.mean(),
        'std': subj_samples.std()
    }

# ✅ TEST TIME:
# Use population stats OR online adaptation (first N trials)
# NO access to full test subject's data!
```

**Constraint:** Must verify no test-time leakage!

**Time:** 1 hour (careful implementation)

---

## ❌ Methods to EXCLUDE (Violate Constraints or Ineffective)

### ❌ Domain-Adversarial Training (GRL)
**Reason:** Requires release labels at test time (R4/R5 identity unknown)  
**Alternative:** Use CORAL instead (doesn't need release labels)

### ❌ FiLM with Subject Embeddings
**Reason:** Would require test subject metadata (may not be available)  
**Alternative:** Use population-level normalization

### ❌ Trial Context Windows (Sequential Models)
**Reason:** Test data likely presented as individual trials, no sequence  
**Alternative:** Focus on single-trial features

### ❌ P300 Features
**Reason:** Already validated - correlation = 0.007 (useless!)  
**Status:** Abandoned

### ❌ External EEG Datasets
**Reason:** Competition rules prohibit external data  
**Status:** Not allowed

### ❌ Stacked Generalization (Meta-Learner)
**Reason:** OOF predictions may not generalize to R4/R5  
**Alternative:** Simple ensemble (median/mean) is safer

---

## 📋 Implementation Todo List

### Phase 1: Tonight (3 hours) - Priority 1

```markdown
- [ ] Create grouped CV training scripts
  - [ ] train_challenge1_grouped_cv.py
  - [ ] train_challenge2_grouped_cv.py
- [ ] Implement 3-fold strategy (R1+R2→R3, R1+R3→R2, R2+R3→R1)
- [ ] Train 6 models (3 folds × 2 challenges)
- [ ] Save fold predictions and weights
- [ ] Create ensemble_predictions.py
- [ ] Generate submission_v2.zip
- [ ] Upload and check leaderboard
```

**Expected Result:** Rank #25-30 (from #47)

---

### Phase 2: Tomorrow (3 hours) - Priority 2 + 3

```markdown
- [ ] Implement Huber loss function
- [ ] Add residual reweighting (warmup 5 epochs)
- [ ] Test on one fold (quick validation)
- [ ] Implement MultiScaleTCN architecture
  - [ ] Parallel branches (ks=5,15,45,125)
  - [ ] SE attention block
  - [ ] Keep params ~400-600K
- [ ] Train with new architecture + Huber loss
- [ ] Compare validation scores
- [ ] Generate submission_v3.zip if better
```

**Expected Result:** Rank #15-20 (from #25-30)

---

### Phase 3: Weekend (4 hours) - Priority 4 + 5

```markdown
- [ ] Implement CORAL loss
  - [ ] Add to training loop
  - [ ] Tune lambda_coral (1e-3, 5e-3, 1e-2)
- [ ] Implement EEG augmentations
  - [ ] Gaussian noise (σ=0.1)
  - [ ] Time shift (±50ms)
  - [ ] Channel dropout (p=0.15)
  - [ ] Amplitude scale (0.9-1.1)
- [ ] Ablation study (which helps most?)
- [ ] Train with best combination
- [ ] Generate submission_v4.zip
```

**Expected Result:** Rank #10-15 (from #15-20)

---

## 🔬 Validation Strategy

### Local Validation Protocol

**DO NOT trust single-fold validation!**

```python
# ✅ Correct validation:
1. 3-fold grouped CV (by release)
2. Compute OOF (out-of-fold) predictions
3. Report OOF NRMSE as validation score
4. This approximates test performance better!

# ❌ Wrong validation:
1. Train R1+R2, val R3
2. Report R3 score
3. Overestimates performance (as we learned!)
```

### Sanity Checks Before Submission

```python
# 1. No data leakage
assert len(set(train_indices) & set(val_indices)) == 0

# 2. Grouped by release
for fold in folds:
    train_releases = get_releases(train_idx)
    val_releases = get_releases(val_idx)
    assert len(set(train_releases) & set(val_releases)) == 0

# 3. Model size reasonable
model_size_mb = os.path.getsize('model.pt') / 1e6
assert model_size_mb < 50  # <50 MB

# 4. Predictions in valid range
assert pred.min() >= 0  # RT cannot be negative
assert pred.max() < 10  # RT unlikely >10s

# 5. No NaN/Inf
assert not torch.isnan(pred).any()
assert not torch.isinf(pred).any()
```

---

## 📊 Expected Performance Trajectory

| Version | Methods | Val NRMSE (OOF) | Est. Test NRMSE | Est. Rank | Time |
|---------|---------|-----------------|-----------------|-----------|------|
| v1 (current) | R1+R2 train, R3 val | 0.65 | 2.01 (actual) | #47 | Done |
| v2 (tonight) | 3-fold CV + ensemble | 0.75-0.85 | 1.4-1.6 | #25-30 | 3h |
| v3 (tomorrow) | + Huber + MultiScale | 0.70-0.80 | 1.2-1.4 | #15-20 | 3h |
| v4 (weekend) | + CORAL + augment | 0.65-0.75 | 1.0-1.2 | #10-15 | 4h |

**Key Insight:** OOF validation will be "worse" (higher) than current R3-only validation, but test performance will be MUCH better!

---

## 🎯 Success Metrics

### Minimum Acceptable Result (v2)
- OOF NRMSE < 0.90 (expect ~0.80)
- Test Overall < 1.8
- Rank < 35

### Target Result (v3)
- OOF NRMSE < 0.80 (expect ~0.75)
- Test Overall < 1.4
- Rank < 25

### Stretch Goal (v4)
- OOF NRMSE < 0.75 (expect ~0.70)
- Test Overall < 1.2
- Rank < 15

---

## 🚨 Red Flags to Watch

### During Training
- **Validation worse than training:** Normal (regularization working)
- **Validation increasing:** Stop early! Overfitting
- **NaN losses:** Reduce learning rate or check data
- **GPU OOM:** Reduce batch size

### After Submission
- **Test >> OOF:** Still overfitting (but less than v1!)
- **Test << OOF:** Unrealistic (unlikely)
- **Test ≈ OOF:** Good generalization! 🎉

---

## 💡 Key Principles

1. **Data Strategy > Architecture:** Fix data split first (Priority 1)
2. **Simple Ensemble > Complex Meta-Learner:** Median is robust
3. **Validation Realism:** OOF CV approximates test better
4. **Conservative Augmentation:** Only physiologically valid
5. **Regularization > Capacity:** Dropout + weight decay before bigger model
6. **Measure Everything:** Log per-release, per-subject metrics

---

## 📚 References

**Approved Techniques:**
- Release-Grouped CV: sklearn.GroupKFold
- Huber Loss: PyTorch built-in
- Multi-Scale CNN: Proven in EEG literature
- CORAL: "Deep CORAL: Correlation Alignment for Deep Domain Adaptation"
- SE Blocks: "Squeeze-and-Excitation Networks"

**Rejected Techniques:**
- DANN/GRL: Needs test domain labels
- IRM: Complex, marginal benefit
- MMD: Similar to CORAL, more compute
- Trial Context: Test data structure unclear

---

## 🎯 Next Action

**START NOW with Priority 1!**

```bash
# 1. Create grouped CV script
cp scripts/train_challenge1_multi_release.py scripts/train_challenge1_grouped_cv.py

# 2. Implement 3-fold strategy
# (See IMPROVEMENT_ROADMAP.md for detailed code)

# 3. Start training
python scripts/train_challenge1_grouped_cv.py --fold 0 &
python scripts/train_challenge1_grouped_cv.py --fold 1 &
python scripts/train_challenge1_grouped_cv.py --fold 2 &

# Expected: 3 hours → Rank #25-30
```

Let's improve that rank! 🚀
