# Variance Reduction Strategy - EEG2025 Phase 2

**Date**: October 31, 2025  
**Current Status**: V10 submitted, score 1.00052 (rank #72)  
**Goal**: Push to ~1.00025 (rank #40-50) through variance reduction

---

## Key Insights from Expert Guidance

### Critical Understanding
- **NRMSE ≥ 1.0**: Scores < 1.0 are impossible; 1.0 = perfect baseline
- **Current position**: C1 at 1.00019 (only 0.00019 headroom), C2 at 1.00066 (0.00066 headroom)
- **Primary strategy**: Variance reduction > architecture changes
- **ROI prioritization**: C2 has 3-4x more potential improvement than C1

### Variance Reduction Techniques
1. **EMA (Exponential Moving Average)**: decay=0.999
2. **Multi-seed ensemble**: 3-7 seeds with careful validation
3. **Conservative augmentation**: Reduce probabilities and magnitudes
4. **Light TTA**: 3-5 augmentations, bias-free only
5. **Linear calibration**: Post-hoc y_cal = a*y_pred + b on validation

---

## Execution Plan

### Phase 2A: C2 Ensemble (PRIMARY - In Progress ✅)

**Status**: Training launched at 19:05 Oct 31

**Configuration**:
```python
Seeds: [42, 123, 456, 789, 1337]  # 5 models
Epochs: 25
EMA decay: 0.999
Early stopping: patience=10
LR scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
Mixup alpha: 0.15 (reduced from 0.2)
Augmentation: Conservative (tighter ranges)
```

**Training Details**:
- Architecture: EEGNeX (62,353 parameters)
- Optimizer: AdamW (lr=0.002, weight_decay=0.001)
- Validation: 30 subjects from ds005509-bdf
- Checkpoints: `checkpoints/c2_phase2_seed{seed}_ema_best.pt`

**Expected Timeline**:
- Per seed: ~1.5-2 hours
- Total: 7.5-10 hours
- ETA: Oct 31 night / Nov 1 morning

**Expected Improvement**:
- Baseline: 1.00066
- Target: 1.00035-1.00050
- Delta: -0.00016 to -0.00031 (24-47% error reduction)

**Next Steps**:
1. [ ] Monitor training (check logs/c2_phase2_ema_*.log)
2. [ ] Create ensemble submission.py with TTA
3. [ ] Add linear calibration (fit on validation)
4. [ ] Test locally
5. [ ] Submit V11

---

### Phase 2B: C1 Multi-Seed (SECONDARY - Ready to Launch)

**Status**: Script created, ready to launch after C2 completes

**Configuration**:
```python
Seeds: [42, 123, 456]  # 3 models (can expand to 5)
Epochs: 25
EMA decay: 0.999
Early stopping: patience=10
LR scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
Mixup alpha: 0.1 (very conservative)
Augmentation: Very conservative
```

**Training Details**:
- Architecture: CompactResponseTimeCNN (proven, 75K parameters)
- Optimizer: AdamW (lr=0.001, weight_decay=0.05)
- Validation: R4 dataset
- Checkpoints: `checkpoints/c1_multiseed_seed{seed}_ema_best.pt`

**Expected Timeline**:
- Per seed: ~20-30 minutes
- Total: 1-1.5 hours

**Expected Improvement**:
- Baseline: 1.00019
- Target: 1.00009-1.00012
- Delta: -0.00007 to -0.00010 (37-53% error reduction)

**Next Steps**:
1. [ ] Launch training (after C2 analysis)
2. [ ] Create ensemble with TTA
3. [ ] Linear calibration on R4
4. [ ] Combine with C2 for V12

---

## Implementation Details

### EMA Implementation
```python
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}  # EMA weights
        
    def update(self, model):
        # After each training step
        shadow = decay * shadow + (1-decay) * current_weights
    
    def apply_shadow(self, model):
        # Use for validation/test
        
    def restore(self, model):
        # Restore training weights
```

**Usage**:
- Update EMA every training step
- Apply shadow weights for validation
- Save shadow weights in checkpoint
- Restore for continued training

### Light TTA (Test-Time Augmentation)
```python
def predict_with_tta(model, X, n_aug=3):
    predictions = [model(X)]  # Original
    
    for _ in range(n_aug - 1):
        X_aug = X.clone()
        
        # Small time shift (±2-3 samples)
        shift = np.random.randint(-3, 4)
        X_aug = torch.roll(X_aug, shifts=shift, dims=2)
        
        # Low noise (SNR > 30 dB)
        noise = torch.randn_like(X_aug) * 0.01
        X_aug = X_aug + noise
        
        predictions.append(model(X_aug))
    
    return torch.stack(predictions).mean(dim=0)
```

**Rules**:
- Only bias-free augmentations
- Tune noise level on validation (shouldn't degrade score)
- 3-5 augmentations optimal

### Linear Calibration
```python
from sklearn.linear_model import Ridge

# Fit on validation
y_pred_val = ensemble_predict(X_val)
calibrator = Ridge(alpha=0.1)
calibrator.fit(y_pred_val.reshape(-1, 1), y_val)

# Apply to test
y_pred_test = ensemble_predict(X_test)
y_pred_cal = calibrator.predict(y_pred_test.reshape(-1, 1))
```

**Benefits**:
- Corrects small systematic bias
- Expected: 1e-5 to 5e-5 improvement
- No risk (always use validation to fit)

---

## Expected Outcomes

### V11 Submission (C2 Ensemble)
```
Challenge 1: 1.00019 (unchanged from V10)
Challenge 2: 1.00035 - 1.00050 (improved from 1.00066)
Overall:     1.00027 - 1.00035
Rank:        #55-65 (up from #72)
```

### V12 Submission (C1 + C2 Ensemble)
```
Challenge 1: 1.00009 - 1.00012 (improved from 1.00019)
Challenge 2: 1.00035 - 1.00050 (from V11)
Overall:     1.00022 - 1.00031
Rank:        #40-50 (up from #55-65)
```

---

## Risk Mitigation

### Validation Stability
- Fixed random seeds for data loaders
- Deterministic CUDA operations
- Same validation set across all seeds
- Log per-seed variance

### Early Stopping
- Patient: 10 epochs (prevents premature stopping)
- LR reduction: 5 epochs patience (allows recovery)
- Monitor for seed drift

### Conservative Approach
- Keep proven architectures
- Reduce augmentation probabilities
- Lower mixup alpha for regression
- Validate every change on val set

---

## Monitoring & Checkpoints

### C2 Phase 2 Monitoring
```bash
# Check training progress
tail -f logs/c2_phase2_ema_*.log

# Check if training complete
ls -lh checkpoints/c2_phase2_seed*_ema_best.pt

# Verify 5 checkpoints exist
ls checkpoints/c2_phase2_seed*_ema_best.pt | wc -l  # Should be 5
```

### Success Criteria
- All 5 seeds complete training
- Val loss variance < 5% CV (coefficient of variation)
- Mean val loss < 0.30 (current Phase 1: 0.252)
- No seed diverges (early stopping not at epoch 1-5)

---

## Next Actions (Prioritized)

### Immediate (Tonight/Tomorrow)
1. ✅ C2 Phase 2 training launched (ETA: 7-10 hours)
2. [ ] Monitor C2 training logs
3. [ ] Create ensemble submission template
4. [ ] Implement TTA & calibration code

### After C2 Completes
5. [ ] Analyze C2 ensemble results
6. [ ] Test ensemble locally
7. [ ] Submit V11 (C2 ensemble)
8. [ ] Launch C1 multi-seed training (1-1.5 hours)

### After C1 Completes
9. [ ] Analyze C1 ensemble results
10. [ ] Create combined V12 submission
11. [ ] Test V12 locally
12. [ ] Submit V12 (C1 + C2 ensemble)

---

## Code Artifacts

### Training Scripts
- `train_c2_phase2_ensemble_ema.py` - C2 ensemble with EMA ✅
- `train_c1_multiseed_ema.py` - C1 multi-seed with EMA ✅

### To Create
- `create_ensemble_submission.py` - Generate submission.py with ensemble
- `test_tta.py` - Test TTA on validation set
- `fit_calibrator.py` - Fit linear calibrator on validation
- `test_v11_locally.py` - Local testing before submission

---

## References

### Key Parameters
```python
# EMA
EMA_DECAY = 0.999

# Augmentation (Conservative)
C2: time_shift=0.5, amp_scale=0.5, noise=0.3, mixup=0.15
C1: time_shift=0.4, amp_scale=0.4, noise=0.2, mixup=0.1

# TTA
N_AUGMENTATIONS = 3
TIME_SHIFT_RANGE = (-3, 4)
NOISE_STD = 0.01

# Calibration
RIDGE_ALPHA = 0.1
```

### Files to Monitor
```
logs/c2_phase2_ema_20251031_*.log
checkpoints/c2_phase2_seed*_ema_best.pt
checkpoints/c1_multiseed_seed*_ema_best.pt
```

---

**Status**: Phase 2A in progress, Phase 2B ready  
**Next Check**: Monitor C2 training (ETA: 7-10 hours)  
**Goal**: V11 within 24 hours, V12 within 36 hours
