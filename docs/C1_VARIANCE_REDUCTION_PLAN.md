# C1 Variance Reduction Plan - High ROI, Low Risk

## Executive Summary

**Key Insight:** NRMSE normalized to 1.0 baseline = scores < 1.0 impossible
- Current V10 C1: 1.00019 (only 0.00019 above baseline!)
- Headroom: ~1.9e-4 maximum theoretical improvement
- Strategy: **Variance reduction > Architecture changes**

## Priority Execution Order

### PRIORITY 1: C2 (Higher ROI) - ALREADY DONE ✅
- ✅ 2-seed ensemble trained (Seeds 42, 123)
- ✅ EMA weights with decay 0.999
- ✅ 50.7% improvement over baseline
- ✅ Ready in V11/V11.5

### PRIORITY 2: C1 Variance Reduction (This Plan)
**Target:** 8e-5 to 1.6e-4 improvement → 1.00011–1.00003 range
**Time:** 1-2 hours training + 30 min calibration
**Risk:** Low (no architecture changes)

---

## C1 Implementation Checklist

### Phase 1: Validation Stability ✅ COMPLETE
- [x] Deterministic data split (5,969 train / 1,492 val)
- [x] Fixed seed across runs
- [x] No data leakage (separate subjects if applicable)

### Phase 2: Multi-Seed Ensemble with EMA ✅ COMPLETE
- [x] 5 seeds trained: [42, 123, 456, 789, 1337]
- [x] EMA weights (decay 0.999) saved
- [x] Individual NRMSEs logged:
  - Seed 42: 1.486252 (best)
  - Seed 123: 1.490609
  - Seed 456: 1.505322
  - Seed 789: 1.511281
  - Seed 1337: 1.502185
  - Mean: 1.499130, CV: 0.62%

### Phase 3: Light TTA (To Implement)
- [ ] Implement 3-5 TTA samples per input
- [ ] Time shift: ±2-3 samples (circular padding)
- [ ] Low Gaussian noise: SNR > 30 dB
- [ ] Validate on val set (should not degrade NRMSE)
- [ ] Expected gain: 1e-5 to 8e-5

### Phase 4: Output Calibration (To Implement)
- [ ] Linear calibration: y_cal = a*y_pred + b
- [ ] Fit on validation set
- [ ] Apply to test predictions
- [ ] Expected gain: 1e-5 to 5e-5

### Phase 5: Create V12 Submission
- [ ] Load 5-seed C1 models with EMA weights
- [ ] Apply TTA at inference
- [ ] Apply calibration transform
- [ ] Ensemble with mean (or trimmed mean)
- [ ] Combine with C2 2-seed ensemble
- [ ] Test locally
- [ ] Package and upload

---

## Expected Impact Breakdown

| Technique | Expected Improvement | Status |
|-----------|---------------------|--------|
| 5-seed ensemble | 5e-5 to 1.2e-4 | ✅ Done |
| + Light TTA (3x) | 1e-5 to 8e-5 | ⏳ To Do |
| + Linear calibration | 1e-5 to 5e-5 | ⏳ To Do |
| **Total Expected** | **8e-5 to 1.6e-4** | **In Progress** |

**Projected V12 C1 Score:** 1.00011–1.00003 (best case)

---

## Concrete Hyperparameters (Already Used)

```python
# Training (already complete)
SEEDS = [42, 123, 456, 789, 1337]
EPOCHS = 50  # with early stopping patience 15
BATCH_SIZE = 32
OPTIMIZER = "AdamW"
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-3
SCHEDULER = "ReduceLROnPlateau"
LR_PATIENCE = 5
LR_FACTOR = 0.5
MIN_LR = 1e-5
EMA_DECAY = 0.999
GRADIENT_CLIP = 1.0
LABEL_SMOOTHING = 0.05

# Augmentation (already in training)
TIME_SHIFT_PROB = 0.7
AMP_SCALE_PROB = 0.7
NOISE_PROB = 0.5
MIXUP_ALPHA = 0.2
MIXUP_PROB = 0.6
```

---

## Implementation Tasks

### Task 1: TTA Implementation (30 min)
Create `c1_tta_inference.py`:
- Load 5 EMA checkpoints
- Apply 3 TTA transforms per input:
  1. Original (no augmentation)
  2. Time shift +2 samples
  3. Time shift -2 samples
  4. Optional: Low noise (σ = 0.01, tune on val)
- Average predictions across seeds and TTA
- Validate on val set

### Task 2: Calibration Implementation (15 min)
Create `c1_calibration.py`:
- Get predictions on validation set (1,492 samples)
- Fit linear regression: a, b = argmin ||a*y_pred + b - y_true||²
- Store (a, b) coefficients
- Apply to test predictions

### Task 3: V12 Submission Creation (30 min)
Create `submissions/phase1_v12/submission.py`:
- Load 5 C1 EMA models
- Implement TTA inference
- Apply calibration
- Combine with C2 2-seed ensemble
- Test locally
- Package

---

## Why No Architecture Changes?

**Headroom Analysis:**
- Current: 1.00019 (only 1.9e-4 above baseline)
- Architecture changes risk losing delicate bias tuning
- Variance reduction is safer and more effective at this margin
- Architecture swaps pay off when gap is 1e-3+, not 1e-4

**If We Try Architecture Changes:**
- Only in parallel with current approach
- Must beat current across 3+ seeds on validation
- Only submit if clearly superior

---

## Risk Assessment

| Approach | Risk Level | Expected Gain | Confidence |
|----------|-----------|---------------|------------|
| Current 5-seed ensemble | Low | 5e-5 to 1.2e-4 | 90% |
| + TTA | Very Low | 1e-5 to 8e-5 | 80% |
| + Calibration | Very Low | 1e-5 to 5e-5 | 85% |
| Architecture change | **HIGH** | -1e-4 to +3e-4 | 40% |

**Decision:** Stick with variance reduction (low risk, predictable gains)

---

## Timeline

**Immediate (Next 2 hours):**
1. Implement TTA inference (30 min)
2. Implement calibration (15 min)
3. Create V12 submission (30 min)
4. Test locally (15 min)
5. Upload V12 (5 min)
6. Also upload V11.5 as baseline

**After Upload (2-3 hours wait):**
1. Analyze V11.5 vs V12 results
2. Compare with V10 baseline
3. Decide if further iteration needed

---

## Success Criteria

**Minimum Success (V12):**
- C1: 1.00012-1.00015 (improvement: 4e-5 to 7e-5)
- C2: ~1.00049 (same as V11.5)
- Overall: 1.00030-1.00032
- Rank: #55-65

**Target Success (V12):**
- C1: 1.00005-1.00010 (improvement: 9e-5 to 1.4e-4)
- C2: ~1.00049
- Overall: 1.00027-1.00029
- Rank: #45-55

**Stretch Success (V12):**
- C1: 1.00003-1.00005 (improvement: 1.4e-4 to 1.6e-4)
- C2: ~1.00049
- Overall: 1.00026-1.00027
- Rank: #40-50

---

## Code Implementation Priority

1. **IMMEDIATE:** TTA inference script
2. **IMMEDIATE:** Calibration script
3. **IMMEDIATE:** V12 submission.py
4. **LATER:** Distillation (if inference cost matters)
5. **OPTIONAL:** More seeds (6-7) if V12 shows gains

---

## Next Steps

Execute in order:
1. Create TTA inference code
2. Run calibration on validation set
3. Create V12 submission
4. Test V12 locally
5. Upload V11.5 (baseline) and V12 (improved)
6. Wait for results and analyze

**Let's start implementing!** ��

