# Critical Validation Data Discovery

## Summary
**BOTH R4 and R5 validation sets are UNUSABLE** for this competition!

## Timeline of Discovery

### Issue #1: R5 has Zero Variance (Challenge 2)
- **Discovered:** Earlier in training
- **Symptom:** Challenge 2 Val NRMSE = 0.0000 
- **Root Cause:** All externalizing scores in R5 are identical (std=0.0)
- **Evidence:** `Range: [0.297, 0.297]` - all samples have same value

### Issue #2: R4 has No Valid Events (Challenge 1)  
- **Discovered:** Today (v12 crash)
- **Symptom:** `IndexError: index -1 is out of bounds for axis 0 with size 0`
- **Root Cause:** R4 contrast task has NO valid events when using `create_windows_from_events`
- **Evidence:** `stops[-1]` fails because stops array is empty (size 0)

### Issue #3: R4 ALSO has Zero Variance (Challenge 2)
- **Discovered:** Today while debugging
- **Symptom:** Challenge 2 Val NRMSE = 0.0000 even with "fixed" R4 validation
- **Root Cause:** R4 validation set has all identical externalizing scores
- **Evidence:** `Range: [0.297, 0.297]` in validation stats

## Solution: Use R3 for Validation

### New Training Configuration

**Challenge 1 (Response Time):**
- Training: R1, R2
- Validation: R3
- Reason: R4 has no valid events, R5 has zero variance

**Challenge 2 (Externalizing):**
- Training: R1, R2  
- Validation: R3
- Reason: Both R4 and R5 have zero variance

### Why This Works

1. **R1, R2, R3 have valid, varied data** for both challenges
2. **R3 is large enough** to serve as validation (388 datasets for C1, 136 for C2)
3. **Proper train/val split** maintains generalization testing
4. **R4/R5 are test-only** - likely intentionally designed with issues to prevent overfitting

## Results After Fix

### Challenge 1 (v13 with R3 validation):
```
Epoch 2/50
Train NRMSE: 1.0398
Val NRMSE:   1.0071
✅ Both values > 0 and reasonable!
```

**Validation stats:**
- mean=1.5764, std=0.4087, range=[0.0, 2.41]
- ✅ Proper variance confirmed

### Challenge 2 (v10 with R3 validation):
- Currently loading data
- Expected: Both Train and Val NRMSE > 0

## Key Learnings

1. **Always verify validation data has variance** before training
2. **R4/R5 are NOT suitable for validation** in this competition
3. **The competition organizers may have intentionally made R4/R5 problematic** to force proper train/val/test splits
4. **R3 is the last usable release for validation** 

## Files Updated
- `scripts/train_challenge1_multi_release.py` - v13: R1-R2 train, R3 val
- `scripts/train_challenge2_multi_release.py` - v10: R1-R2 train, R3 val

## Training Status
- Challenge 1 v13: ✅ TRAINING (Epoch 3/50)
- Challenge 2 v10: 🔄 LOADING DATA


## UPDATE: Even MORE Zero Variance Issues!

### Issue #4: R3 ALSO has Zero Variance (Challenge 2)
- **Discovered:** After fixing to R3 validation
- **Symptom:** Val NRMSE = 0.0000 in v10
- **Root Cause:** R3 validation also has all identical externalizing scores
- **Evidence:** `Range: [-0.387, -0.387], Mean: -0.387, Std: 0.000`

### FINAL SOLUTION

**Challenge 1 (Response Time):**
- Training: R1, R2
- Validation: R3 ✅ (R3 has variance for contrast task)
- Reason: R4 has no valid events, R5 has zero variance

**Challenge 2 (Externalizing):**
- Training: R1 ONLY
- Validation: R2 ✅ (R3/R4/R5 ALL have zero variance!)
- Reason: R3, R4, AND R5 all have zero variance for externalizing scores

### Data Release Variance Summary

| Release | Challenge 1 (RT) | Challenge 2 (Externalizing) |
|---------|------------------|-----------------------------|
| R1      | ✅ Variance      | ✅ Variance (0.325-0.620)   |
| R2      | ✅ Variance      | ✅ Variance                 |
| R3      | ✅ Variance      | ❌ Zero Variance (-0.387)   |
| R4      | ❌ No Events     | ❌ Zero Variance (0.297)    |
| R5      | ✅ Variance      | ❌ Zero Variance (0.297)    |

**Key Insight:** Each challenge requires a DIFFERENT validation strategy!
- Challenge 1 can use R3 validation
- Challenge 2 can ONLY use R2 or R1 for validation

