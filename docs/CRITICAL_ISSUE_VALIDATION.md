# CRITICAL ISSUE: Validation Set Has Zero Variance

**Discovered:** October 16, 2025 14:15
**Status:** 🚨 BLOCKING ISSUE

---

## Problem

### Challenge 1: Response Time
- **Train NRMSE:** 0.0 (all targets are 0.0)
- **Val NRMSE:** 0.0 (all targets are 0.0)
- **Root Cause:** `rt_from_stimulus` is returning 0.0 for all windows

### Challenge 2: Externalizing
- **Train NRMSE:** 0.8494 (learning! ✅)
- **Val NRMSE:** 20,000,000 (completely broken ❌)
- **Root Cause:** R5 validation set has std = 0.0 (all same value)

```
Training (R1-R4):  Range [-0.387, 0.620], Mean 0.203, Std 0.352 ✅
Validation (R5):   Range [-0.364, -0.364], Mean -0.364, Std 0.000 ❌
```

**NRMSE = RMSE / std**
- When std = 0, NRMSE = RMSE / 0 = infinity

---

## Solutions

### Option 1: Use Hold-Out from Training Set (RECOMMENDED)
Instead of using R5 as validation, split R1-R4:
- **Train:** R1, R2, R3 (75%)
- **Val:** R4 (25%)
- **This ensures validation has variance**

### Option 2: Stratified Split Within Multi-Release
- Combine R1-R5
- Do 80/20 train/val split
- Ensures both have variance

### Option 3: Use R5 for Testing Only
- Train on R1-R4
- Validate on held-out portion of R1-R4
- Test final model on R5 (even with zero variance, can still evaluate)

---

## Immediate Actions

1. ❌ **STOP current training** - it's producing invalid results
2. 🔧 **Fix Challenge 1** - debug why rt_from_stimulus is 0.0
3. 🔄 **Restructure validation** - use R1-R3 train, R4 val
4. ✅ **Restart training** with corrected approach

---

## Challenge 1 Specific Issue

**All targets are 0.0!** This means:
- `rt_from_stimulus` is not in metadata, OR
- All values are being replaced with default 0.0

Need to:
1. Print first 10 metadata dicts to see what's actually there
2. Check if field name is correct
3. Verify preprocessing step is working

---

## Updated Training Strategy

### NEW APPROACH:
```python
# Train on R1, R2, R3
train_dataset = MultiReleaseDataset(releases=['R1', 'R2', 'R3'])

# Validate on R4
val_dataset = MultiReleaseDataset(releases=['R4'])

# After training, test on R5 (even with zero variance)
# NRMSE won't be meaningful, but predictions still valid
```

This ensures:
- Training has diverse data (R1-R3)
- Validation has diverse data (R4)
- Both can compute meaningful NRMSE

---

## Timeline Impact

- Current training: INVALID, must restart
- Fix and restart: ~1 hour
- Complete training: ~3 hours from restart
- **New estimated completion: ~18:00 (6 PM)**

---

## Next Steps

1. Kill current training
2. Fix Challenge 1 metadata extraction
3. Change validation from R5 to R4
4. Print debug info on first batch
5. Restart training
6. Monitor first epoch carefully

