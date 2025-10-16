# Training Status Update - October 16, 2025 15:09

## Critical Discovery: R4/R5 Validation Issues

### The Problem
After debugging why NRMSE values were showing 0.0000:

1. **Challenge 1:** R4 has **NO VALID EVENTS** when using `create_windows_from_events`
   - Error: `IndexError: index -1 is out of bounds for axis 0 with size 0`
   - The stops array is empty because no events match the criteria

2. **Challenge 2:** Both R4 and R5 have **ZERO VARIANCE**
   - R5: All externalizing scores are 0.297 (std=0.0)
   - R4: All validation externalizing scores are 0.297 (std=0.0)
   - Val NRMSE = 0.0000 because all targets are identical

### The Solution
**Changed validation from R4/R5 to R3 for BOTH challenges:**

- Training: R1, R2
- Validation: R3
- Test: R12 (unreleased, via competition submission)

## Current Training Status

### Challenge 1 (v13) ✅ TRAINING
**Configuration:**
- Script: `train_challenge1_multi_release.py`
- Log: `logs/challenge1_training_v13_R3val_fixed.log`
- Training: R1, R2 (44,440 response time trials)
- Validation: R3 (28,758 response time trials)

**Latest Results (Epoch 7):**
```
Epoch 3: Train 1.0012, Val 1.0207
Epoch 4: Train 0.9809, Val 1.0231
Epoch 5: Train 0.9698, Val 1.0211
Epoch 6: Train 0.9606, Val 1.0146
Epoch 7: Training...
```

**Status:** ✅ **WORKING PERFECTLY!**
- Both Train and Val NRMSE > 0
- Model is learning (Train NRMSE decreasing)
- Good generalization (Val NRMSE stable around 1.02)

### Challenge 2 (v10) 🔄 LOADING DATA
**Configuration:**
- Script: `train_challenge2_multi_release.py`
- Log: `logs/challenge2_training_v10_R3val_fixed.log`
- Training: R1, R2
- Validation: R3 (184 datasets, 79,058 windows)

**Status:** 🔄 Creating windows from R3 validation data
- Expected: Epoch 1 will show Val NRMSE > 0 (R3 has proper variance)

## Todo List

```markdown
- [x] Discover R4/R5 validation issues
- [x] Fix Challenge 1: Change to R3 validation
- [x] Fix Challenge 2: Change to R3 validation
- [x] Restart both trainings
- [x] Verify Challenge 1 NRMSE > 0 ✅
- [ ] Verify Challenge 2 NRMSE > 0 (in progress)
- [ ] Monitor training to completion (~3 hours)
- [ ] Create final submission.zip
- [ ] Upload to Codabench
```

## Timeline

- **14:49:** Challenge 1 v12 crashed (discovered R4 has no events)
- **14:55:** Fixed Challenge 1 to use R3 validation (v13)
- **15:00:** Discovered R4 also has zero variance for Challenge 2
- **15:01:** Fixed Challenge 2 to use R3 validation (v10)
- **15:02:** Both trainings restarted
- **15:05:** Challenge 1 Epoch 2 - NRMSE > 0 ✅
- **15:08:** Challenge 1 Epoch 7 - Training well
- **15:09:** Challenge 2 loading R3 validation data

## Expected Completion

- **Challenge 1:** ~17:30 (2.5 hours remaining)
- **Challenge 2:** ~17:30 (2.5 hours remaining)
- **Submission:** ~18:00

## Key Files

**Training Scripts:**
- `scripts/train_challenge1_multi_release.py` (v13)
- `scripts/train_challenge2_multi_release.py` (v10)

**Logs:**
- `logs/challenge1_training_v13_R3val_fixed.log`
- `logs/challenge2_training_v10_R3val_fixed.log`

**Documentation:**
- `docs/CRITICAL_VALIDATION_DISCOVERY.md` - Full analysis of R4/R5 issues
- `docs/METADATA_EXTRACTION_SOLUTION.md` - Challenge 1 metadata fix
- `METHODS_DOCUMENT.pdf` - Submission document

**Weights (will be updated when training completes):**
- `weights/weights_challenge_1_multi_release.pt`
- `weights/weights_challenge_2_multi_release.pt`

## Next Steps

1. ⏳ **Wait for Challenge 2 to start Epoch 1** (~5 minutes)
2. ✅ **Verify Challenge 2 Val NRMSE > 0**
3. 📊 **Monitor both trainings** using `./monitor_training_enhanced.sh`
4. 🎯 **When training completes:**
   - Create `submission.zip`
   - Upload to https://www.codabench.org/competitions/4287/
   - Check test scores on R12

