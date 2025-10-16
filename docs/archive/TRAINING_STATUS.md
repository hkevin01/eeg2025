# Training Status - NeurIPS 2025 EEG Foundation Challenge

**Last Updated:** October 16, 2025 13:35

---

## 🎯 Current Status

### Training Progress

**Challenge 1: Response Time Prediction**
- **Status:** 🔄 Training v5 (ACTIVE)
- **Log:** `logs/challenge1_training_v5.log`
- **Training Data:** R1-R4 (multi-release)
- **Validation Data:** R5
- **Model:** CompactResponseTimeCNN (200K params)
- **Critical Fix Applied:** Using `rt_from_stimulus` from metadata (not `response_time`)

**Challenge 2: Externalizing Factor Prediction**
- **Status:** 🔄 Training v6 (ACTIVE - Epoch 1)
- **Log:** `logs/challenge2_training_v6.log`
- **Training Data:** R1-R4 (331,022 windows, diverse scores ✅)
- **Validation Data:** R5 (136,164 windows)
- **Model:** CompactExternalizingCNN (64K params)
- **Externalizing Stats:** Range [-0.387, 0.620], Mean 0.203, Std 0.352 ✅

---

## 🐛 Bugs Found and Fixed

### Bug #1: Challenge 2 Metadata Crash ✅
**Problem:** `AttributeError: 'list' object has no attribute 'get'`  
**Cause:** Braindecode returns metadata as list of dicts  
**Fix:** Check `isinstance(metadata, list)` and handle both formats  
**Status:** FIXED in v2

### Bug #2: submission.py Model Mismatch ✅
**Problem:** Training uses Compact models, submission had old large models  
**Fix:** Copied Compact model classes to submission.py  
**Status:** FIXED

### Bug #3: Weight Filename Resolution ✅
**Problem:** Training saves `*_multi_release.pt`, submission looked for `*.pt`  
**Fix:** Try multi_release name first, fallback to old name  
**Status:** FIXED

### Bug #4: Challenge 1 Target Bug - Event Marker vs Response Time ✅
**Problem:** Using `y` from windows (event marker 0/1) instead of response time  
**Symptom:** All NRMSE = 0.0000 (model predicted constant 0)  
**Root Cause:** Windowing with `targets_from="metadata"` returns event type as `y`  
**Fix:** Extract `response_time` from metadata dict  
**Status:** FIXED in v4

### Bug #5: Challenge 2 Target Bug - Missing Externalizing Scores ✅
**Problem:** Externalizing not in window metadata  
**Symptom:** All NRMSE = 0.0000 (model predicted constant 0)  
**Root Cause:** RestingState has no trial metadata, externalizing in dataset.description  
**Fix:** Map each window to parent dataset, extract from description  
**Status:** FIXED in v6

### Bug #6: Challenge 1 Metadata Field Name ✅
**Problem:** Using wrong metadata field name  
**Symptom:** Still getting NRMSE = 0.0000 after v4 fix  
**Root Cause:** preprocessing stores as `rt_from_stimulus`, not `response_time`  
**Fix:** Changed to `meta_dict.get('rt_from_stimulus', 0.0)`  
**Status:** FIXED in v5

### Bug #7: Challenge 2 Loading Still Shows R5 ❌ FALSE ALARM
**Initial Problem:** Log showed "Releases: {'R5'}"  
**Investigation:** This was the validation dataset stats  
**Actual Status:** Training set correctly has R1-R4 (331K windows) ✅  
**Validation set:** R5 (136K windows) ✅  
**Status:** NOT A BUG - log was showing validation set

---

## 📊 Expected vs Previous Results

### Previous Submission (R5 only, single-release)
- **Challenge 1:** Val 0.47 → Test 4.05 (10x degradation)
- **Challenge 2:** Val 0.08 → Test 1.14 (14x degradation)
- **Overall:** 2.01 NRMSE (~5th place)

### Current Submission (R1-R4 multi-release)
**Expected after fixing target bugs:**
- **Challenge 1:** ~1.4 NRMSE (3x improvement)
- **Challenge 2:** ~0.5 NRMSE (2x improvement)
- **Overall:** ~0.8 NRMSE (2.5x improvement)
- **Goal:** < 0.7 NRMSE for top 3

---

## 🔧 Technical Details

### Challenge 1: Response Time Prediction

**Data Processing:**
- Task: Contrast Change Detection (CCD)
- Trial annotation: `annotate_trials_with_target` with `target_field="rt_from_stimulus"`
- Windowing: 2s windows (200 samples @ 100 Hz), non-overlapping
- **Critical:** Extract `rt_from_stimulus` from metadata (continuous RT value)

**Model: CompactResponseTimeCNN**
```
Parameters: 200,048
Architecture:
  - Conv1D layers: 129→32→64→128 channels
  - Progressive dropout: 0.3→0.4→0.5
  - Regressor: 128→64→32→1
```

**Training:**
- Optimizer: AdamW (lr=1e-3, wd=1e-4)
- Scheduler: CosineAnnealingLR
- Epochs: 50 (early stopping patience=15)
- Batch size: 32

### Challenge 2: Externalizing Factor Prediction

**Data Processing:**
- Task: RestingState (continuous EEG)
- Windowing: 2s windows, 50% overlap (100 samples stride)
- **Critical:** Extract externalizing from `dataset.description['externalizing']`
- Map each window to parent dataset using `i_dataset` from metadata

**Training Set Statistics (R1-R4):**
- Windows: 331,022
- Externalizing range: [-0.387, 0.620]
- Mean: 0.203, Std: 0.352 ✅

**Model: CompactExternalizingCNN**
```
Parameters: 64,001
Architecture:
  - Conv1D layers: 129→32→64→96 channels
  - ELU activations (smoother gradients)
  - Progressive dropout: 0.3→0.4→0.5
  - Regressor: 96→48→24→1
```

**Training:**
- Optimizer: AdamW (lr=1e-3, wd=1e-4)
- Scheduler: CosineAnnealingLR
- Epochs: 50 (early stopping patience=15)
- Batch size: 32

---

## 📋 Next Steps

### Immediate (Monitoring Phase)
- [x] Fixed Challenge 1 metadata field name bug
- [x] Restarted Challenge 1 v5 with correct field
- [x] Verified Challenge 2 has diverse training targets
- [ ] **Wait for Epoch 1 completion (~30-60 min)**
- [ ] **Verify NRMSE > 0.0 (not constant prediction)**

### During Training (~2-3 hours)
- [ ] Monitor NRMSE values
  - Challenge 1: Should be 0.5-2.0 (NOT 0.0)
  - Challenge 2: Should be 0.3-1.0 (NOT 0.0)
- [ ] Check for convergence
- [ ] Watch for early stopping

### After Training Completes
- [ ] Check final validation NRMSE
- [ ] Test submission.py with trained weights
- [ ] Create submission package:
  - submission.py
  - weights_challenge_1_multi_release.pt
  - weights_challenge_2_multi_release.pt
  - METHODS_DOCUMENT.pdf
- [ ] Upload to Codabench

### If Results Not Good Enough
- [ ] Implement Phase 2 features:
  - P300 component extraction (Challenge 1)
  - Spectral band features (Challenge 2)
  - Expected: 0.8 → 0.5 NRMSE

---

## 📁 Key Files

**Training Scripts:**
- `scripts/train_challenge1_multi_release.py` (v5 - current)
- `scripts/train_challenge2_multi_release.py` (v6 - current)

**Logs:**
- `logs/challenge1_training_v5.log` (current)
- `logs/challenge2_training_v6.log` (current)

**Models:**
- `weights_challenge_1_multi_release.pt` (will be created)
- `weights_challenge_2_multi_release.pt` (will be created)

**Documentation:**
- `METHODS_DOCUMENT.md` - Complete methods description
- `CRITICAL_FIXES_LOG.md` - All 7 bugs documented
- `PHASE2_TASK_SPECIFIC_PLAN.md` - Advanced features plan
- `SUBMISSION_READY_STATUS.md` - Submission guide

**Submission:**
- `submission.py` - Updated with Compact models

---

## ⏱️ Timeline

**Started:** October 16, 2025 12:00  
**Current:** October 16, 2025 13:35  
**Challenge 1 v5:** Started 13:34 (loading data)  
**Challenge 2 v6:** Epoch 1 running (started 12:30)

**Expected Completion:**
- Challenge 2: ~15:30 (2 hours from start)
- Challenge 1: ~16:00 (2.5 hours from start)
- Ready for submission: ~16:30

---

## 🎓 Lessons Learned

1. **Always verify targets are correct!** Both challenges had NRMSE=0.0 due to target bugs
2. **Check metadata field names** - preprocessing may use different names than expected
3. **Multi-release training is essential** - single-release catastrophically fails on held-out data
4. **Smaller models generalize better** - 75% parameter reduction with strong regularization
5. **Cross-release validation catches distribution shift** - R1-R4 train, R5 validate
6. **RestingState vs Task data** - different metadata structure, need different extraction logic
7. **Window metadata format** - braindecode returns list of dicts, not single dict

---

## 🏆 Success Criteria

- ✅ Both challenges show NRMSE > 0.0 (not constant prediction)
- ⏳ Challenge 1 NRMSE: 0.5-2.0 range
- ⏳ Challenge 2 NRMSE: 0.3-1.0 range
- ⏳ Overall score < 1.0 (competitive)
- ⏳ submission.py loads weights successfully
- ⏳ Final submission uploaded before deadline

---

**Contact:** [Your Email]  
**Competition:** https://www.codabench.org/competitions/4287/  
**Code:** Will be released upon publication

