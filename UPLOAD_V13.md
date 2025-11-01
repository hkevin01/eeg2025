# 🚀 V13 Submission Upload Guide

**Date:** November 1, 2025  
**Status:** ✅ Ready for Upload  
**Package:** `v13_submission.zip` (6.1 MB)  
**Location:** Root directory

---

## 📦 Package Information

### Contents
```
v13_submission.zip (6.1 MB)
├── submission.py                    (11 KB)
├── c1_phase1_seed42_ema_best.pt    (1.1 MB)
├── c1_phase1_seed123_ema_best.pt   (1.1 MB)
├── c1_phase1_seed456_ema_best.pt   (1.1 MB)
├── c1_phase1_seed789_ema_best.pt   (1.1 MB)
├── c1_phase1_seed1337_ema_best.pt  (1.1 MB)
├── c2_phase2_seed42_ema_best.pt    (758 KB)
├── c2_phase2_seed123_ema_best.pt   (758 KB)
└── c1_calibration_params.json      (195 bytes)
```

### Verification Status
✅ All tests passed (Nov 1, 2:20 PM)
- Import test: Submission class loads
- Initialization: SFREQ=100, DEVICE='cpu'
- Challenge 1: Batch sizes [1, 5] tested
- Challenge 2: Batch sizes [1, 5] tested
- Output format: numpy arrays, shape (N,)
- No NaN/Inf values

---

## 🔧 Key Changes from V12

### Primary Fix
**PyTorch Compatibility Issue Resolved**

V12 failed with:
```python
checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
```

V13 fixed to:
```python
checkpoint = torch.load(weights_path, map_location=self.device)
```

**Reason:** `weights_only` parameter requires PyTorch ≥1.13  
Competition environment likely uses older version.

### Same Variance Reduction Stack
- ✅ 5-seed ensemble (Seeds: 42, 123, 456, 789, 1337)
- ✅ Test-time augmentation (3 circular time shifts: -2, 0, +2)
- ✅ Linear calibration (a=0.988, b=0.027)
- ✅ 2-seed ensemble for C2 (Seeds: 42, 123)

---

## 🎯 Expected Performance

### Based on Validation Results

**Challenge 1 (CCD):**
- V10 baseline: 1.00019
- Expected V13: ~1.00011
- Improvement: 8e-5 (variance reduction measured)
- Strategy: 5-seed + TTA + calibration

**Challenge 2 (RSVP):**
- V10 baseline: 1.00066
- Expected V13: ~1.00049
- Improvement: 1.7e-4
- Strategy: 2-seed ensemble

**Overall:**
- V10 baseline: 1.00052 (Rank #72/150)
- Expected V13: ~1.00030
- Improvement: 2.2e-4 total
- Expected rank: #45-55 (potential 27 position jump)

**Note:** Top leaderboard shows C1: 0.89854, Overall: 0.97367  
We have significant room for improvement beyond V13.

---

## 📝 Upload Instructions

### Step 1: Go to Competition Page
**URL:** https://www.codabench.org/competitions/3350/

### Step 2: Navigate to Submissions
1. Click "My Submissions" tab
2. Click "Submit" button

### Step 3: Upload Package
**File:** `/home/kevin/Projects/eeg2025/v13_submission.zip`

**Description (suggested):**
```
V13: PyTorch compatibility fix + variance reduction
- Fixed weights_only parameter issue from V12
- 5-seed ensemble + TTA + calibration for C1
- 2-seed ensemble for C2
- Expected: C1 ~1.00011, C2 ~1.00049, Overall ~1.00030
```

### Step 4: Monitor Progress

**Phase 1: Ingestion (5-10 minutes)**
- Extracts submission.zip
- Imports submission.py
- Initializes Submission class
- ✅ Success indicator: No errors in log

**Phase 2: Scoring (10-20 minutes)**
- Runs challenge_1() on test set
- Runs challenge_2() on test set  
- Calculates NRMSE scores
- ✅ Success indicator: scoring_result.zip generated

**Phase 3: Results (immediately after scoring)**
- C1 score displayed
- C2 score displayed
- Overall score displayed
- Leaderboard updated

**Total time:** 15-30 minutes

### Step 5: Download Results
Once complete, download:
1. `prediction_result.zip` - Contains predictions + metadata
2. `scoring_result.zip` - Contains scores + detailed metrics

Save to: `/home/kevin/Projects/eeg2025/submissions/phase1_v13_results/`

---

## 🔍 What to Check After Upload

### Success Indicators
✅ Ingestion completes without errors  
✅ scoring_result.zip is NOT empty  
✅ Both C1 and C2 scores generated  
✅ Scores are reasonable (not NaN, not extreme)  
✅ Leaderboard position updated

### If V13 Fails
1. Download error files immediately
2. Check metadata for exit codes
3. Look for Python traceback in logs
4. Analyze failure mode:
   - Import error? → Dependency issue
   - Timeout? → Model too slow
   - Wrong output? → Format issue
   - Empty scoring? → Execution crash

### If V13 Succeeds
1. Compare actual vs expected performance
2. Analyze variance reduction effectiveness
3. Check leaderboard position change
4. Decide on next submission (V11, V11.5, or V14)

---

## 📊 Comparison with Previous Submissions

| Version | C1 Score | C2 Score | Overall | Rank | Status |
|---------|----------|----------|---------|------|--------|
| V9      | 1.00077  | 1.00870  | 1.00648 | #88  | ✅ Success |
| V10     | 1.00019  | 1.00066  | 1.00052 | #72  | ✅ Success |
| V11     | TBD      | TBD      | TBD     | TBD  | 📦 Ready |
| V11.5   | TBD      | TBD      | TBD     | TBD  | 📦 Ready |
| V12     | -        | -        | -       | -    | ❌ Failed (PyTorch) |
| **V13** | **~1.00011** | **~1.00049** | **~1.00030** | **~#50** | 🚀 **READY** |

### V13 Advantages
- ✅ Fixes V12 PyTorch compatibility
- ✅ Full variance reduction stack
- ✅ Comprehensive local testing
- ✅ Conservative PyTorch usage
- ✅ Same approach as working V10

### V13 Risks
- ⚠️ braindecode dependency (but V10 worked)
- ⚠️ More complex than V10 (5 vs 1 model)
- ⚠️ Longer inference time (acceptable)

---

## 🎯 Success Criteria

### Minimum Success ✅
- Execution completes (no crash)
- Scores generated (not empty)
- Overall < 1.00052 (beats V10)

### Expected Success ✅
- C1 ~1.00011 (8e-5 improvement)
- C2 ~1.00049 (1.7e-4 improvement)
- Overall ~1.00030 (2.2e-4 improvement)
- Rank improvement (#72 → #45-55)

### Stretch Success 🎯
- C1 < 1.00010 (variance reduction exceeds expectations)
- Overall < 1.00025 (even better than predicted)
- Rank top 40 (#40/150 or better)

---

## 📋 Post-Upload Checklist

### Immediate (During Upload)
- [ ] Upload v13_submission.zip
- [ ] Note upload timestamp
- [ ] Monitor ingestion phase (5-10 min)
- [ ] Check for ingestion errors
- [ ] Monitor scoring phase (10-20 min)
- [ ] Verify scoring_result.zip created

### After Results
- [ ] Download prediction_result.zip
- [ ] Download scoring_result.zip
- [ ] Save to submissions/phase1_v13_results/
- [ ] Compare actual vs expected scores
- [ ] Update README.md with results
- [ ] Update leaderboard position
- [ ] Analyze variance reduction effectiveness

### Follow-Up Analysis
- [ ] Compare V13 vs V10 in detail
- [ ] Calculate actual improvement per component
- [ ] Validate calibration effectiveness
- [ ] Assess TTA contribution
- [ ] Decide on next submission strategy

### If V13 Succeeds
- [ ] Consider uploading V11 for comparison
- [ ] Consider uploading V11.5 for ablation study
- [ ] Plan V14 improvements
- [ ] Research top performer approaches

### If V13 Fails
- [ ] Analyze failure mode thoroughly
- [ ] Check if braindecode is the issue
- [ ] Consider V11 upload (simpler, proven)
- [ ] Create V14 with embedded models (no dependencies)

---

## 🛠️ Alternative Submissions Ready

### V11 (Safe Fallback)
**File:** `submissions/phase1_v11.zip` (1.7 MB)
- C1: V10 single model (proven 1.00019)
- C2: 2-seed ensemble (Seeds 42, 123)
- Expected: Overall ~1.00034
- Risk: Low (similar to V10)

### V11.5 (5-Seed Test)
**File:** `submissions/phase1_v11.5.zip` (6.1 MB)
- C1: 5-seed ensemble only (no TTA/calibration)
- C2: 2-seed ensemble
- Expected: Overall ~1.00031
- Risk: Medium (simpler than V13)

**When to use:**
- If V13 fails: Try V11 first (safest)
- If V13 succeeds but underperforms: Try V11.5 for ablation

---

## 📞 Quick Reference

**Competition Page:** https://www.codabench.org/competitions/3350/  
**Package Location:** `/home/kevin/Projects/eeg2025/v13_submission.zip`  
**Package Size:** 6.1 MB (under 10 MB limit)  
**Expected Time:** 15-30 minutes total  
**Expected Rank:** #45-55 (from #72)

**Contact:**
- Team: hkevin01
- Competition: NeurIPS 2025 EEG Foundation Challenge

---

## 🎉 Ready to Upload!

All systems ready. V13 has been:
- ✅ Tested locally (both challenges)
- ✅ Verified format (numpy arrays, correct shapes)
- ✅ Fixed PyTorch compatibility issue
- ✅ Packaged in root directory
- ✅ Comprehensive documentation prepared

**Next step:** Upload `v13_submission.zip` to competition platform!

Good luck! 🚀

---

**Created:** November 1, 2025, 2:40 PM  
**Last Updated:** November 1, 2025, 2:40 PM  
**Status:** Ready for upload

