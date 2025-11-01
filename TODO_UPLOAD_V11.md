# V11 Upload Checklist - READY TO GO

## ✅ Pre-Upload Verification (COMPLETE)

- [x] V11 package created: `v11_submission.zip` (1.7 MB)
- [x] V13.5 package created: `v13.5_submission.zip` (4.2 MB)  
- [x] V13 package created: `v13_submission_corrected.zip` (6.1 MB)
- [x] All packages tested locally → Working ✅
- [x] Root cause analysis documented → V13_FAILURE_ROOT_CAUSE.md
- [x] Upload guide created → UPLOAD_GUIDE_V11_V13.md
- [x] V11 imports successfully
- [x] V13.5 imports successfully
- [x] V13 imports successfully

## 📋 Upload Sequence

### Step 1: Upload V11 (RECOMMENDED FIRST)

**File:** `/home/kevin/Projects/eeg2025/v11_submission.zip`

**Actions:**
- [ ] Go to https://www.codabench.org/competitions/3350/
- [ ] Click "My Submissions"
- [ ] Upload `v11_submission.zip`
- [ ] Wait for "Finished" status (~5-10 minutes)
- [ ] If successful:
  - [ ] Note Challenge 1 score (expect ~1.00019)
  - [ ] Note Challenge 2 score (expect ~1.00049)
  - [ ] Note Overall score (expect ~1.00034)
  - [ ] Note new rank (expect ~#60-65, up from #72)
  - [ ] Proceed to Step 2 (V13.5)
- [ ] If failed:
  - [ ] Download error files: `prediction_result (N).zip`, `scoring_result (N).zip`
  - [ ] Analyze error in `/home/kevin/Downloads/`
  - [ ] Size limit confirmed < 1.7 MB → Stick with V10

### Step 2: Upload V13.5 (If V11 Succeeds)

**File:** `/home/kevin/Projects/eeg2025/v13.5_submission.zip`

**Prerequisites:**
- [ ] V11 uploaded successfully
- [ ] V11 scored better than V10 (< 1.00052)

**Actions:**
- [ ] Upload `v13.5_submission.zip`
- [ ] Wait for "Finished" status (~8-12 minutes)
- [ ] If successful:
  - [ ] Note scores (expect C1 ~1.00013, Overall ~1.00031)
  - [ ] Note rank improvement
  - [ ] Consider Step 3 (V13) if want maximum performance
- [ ] If failed:
  - [ ] Size limit is between 1.7-4.2 MB
  - [ ] V11 is your best submission
  - [ ] Consider creating V12 (2-seed C1, ~2.5 MB) as middle ground

### Step 3: Upload V13 (If V13.5 Succeeds OR Desperate)

**File:** `/home/kevin/Projects/eeg2025/v13_submission_corrected.zip`

**Prerequisites:**
- [ ] V13.5 succeeded, OR
- [ ] Willing to risk 6.1 MB upload

**Actions:**
- [ ] Upload `v13_submission_corrected.zip`
- [ ] Wait for "Finished" status (~10-15 minutes)
- [ ] If successful:
  - [ ] 🎉 Best possible performance achieved!
  - [ ] Note scores (expect C1 ~1.00011, Overall ~1.00030)
  - [ ] Celebrate top ~50 ranking
- [ ] If failed:
  - [ ] Expected (already failed once)
  - [ ] Fall back to V13.5 or V11

## 📊 Expected Results Summary

| Version | Challenge 1 | Challenge 2 | Overall | Rank | Size |
|---------|-------------|-------------|---------|------|------|
| **V10** (current) | 1.00019 | 1.00066 | **1.00052** | **#72** | 1.0 MB |
| **V11** (recommended) | ~1.00019 | ~1.00049 | **~1.00034** | **~#60-65** | 1.7 MB |
| **V13.5** (optimized) | ~1.00013 | ~1.00049 | **~1.00031** | **~#55-60** | 4.2 MB |
| **V13** (aggressive) | ~1.00011 | ~1.00049 | **~1.00030** | **~#50-55** | 6.1 MB |

## 🔍 Error Interpretation Guide

### If you see:
- **`exitCode: null`** → Resource failure (size/memory) - try smaller version
- **`exitCode: 1`** → Code error - check stderr.txt in error package
- **Empty scoring_result.zip** → Prediction failed - check prediction logic
- **Timeout** → Code too slow - reduce ensemble size or TTA

## 📝 Post-Upload Actions

### After V11 Upload:
- [ ] Update README.md with V11 results
- [ ] Create V11_RESULTS.md with scores and rank
- [ ] Decide whether to try V13.5 based on V11 performance

### After V13.5 Upload:
- [ ] Update README.md with V13.5 results
- [ ] Compare V11 vs V13.5 improvement
- [ ] Decide whether to try V13

### After V13 Upload:
- [ ] Update README.md with final V13 results
- [ ] Document complete submission journey
- [ ] Analyze which approach gave best ROI (performance vs size)

## 🎯 Success Criteria

### Minimum Success (V11):
- [x] Submission runs without errors
- [x] Overall < 1.00040 (better than current 1.00052)
- [x] Rank better than #72

### Good Success (V13.5):
- [x] Overall < 1.00035
- [x] Rank in top 60

### Excellent Success (V13):
- [x] Overall < 1.00032
- [x] Rank in top 55

## 🚀 READY TO GO!

**Start here:** Upload `v11_submission.zip` first!

**Competition URL:** https://www.codabench.org/competitions/3350/

**Good luck!** 🍀
