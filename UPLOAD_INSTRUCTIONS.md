# 🚀 EEG Foundation Challenge 2025 - Upload Instructions

**Date**: October 21, 2025  
**Status**: ✅ Ready to upload (with timezone fix)

---

## 📦 Submission Packages Ready

### Primary Package (RECOMMENDED)
**File**: `submission_simple_READY_TO_UPLOAD.zip` (2.4 MB)
- **Location**: `/home/kevin/Projects/eeg2025/submission_simple_READY_TO_UPLOAD.zip`
- **Contents**:
  - `submission.py` - Inference code using braindecode
  - `weights_challenge_1.pt` - TCN weights (epoch 2)
  - `weights_challenge_2.pt` - EEGNeX weights (epoch 1)
  - `localtime` - System timezone file (fixes Codabench Python initialization)
- **Status**: ✅ Tested locally with smoke tests

### Backup Package
**File**: `submission_standalone_BACKUP.zip` (2.4 MB)
- **Location**: `/home/kevin/Projects/eeg2025/submission_standalone_BACKUP.zip`
- **Contents**: Same weights + standalone EEGNeX implementation
- **Use**: If simple version fails due to braindecode issues

---

## 🎯 Upload Steps

### 1. Open Competition Page
https://www.codabench.org/competitions/4287/

### 2. Navigate to Submissions
- Click **"My Submissions"** tab
- Click **"Submit"** button (or **"+ New submission"**)

### 3. Upload Primary Package
- **File**: Select `submission_simple_READY_TO_UPLOAD.zip`
- **Method/Tag**: Leave default or enter "Simple with localtime"
- **Description**: 
  ```
  TCN + EEGNeX submission using braindecode (includes timezone fix)
  - Challenge 1: TCN (196,225 params, epoch 2)
  - Challenge 2: EEGNeX (62,353 params, epoch 1)
  ```
- Click **Submit**

### 4. Monitor Validation
- Wait 10-20 minutes for scoring
- Check for:
  - ✅ "Finished" status
  - ✅ Scores appear in leaderboard
  - ❌ Any error messages in logs

---

## 🔍 What Changed Since Last Upload

### Previous Failure (submission #4)
- ❌ **Error**: `Fatal Python error: init_interp_main: can't initialize time`
- ❌ **Cause**: Missing `/etc/localtime` in container
- ❌ **Result**: Scoring process crashed before model evaluation

### Current Fix
- ✅ **Solution**: Bundled `/etc/localtime` (3.5 KB) into submission ZIP
- ✅ **Testing**: Confirmed weight loading and inference still work
- ✅ **Impact**: Python initialization should succeed on Codabench

---

## 📊 Model Details

### Challenge 1: CCD Response Time Prediction
- **Architecture**: Temporal Convolutional Network (TCN)
- **Parameters**: 196,225
- **Training**: HBN R1 dataset
- **Best Epoch**: 2
- **Validation Loss**: 0.010170
- **Input**: (batch, 129 channels, 200 timepoints)
- **Output**: Response time prediction (seconds)

### Challenge 2: Externalizing Factor Prediction
- **Architecture**: EEGNeX (via braindecode)
- **Parameters**: 62,353
- **Training**: HBN R1+R2 combined (129,655 samples)
- **Best Epoch**: 1
- **Validation Loss**: 0.000084
- **Input**: (batch, 129 channels, 200 timepoints)
- **Output**: Externalizing factor score

---

## ⚠️ If Upload Fails Again

### Check Error Logs
1. Download `scoring_result.zip` from submission page
2. Extract and check `stdout.txt` and `stderr.txt`
3. Look for specific Python errors

### Common Issues & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: braindecode` | Platform missing library | Upload `submission_standalone_BACKUP.zip` |
| `FileNotFoundError: weights` | Path resolution issue | Check `resolve_path()` implementation |
| `Time initialization error` | Still missing timezone | Contact competition organizers |
| `CUDA/GPU errors` | Wrong device code | Already fixed (CPU-only) |

### Backup Plan
If simple version continues to fail:
1. Upload `submission_standalone_BACKUP.zip`
2. Update description: "Standalone implementation (no braindecode)"
3. This version has zero external dependencies

---

## 📁 File Locations Summary

```
/home/kevin/Projects/eeg2025/
├── submission_simple_READY_TO_UPLOAD.zip       ← Upload this first
├── submission_standalone_BACKUP.zip            ← Backup if needed
├── submission.py                               ← Root reference copy
├── submissions/
│   ├── simple/submission.py                   ← Source (simple)
│   ├── standalone/submission.py               ← Source (standalone)
│   └── packages/
│       ├── 2025-10-21_simple_braindecode/
│       │   ├── submission.zip                 ← Stored package
│       │   ├── README.md                      ← Package docs
│       │   └── artifacts/unpacked/            ← Test extraction
│       └── 2025-10-21_standalone/
│           ├── submission.zip                 ← Stored package
│           └── README.md                      ← Package docs
└── docs/reports/root_reports/
    ├── SUBMISSION_COMPARISON.md               ← Version comparison
    └── UPLOAD_NOW.md                          ← Previous upload guide
```

---

## ✅ Pre-Upload Checklist

- [x] Submission ZIP includes all required files (submission.py, weights, localtime)
- [x] Tested locally with `/tmp/test_simple_tz` smoke test
- [x] Both Challenge 1 and Challenge 2 models load correctly
- [x] Forward passes produce expected output shapes
- [x] CPU-only execution confirmed (no GPU dependencies)
- [x] Backup standalone package available
- [x] Documentation updated

---

## 🎯 Expected Outcome

**If Successful**:
- Validation completes in ~15 minutes
- Two scores appear on leaderboard:
  - Challenge 1: Correlation coefficient (higher is better)
  - Challenge 2: Correlation coefficient (higher is better)
- Overall rank updates based on combined score

**Success Criteria**:
- ✅ No Python initialization errors
- ✅ Both models load weights successfully
- ✅ Predictions generated for all test samples
- ✅ Scores computed and submitted to leaderboard

---

## 📞 Support

If issues persist after uploading both versions:
- Check competition forum: https://www.codabench.org/forums/
- Review starter kit: `/home/kevin/Projects/eeg2025/starter_kit_integration/`
- Compare with reference submission in starter kit

---

**Good luck! 🍀 Upload and monitor the validation logs.**
