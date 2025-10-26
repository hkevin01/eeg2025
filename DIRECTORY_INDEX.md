# 📂 EEG2025 Directory Index

**Last Updated:** October 26, 2025  
**Status:** Root directory cleaned and organized

---

## 🎯 Quick Access

### Current Work
```bash
./submission_sam_fixed_v5.zip      # Latest submission (READY TO UPLOAD)
./submission.py                    # Active submission script
./README.md                        # Main project documentation
```

### Key Documentation
```bash
docs/analysis/SUBMISSION_V5_ANALYSIS.md    # Why v4 failed, how v5 fixes it
docs/status/TRAINING_STATUS_CURRENT.md     # Current training status
docs/CACHED_DATA_INFO.md                   # H5 cache structure and usage
docs/analysis/VSCODE_CRASH_ANALYSIS.md     # VS Code crash investigation
```

---

## 📁 Directory Structure

### `/` (Root - Essential Files Only)
```
eeg2025/
├── submission_sam_fixed_v5.zip    ← Latest submission (466 KB, ready)
├── submission.py                   ← Working submission script
├── setup.py                        ← Python package setup
├── README.md                       ← Main documentation
├── requirements.txt                ← Dependencies
├── pyproject.toml                  ← Project configuration
└── DIRECTORY_INDEX.md              ← This file
```

### `/docs/` - Documentation
```
docs/
├── status/                         ← Training and progress reports
│   ├── TRAINING_STATUS_CURRENT.md    (Active training info)
│   ├── TRAINING_STATUS.md
│   ├── TRAINING_STOPPED_ANALYSIS.md
│   └── CHALLENGE1_IMPROVEMENT_ACTIVE.md
│
├── analysis/                       ← Technical investigations
│   ├── SUBMISSION_V5_ANALYSIS.md     (v4 failure root cause)
│   ├── SUBMISSION_FIX_REPORT.md
│   ├── VSCODE_CRASH_ANALYSIS.md      (115 extensions issue)
│   ├── EXTENSION_CLEANUP_COMPLETE.md
│   ├── gpu_issue_explanation.md
│   ├── BEFORE_AFTER_COMPARISON.md
│   ├── ROCM_GPU_SOLUTION_COMPLETE.md
│   ├── ROCM_SOLUTION_FINAL_STATUS.md
│   ├── FINAL_ROCM_SOLUTION_PLAN.md
│   └── FINAL_STATUS_REALISTIC.md
│
├── submissions/                    ← Submission guides
│   ├── SUBMISSION_READY_V4.md
│   ├── SUBMISSION_UPLOAD_CHECKLIST.md
│   └── CHECKLIST_COMPLETE.md
│
└── CACHED_DATA_INFO.md             ← H5 data structure reference
```

### `/submissions/` - Submission Packages
```
submissions/
├── versions/                       ← Previous versions (archived)
│   ├── submission_sam_fixed_v3.zip
│   ├── submission_sam_fixed_v3.py
│   └── submission_sam_fixed_v4.zip  (failed - fallback issue)
│
└── scripts/                        ← Helper scripts
    └── submission_correct.py        (template for v5)
```

### `/tests/` - Test Scripts
```
tests/
├── validation/                     ← Validation tests
│   ├── test_submission_syntax.py
│   ├── test_conv_*.py             (GPU convolution tests)
│   ├── test_cpu_conv.py
│   └── test_simple_conv.py
│
└── [other test directories]
```

### `/scripts/` - Utility Scripts
```
scripts/
├── testing/                        ← Testing utilities
│   ├── evaluate_existing_model.py
│   ├── evaluate_simple.py
│   ├── final_gpu_validation.py
│   └── quick_gpu_status.py
│
├── monitor_training.sh             ← Training monitor
└── [other scripts]
```

### `/training/` - Training Scripts
```
training/
├── train_c1_cached.py              ← Challenge 1 (uses cached H5 data)
├── train_c1_improved_fast.py
├── train_c1_improved_v2.py
└── [other training scripts]
```

### `/checkpoints/` - Model Weights
```
checkpoints/
├── c1_improved_best.pt             ← Auto-saved (training in progress)
├── challenge1_tcn_competition_best.pth
└── [other checkpoints]
```

### `/data/` - Datasets
```
data/
├── cached/                         ← Pre-processed H5 files
│   ├── challenge1_R1_windows.h5      (660 MB, 7,276 windows)
│   ├── challenge1_R2_windows.h5      (682 MB, 7,524 windows)
│   ├── challenge1_R3_windows.h5      (853 MB, 9,551 windows)
│   └── challenge1_R4_windows.h5      (1.5 GB, 16,554 windows)
│
├── ds005506-bdf/                   ← Raw BDF data (R1)
├── ds005507-bdf/                   ← Raw BDF data (R2)
└── [other data directories]
```

### `/logs/` - Training Logs
```
logs/
├── c1_cached_training.log          ← Current training (PID 1847269)
├── c1_improved_fast.log
└── [other logs]
```

---

## 📊 File Categories

### Submission Files
| File | Location | Status | Purpose |
|------|----------|--------|---------|
| `submission_sam_fixed_v5.zip` | `/` | ✅ Ready | Latest (upload this) |
| `submission_sam_fixed_v4.zip` | `submissions/versions/` | ❌ Failed | Archived (fallback issue) |
| `submission_sam_fixed_v3.zip` | `submissions/versions/` | ❌ Old | Archived |
| `submission.py` | `/` | 🔄 Working | Active script |
| `submission_correct.py` | `submissions/scripts/` | 📝 Template | Source for v5 |

### Documentation Files
| File | Location | Type | Topic |
|------|----------|------|-------|
| `SUBMISSION_V5_ANALYSIS.md` | `docs/analysis/` | Analysis | v4 failure & v5 fix |
| `TRAINING_STATUS_CURRENT.md` | `docs/status/` | Status | Active training |
| `CACHED_DATA_INFO.md` | `docs/` | Reference | H5 structure |
| `VSCODE_CRASH_ANALYSIS.md` | `docs/analysis/` | Analysis | 115 extensions issue |
| `SUBMISSION_READY_V4.md` | `docs/submissions/` | Guide | Upload instructions |

### Test Scripts
| File | Location | Purpose |
|------|----------|---------|
| `test_submission_syntax.py` | `tests/validation/` | Submission validation |
| `test_conv_*.py` | `tests/validation/` | GPU convolution tests |
| `evaluate_existing_model.py` | `scripts/testing/` | Model evaluation |

---

## 🔍 Finding Things

### "Where is...?"

**Latest submission?**
→ `./submission_sam_fixed_v5.zip` (root directory)

**Training status?**
→ `docs/status/TRAINING_STATUS_CURRENT.md`

**Why did v4 fail?**
→ `docs/analysis/SUBMISSION_V5_ANALYSIS.md`

**Cached data structure?**
→ `docs/CACHED_DATA_INFO.md`

**Old submissions?**
→ `submissions/versions/`

**Training script using cached data?**
→ `training/train_c1_cached.py`

**VS Code crash info?**
→ `docs/analysis/VSCODE_CRASH_ANALYSIS.md`

**Current training log?**
→ `logs/c1_cached_training.log`

**Best model checkpoint?**
→ `checkpoints/c1_improved_best.pt` (auto-saved during training)

---

## 🗑️ What Got Cleaned Up

### Moved to Organized Locations
- ✅ 24 .md files → `docs/` (categorized)
- ✅ 3 .zip files → `submissions/versions/`
- ✅ 2 submission .py → `submissions/scripts/`
- ✅ 6 test .py → `tests/validation/`
- ✅ 4 utility .py → `scripts/testing/`

### Kept in Root (Essential Only)
- ✅ `setup.py` - Package setup
- ✅ `submission.py` - Working submission
- ✅ `submission_sam_fixed_v5.zip` - Latest package
- ✅ `README.md` - Main docs
- ✅ Configuration files (requirements.txt, pyproject.toml, etc.)

---

## 📝 Maintenance

### When Adding New Files

**Status/Progress Reports:**
→ Add to `docs/status/`

**Analysis/Investigation Docs:**
→ Add to `docs/analysis/`

**Submission Packages:**
→ Add to `submissions/versions/` (keep latest in root)

**Test Scripts:**
→ Add to `tests/validation/`

**Utility Scripts:**
→ Add to `scripts/testing/` or appropriate subfolder

### Keep Root Clean
- Only essential files in root
- Move old versions to subfolders
- Archive completed work
- Update this index when structure changes

---

*Last organized: October 26, 2025*  
*Root directory: Clean and maintainable*

