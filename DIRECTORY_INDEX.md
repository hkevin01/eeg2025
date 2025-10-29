# 📂 EEG2025 Directory Index

**Last Updated:** October 29, 2025, 11:00 AM  
**Status:** Root directory cleaned and organized (Oct 29 validation cleanup)

---

## 🎯 Quick Access

### Ready-to-Upload Submissions (Oct 28-29)
```bash
submissions/submission_all_rsets_v1.zip          # LATEST (957 KB) - ALL R-sets training
submissions/submission_cross_rset_v6.zip         # Previous (C1 regression)
submissions/submission_v10_single_FIXED.zip      # Backup
```

### Key Documentation
```bash
docs/validation/VALIDATION_PROBLEM_ANALYSIS.md  # Critical validation discovery
docs/validation/VALIDATION_ACTION_PLAN.md       # Implementation plan
docs/status/SUBMISSION_READY_ALL_RSETS_V1.md    # Latest submission info
docs/QUICK_REFERENCE.md                          # Command reference
```

### Essential Root Files
```bash
./README.md                        # Main project documentation (UPDATED with validation learnings)
./submission.py                    # Working submission script
./setup.py                         # Python package setup
./DIRECTORY_INDEX.md               # This file
```

---

## 📁 Directory Structure

### `/` (Root - Essential Files Only)
```
eeg2025/
├── README.md                       ← Main documentation (updated Oct 29)
├── submission.py                   ← Working submission script
├── setup.py                        ← Python package setup
├── LICENSE                         ← MIT license
├── Makefile                        ← Build automation
├── requirements.txt                ← Dependencies
├── requirements-dev.txt            ← Development dependencies
├── pyproject.toml                  ← Project configuration
└── DIRECTORY_INDEX.md              ← This file
```

### `/docs/` - Documentation
```
docs/
├── status/                         ← Status and analysis documents (NEW)
│   ├── ACTION_PLAN_OCT28.md
│   ├── CROSS_RSET_ANALYSIS_OCT28.md
│   ├── LEADERBOARD_ANALYSIS_OCT28.md
│   ├── STATUS_SUMMARY_OCT28_3PM.md
│   ├── SUBMISSION_READY_ALL_RSETS_V1.md
│   └── [18 status documents total]
├── validation/                     ← Validation analysis documents (NEW)
│   ├── VALIDATION_PROBLEM_ANALYSIS.md    ← Critical discovery
│   ├── VALIDATION_ACTION_PLAN.md
│   ├── VALIDATION_STRATEGY_SUMMARY.md
│   ├── TODO_VALIDATION_IMPROVEMENT.md
│   └── QUICK_VALIDATION_GUIDE.md
├── QUICK_REFERENCE.md              ← Command reference (moved from root)
├── README_UPDATE_SUMMARY.md        ← Recent README changes (moved from root)
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
├── submission/                     ← Submission creation scripts (NEW)
│   ├── create_submission_all_rsets.py    (moved from root)
│   └── submission_quickfix_reference.py  (moved from root)
│
├── monitoring/                     ← Progress monitoring scripts (NEW)
│   └── monitor_training.sh             (moved from root)
│
├── training/                       ← Training launch scripts
│   ├── RUN_TRAINING_OCT28.sh           (moved from root)
│   ├── start_training_tmux.sh
│   ├── launch_training_tmux.sh
│   └── start_training.sh
│
├── preprocessing/                  ← Data preprocessing
│   ├── cache_challenge1_with_subjects.py  (subject-aware validation)
│   └── cache_challenge1_windows.py
│
├── experiments/                    ← Training experiments
│   ├── train_c1_all_rsets.py
│   └── train_c1_subject_aware.py       (to be created)
│
├── setup/                          ← Environment setup scripts
│   ├── setup_cpu_training.sh
│   ├── setup_gpu_training.sh
│   └── [other setup scripts]
│
├── organization/                   ← Cleanup & organization scripts
│   └── organize_root.sh
│
└── testing/                        ← Testing utilities
    ├── evaluate_existing_model.py
    ├── evaluate_simple.py
    ├── final_gpu_validation.py
    └── quick_gpu_status.py
```

### `/submissions/` - Submission Packages
```
submissions/
├── submission_all_rsets_v1.zip     ← LATEST (957 KB) - ALL R-sets training
├── submission_cross_rset_v6.zip    ← Previous (C1 regression)
├── submission_v10_single_FIXED.zip ← Backup
├── submission_all_rsets_v1/        ← Unpacked latest submission
│
├── versions/                       ← Previous versions (archived)
│   ├── submission_sam_fixed_v3.zip
│   ├── submission_sam_fixed_v3.py
│   └── submission_sam_fixed_v4.zip  (failed - fallback issue)
│
└── scripts/                        ← Helper scripts
    └── submission_correct.py        (template for v5)
```

### `/weights/` - Model Weights
```
weights/
├── compact_cnn_all_rsets_state.pt      ← LATEST (304 KB) - ALL R-sets training
├── weights_challenge_1.pt              ← Challenge 1 weights (moved from root)  
├── weights_challenge_2.pt              ← Challenge 2 weights (moved from root)
├── weights_challenge_2_BAD_compactcnn.pt ← Bad weights (moved from root)
└── [other weight files]
```
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

**Training scripts?**
→ `scripts/training/` (all .sh training launch scripts)

**Monitor training?**
→ `scripts/monitoring/monitor_training.sh`

---

## 🗑️ What Got Cleaned Up

### Moved to Organized Locations
- ✅ 24 .md files → `docs/` (categorized)
- ✅ 3 .zip files → `submissions/versions/`
- ✅ 2 submission .py → `submissions/scripts/`
- ✅ 6 test .py → `tests/validation/`
- ✅ 4 utility .py → `scripts/testing/`
- ✅ ALL .sh files → `scripts/training/`, `scripts/monitoring/`, `scripts/setup/`

### Kept in Root (Essential Only)
- ✅ `setup.py` - Package setup
- ✅ `submission.py` - Working submission
- ✅ `submission_sam_fixed_v5.zip` - Latest package
- ✅ `README.md` - Main docs
- ✅ `DIRECTORY_INDEX.md` - File location guide
- ✅ Configuration files (requirements.txt, pyproject.toml, Makefile, LICENSE)

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

