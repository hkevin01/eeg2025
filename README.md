# 🧠 NeurIPS 2025 EEG Foundation Challenge

**Competition:** [EEG Foundation Challenge](https://www.codabench.org/competitions/3350/)  
**Team:** hkevin01  
**Current Status:** V12 Ready for Upload  
**Best Verified Score:** V10 - Overall 1.00052, Rank #72/150

---

## 📊 Competition Tasks

- **Challenge 1 (CCD):** Predict response time from EEG
  - Input: 129 channels, 200 timepoints (100 Hz, 2 seconds)
  - Output: Single response time per trial
  
- **Challenge 2 (RSVP):** Predict externalizing factor from EEG
  - Input: 129 channels, 200 timepoints (100 Hz, 2 seconds)
  - Output: Single externalizing score per trial

- **Metric:** NRMSE (Normalized Root Mean Square Error, lower is better)
- **Key Insight:** NRMSE normalized to 1.0 baseline (scores ≥ 1.0)

---

## 🎯 Verified Results

### Submission History
| Version | C1 Score | C2 Score | Overall | Rank | Status |
|---------|----------|----------|---------|------|--------|
| V9 | 1.00077 | 1.00870 | 1.00648 | #88 | ✅ Verified |
| V10 | 1.00019 | 1.00066 | 1.00052 | #72 | ✅ Verified |
| V11 | TBD | TBD | TBD | TBD | 📦 Ready |
| V11.5 | TBD | TBD | TBD | TBD | 📦 Ready |
| V12 | TBD | TBD | TBD | TBD | 📦 Ready |

### Key Milestones
- **V10 Achievement:** C1 only 1.9e-4 above baseline (1.00019)
- **Challenge:** At this margin, variance reduction > architecture changes
- **Strategy Shift:** Focus on ensemble + TTA + calibration

---

## 🚀 Ready for Upload

### V12 - Full Variance Reduction Stack
**Challenge 1: EnhancedCompactCNN**
- 5-seed ensemble (Seeds: 42, 123, 456, 789, 1337)
- Test-Time Augmentation: 3 time shifts (-2, 0, +2)
- Linear calibration (a=0.988, b=0.027)
- Total: 15 predictions per input (5 seeds × 3 TTA)

**Challenge 2: EEGNeX**
- 2-seed ensemble (Seeds: 42, 123)
- EMA weights (decay 0.999)

**Files:**
- `submissions/phase1_v12.zip` (6.1 MB) ✅ Verified
- `submissions/phase1_v11.5.zip` (6.1 MB) ✅ Verified  
- `submissions/phase1_v11.zip` (1.7 MB) ✅ Verified

---

## 💡 Key Lessons Learned

### 1. The 1.9e-4 Problem
- V10 C1 at 1.00019 = only 1.9e-4 above baseline
- At tiny margins: variance reduction > architecture changes
- Focus shifted from model exploration to variance reduction

### 2. Measured Improvements
- **5-seed ensemble:** CV 0.62% (excellent consistency)
- **Linear calibration:** 7.9e-5 improvement on validation (measured!)
- **TTA:** 3 circular time shifts, no bias introduction

### 3. Training Efficiency
- C1 5-seed training: 11.2 minutes (not expected 41 hours!)
- Compact model + HDF5 pipeline = fast iteration
- Lesson: Profile before optimizing assumptions

### 4. Competition Format
Pre-verification caught critical issues:
- Numpy input/output (not torch tensors)
- Constructor: `__init__(SFREQ, DEVICE)`
- Output shapes: `(batch,)` for both challenges
- Proper type conversions

### 5. Power Outage Recovery
- Training interrupted at 88% completion
- Used best 2 of 3 checkpoints (Seeds 42, 123)
- Result: 2 quality seeds > 3 mediocre seeds

---

## 📁 Project Structure

```
eeg2025/
├── submissions/
│   ├── phase1_v12.zip          # V12: Full variance reduction ⭐
│   ├── phase1_v11.5.zip        # V11.5: 5-seed C1 test
│   ├── phase1_v11.zip          # V11: Safe C2 improvement
│   ├── phase1_v10/             # V10: Best verified (Rank #72)
│   └── phase1_v9/              # V9: Baseline (Rank #88)
│
├── checkpoints/
│   ├── c1_phase1_seed*.pt      # 5 C1 models
│   ├── c2_phase2_seed*.pt      # 2 C2 models
│   └── c1_calibration_params.json
│
├── scripts/
│   ├── prepare_c1_data.py      # Data preparation
│   ├── train_c1_phase1_aggressive.py  # 5-seed training
│   ├── c1_calibration.py       # Calibration fitting
│   └── training/               # Training scripts archive
│
├── data/
│   └── processed/
│       └── challenge1_data.h5  # 7,461 CCD segments (679 MB)
│
├── docs/
│   ├── C1_VARIANCE_REDUCTION_PLAN.md
│   ├── V12_VERIFICATION_REPORT.md
│   ├── VARIANCE_REDUCTION_COMPLETE.md
│   ├── SESSION_SUMMARY_NOV1.md
│   ├── archive/                # Historical docs
│   ├── status-reports/         # Status reports
│   └── strategies/             # Strategy documents
│
└── README.md                   # This file
```

---

## 🔬 Technical Details

### EnhancedCompactCNN (Challenge 1)
```python
Input: (batch, 129 channels, 200 timepoints)

Conv1D(129→32, k=7, s=2) + BN + ReLU + Dropout(0.6)
Conv1D(32→64, k=5, s=2) + BN + ReLU + Dropout(0.65)
Conv1D(64→128, k=3, s=2) + BN + ReLU + Dropout(0.7)

Spatial Attention (channel weighting)
AdaptiveAvgPool1D + FC(128→64) + FC(64→1)

Output: (batch, 1) → squeeze → (batch,)
```

**Training:**
- 5 seeds with different initializations
- EMA (decay 0.999) for stability
- Aggressive augmentation (TimeShift, GaussianNoise, ChannelDropout)
- 50 epochs, ~2.2 min/seed on CPU

**Results:**
- Mean NRMSE: 1.499130
- CV: 0.62% (excellent)
- Seeds: 42 (1.486), 123 (1.491), 456 (1.505), 789 (1.511), 1337 (1.502)

### EEGNeX (Challenge 2)
```python
from braindecode.models import EEGNeX

model = EEGNeX(n_chans=129, n_outputs=1, n_times=200, sfreq=100)
```

**Training:**
- 2 seeds (42, 123) with EMA
- 50 epochs, early stopping (patience=10)
- Val loss: Seed 42 (0.122), Seed 123 (0.126)

### Calibration (Challenge 1)
```python
# Fitted on 1,492 validation samples
y_calibrated = 0.988 * y_pred + 0.027

# Ridge regression, alpha=0.1
# Improvement: 7.9e-5 on validation
```

### Test-Time Augmentation
```python
# 3 circular time shifts
shifts = [-2, 0, +2]  # ±20ms at 100Hz

# Total predictions per input
15 = 5 seeds × 3 TTA transforms
```

---

## 🛠️ Setup

### Requirements
```bash
pip install torch torchvision torchaudio
pip install braindecode mne numpy pandas h5py
pip install scikit-learn scipy matplotlib seaborn
```

### Quick Start
```bash
# 1. Prepare data
python scripts/prepare_c1_data.py

# 2. Train models (optional - checkpoints included)
python scripts/train_c1_phase1_aggressive.py

# 3. Fit calibration
python scripts/c1_calibration.py

# 4. Test submission
cd submissions/phase1_v12
python submission.py
```

---

## ✅ Verification

All submissions passed comprehensive testing:
- ✅ Package integrity (ZIP valid)
- ✅ Code structure (required functions present)
- ✅ Input/output format (numpy arrays, correct shapes/dtypes)
- ✅ Batch processing (sizes 1, 5, 16, 32, 64)
- ✅ No NaN/Inf values
- ✅ Model loading (all 7 checkpoints)
- ✅ TTA working (3 shifts, circular padding)
- ✅ Calibration working (linear transform)

See `docs/V12_VERIFICATION_REPORT.md` for details.

---

## 📈 Next Steps

1. **Upload V12** to competition platform
2. **Monitor results** (2-3 hours for evaluation)
3. **Compare submissions:** V12 vs V11.5 vs V11
4. **Analyze improvements:** Which variance reduction techniques helped most
5. **Iterate:** Based on actual leaderboard feedback

---

## 📚 Documentation

**Key Documents:**
- `docs/C1_VARIANCE_REDUCTION_PLAN.md` - Strategy and expected gains
- `docs/V12_VERIFICATION_REPORT.md` - Comprehensive verification
- `docs/VARIANCE_REDUCTION_COMPLETE.md` - Implementation details
- `docs/SESSION_SUMMARY_NOV1.md` - Session summary

**Archives:**
- `docs/archive/` - Historical documentation
- `docs/status-reports/` - Progress reports
- `docs/strategies/` - Strategy evolution

---

## 🙏 Acknowledgments

- NeurIPS 2025 EEG Challenge organizers
- Braindecode team (EEGNeX architecture)
- MNE-Python (EEG preprocessing)
- PyTorch community

---

## 📝 License

MIT License - See LICENSE file

---

**Last Updated:** November 1, 2025, 2:15 PM  
**Status:** V12 Verified and Ready for Upload 🚀  
**Competition:** https://www.codabench.org/competitions/3350/

---
