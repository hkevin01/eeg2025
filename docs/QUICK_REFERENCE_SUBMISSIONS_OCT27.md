# 🚀 Quick Reference - EEG2025 Competition

## 📦 Ready-to-Upload Submissions

### Priority 1: Best Single Model (RECOMMENDED)
```bash
submissions/v9_ensemble/submission_v9_single_best.zip
```
- Size: 0.98 MB
- Model: CompactCNN seed 456
- Validation: Pearson r = 0.0211, NRMSE = 0.1607
- Expected score: ~1.0-1.1

### Priority 2: Ensemble
```bash
submissions/v9_ensemble/submission_v9_ensemble_final.zip
```
- Size: 1.6 MB
- Model: 3× CompactCNN ensemble
- Validation: Pearson r = 0.0179, NRMSE = 0.1608
- Expected score: ~1.0-1.2

### Priority 3: TCN Baseline
```bash
submissions/v8_tcn/submission_tcn_v8.zip
```
- Size: 2.9 MB
- Model: TCN architecture
- Purpose: Architecture comparison

---

## 📂 Key Directories

```bash
docs/status_reports/      # Progress reports & checklists
docs/analysis/            # Analysis documents
submissions/v9_ensemble/  # Latest submissions ⭐
submissions/v8_tcn/       # TCN baseline
weights/ensemble/         # Ensemble weights
weights/single_models/    # Individual model weights
checkpoints/compact_ensemble/  # Training checkpoints
training/                 # Training scripts
scripts/submission_builders/   # Build scripts
```

---

## 🔍 Quick Commands

### View Latest Submissions
```bash
ls -lh submissions/v9_ensemble/
ls -lh submissions/v8_tcn/
```

### Check Training Results
```bash
cat checkpoints/compact_ensemble/training_summary.json
```

### View Status Reports
```bash
ls docs/status_reports/
cat docs/status_reports/FINAL_SUBMISSION_CHECKLIST.md
```

### Check Cleanup Summary
```bash
cat docs/CLEANUP_SUMMARY_OCT27.md
```

---

## 📊 Training Results

| Model | Seed | Validation r | NRMSE | Status |
|-------|------|--------------|-------|--------|
| Model 1 | 42 | 0.0178 | 0.1611 | ✓ |
| Model 2 | 123 | 0.0172 | 0.1610 | ✓ |
| **Model 3** | **456** | **0.0211** | **0.1607** | **BEST** ⭐ |
| Ensemble | avg | 0.0179 | 0.1608 | Worse |

---

## 🎯 Next Actions

- [ ] Upload submission_v9_single_best.zip
- [ ] Upload submission_v9_ensemble_final.zip
- [ ] Upload submission_tcn_v8.zip
- [ ] Record scores
- [ ] Compare results
- [ ] Update strategy

---

## 📝 Documentation

- **Main README**: `README.md`
- **Cleanup Summary**: `docs/CLEANUP_SUMMARY_OCT27.md`
- **Training Complete**: `docs/status_reports/ENSEMBLE_TRAINING_COMPLETE.md`
- **Submission Checklist**: `docs/status_reports/FINAL_SUBMISSION_CHECKLIST.md`
- **Analysis**: `docs/analysis/`

---

**Last Updated**: October 27, 2025
**Status**: ✅ Ready to upload submissions

