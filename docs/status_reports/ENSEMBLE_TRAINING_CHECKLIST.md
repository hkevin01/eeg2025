# 🎯 Ensemble Training Checklist

## Quick Reference - Next Actions

### ✅ COMPLETED

- [x] Analyze best submission (quick_fix: 1.01)
- [x] Stop failing SAM training
- [x] Create v8 TCN submission
- [x] Analyze Path B strategy
- [x] Create training script (~500 lines)
- [x] Implement data loading (H5 files)
- [x] Create ensemble submission script
- [x] Test training (1 epoch - passed)
- [x] Fix shape handling
- [x] Document everything

### 🔄 IN PROGRESS / TODO

#### Phase 2: Upload v8 (5 minutes)
- [ ] Upload `submission_tcn_v8.zip` to competition
- [ ] Record v8 score when available
- [ ] Confirm path (A/B/C) based on score

#### Phase 3: Training (6-12 hours) 🎯 **NEXT**
- [ ] Start training: `python training/train_compact_ensemble.py`
- [ ] Monitor progress (3 models × 2-4 hours each)
- [ ] Verify 3 checkpoints created

#### Phase 4: Ensemble Creation (10 minutes)
- [ ] Run: `python create_ensemble_submission.py`
- [ ] Verify `submission_v9_ensemble_final.zip` created
- [ ] Test locally: `python submission.py`

#### Phase 5: Submit & Validate
- [ ] Upload `submission_v9_ensemble_final.zip`
- [ ] Wait for score
- [ ] Compare vs 1.01 (target: 0.75-0.90)

---

## Files Ready to Use

**Training:**
```bash
python training/train_compact_ensemble.py
# Outputs: checkpoints/compact_ensemble/*.pth
```

**After Training:**
```bash
python create_ensemble_submission.py
# Outputs: submission_v9_ensemble_final.zip
```

**Upload:**
```bash
# Upload: submission_tcn_v8.zip (ready now)
# Upload: submission_v9_ensemble_final.zip (after training)
```

---

## Expected Results

| Metric | Current (v5) | Expected (v9) | Improvement |
|--------|-------------|---------------|-------------|
| Score  | 1.01        | 0.75-0.90     | 15-25%      |
| Data   | Cached only | R1+R2+R3      | +3× data    |
| Aug    | None        | Yes           | +5-8%       |
| Ensemble | Single    | 3 models      | +5-7%       |

---

## Quick Commands

```bash
# Test data loading
python -c "import h5py; print('OK')"

# Start training (full)
python training/train_compact_ensemble.py

# Start training (custom)
python training/train_compact_ensemble.py --epochs 30 --batch_size 64

# Create ensemble after training
python create_ensemble_submission.py

# Test submission
python submission.py
```

---

**Status**: ✅ Ready to train
**Next**: `python training/train_compact_ensemble.py`
**Duration**: 6-12 hours
**Expected output**: 3 checkpoints → ensemble → beat 1.01 by 15-25%

