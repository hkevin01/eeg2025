# Implementation Plan: Submission v6 - Dual TCN Models

## Overview

**Goal:** Create and submit version 6 with TCN models for both challenges  
**Status:** 🔄 In Progress (Challenge 2 training)  
**Expected Completion:** Tonight or tomorrow morning

## ACID Breakdown

### Atomic Tasks

Each task is the smallest complete unit that can be validated independently.

---

#### ✅ A1: Train Challenge 1 TCN Model
**Status:** COMPLETE  
**Date:** October 17, 2025 18:46

**Steps:**
1. Create training script: `scripts/train_tcn_competition_data.py`
2. Fix window_ind indexing bug
3. Run training: R1-R3 train, R4 validate
4. Monitor progress and save checkpoints

**Results:**
- Training complete: 17 epochs
- Best val loss: 0.010170 (epoch 2)
- Checkpoint: `checkpoints/challenge1_tcn_competition_best.pth`
- Improvement: 65% over baseline

**Validation:**
- Model loads correctly
- Predictions in reasonable range (1.5-2.5 seconds)
- Checkpoint file size: 2.4 MB

---

#### ✅ A2: Integrate Challenge 1 TCN into submission.py
**Status:** COMPLETE  
**Date:** October 17, 2025 19:00

**Steps:**
1. Replace old CNN architecture with TCN_EEG
2. Fix TemporalBlock to match trained model (add BatchNorm)
3. Load challenge1_tcn_competition_best.pth weights
4. Test with dummy data

**Code Changes:**
- Added TemporalBlock class (80 lines)
- Added TCN_EEG class (20 lines)
- Updated Submission.__init__ to use TCN
- Proper weight loading with error handling

**Validation:**
- ✅ Model loads without state_dict errors
- ✅ Predictions: 1.88-1.97 seconds (reasonable)
- ✅ No runtime errors
- ✅ File size acceptable (2.4 MB)

---

#### 🔄 A3: Train Challenge 2 TCN Model
**Status:** IN PROGRESS  
**Started:** October 17, 2025 22:18

**Steps:**
1. ✅ Create Challenge2Dataset class (RestingState EEG)
2. ✅ Create training script: `scripts/train_challenge2_tcn.py`
3. ✅ Fix dtype bug (Float64 → Float32)
4. ✅ Launch training in tmux
5. 🔄 Monitor training progress (epoch 4/100)
6. ⏳ Wait for best validation loss
7. ⏳ Verify checkpoint saved

**Current Progress:**
- Epochs completed: 3/100
- Current epoch: 4
- Best val loss: 0.668 (epoch 2, NRMSE 0.817)
- Training time: ~15 minutes so far
- ETA: 30-60 minutes total

**Validation Criteria:**
- [ ] Training completes or early stops
- [ ] Best val loss < 0.30 (NRMSE better than baseline 0.2917)
- [ ] Checkpoint saved: `checkpoints/challenge2_tcn_competition_best.pth`
- [ ] File size reasonable (~2.4 MB)

**Monitoring:**
```bash
./check_c2_training.sh
tail -f logs/train_c2_tcn_20251017_221832.log
```

---

#### ⏳ A4: Integrate Challenge 2 TCN into submission.py
**Status:** PENDING  
**Depends On:** A3 completion

**Steps:**
1. Replace CompactExternalizingCNN with TCN_EEG
2. Load challenge2_tcn_competition_best.pth weights
3. Update model initialization in Submission class
4. Test with dummy data

**Code Changes Needed:**
```python
# In Submission.__init__
self.model_externalizing = TCN_EEG(
    num_channels=129,
    num_outputs=1,
    num_filters=48,
    kernel_size=7,
    dropout=0.3,
    num_levels=5
).to(self.device)

# Load Challenge 2 weights
challenge2_path = resolve_path("challenge2_tcn_competition_best.pth")
checkpoint = torch.load(challenge2_path, map_location=self.device, weights_only=False)
if 'model_state_dict' in checkpoint:
    self.model_externalizing.load_state_dict(checkpoint['model_state_dict'])
```

**Validation:**
- [ ] Model loads without errors
- [ ] Predictions in reasonable range (0-2 typical externalizing scores)
- [ ] Both models coexist in submission.py
- [ ] No conflicts between Challenge 1 and 2

---

#### ⏳ A5: Test Complete Submission Locally
**Status:** PENDING  
**Depends On:** A4 completion

**Steps:**
1. Run submission.py with dummy data
2. Test Challenge 1 predictions
3. Test Challenge 2 predictions
4. Verify both models work together
5. Check memory usage
6. Verify inference speed

**Test Script:**
```python
submission = Submission()

# Test Challenge 1
dummy_eeg_c1 = torch.randn(10, 129, 200).numpy()
response_times = submission.predict_response_time(dummy_eeg_c1)
print(f"Challenge 1: {response_times.min():.2f} - {response_times.max():.2f} seconds")

# Test Challenge 2
dummy_eeg_c2 = torch.randn(10, 129, 200).numpy()
externalizing = submission.predict_externalizing(dummy_eeg_c2)
print(f"Challenge 2: {externalizing.min():.2f} - {externalizing.max():.2f}")
```

**Validation:**
- [ ] No import errors
- [ ] Both models load correctly
- [ ] Predictions in expected ranges
- [ ] No memory leaks
- [ ] Inference time < 1 second per batch

---

#### ⏳ A6: Package Submission v6
**Status:** PENDING  
**Depends On:** A5 completion

**Steps:**
1. Copy required files to clean directory
2. Create submission zip
3. Verify contents
4. Check file size < 50 MB
5. Test extraction locally

**Commands:**
```bash
# Create clean directory
mkdir -p submission_v6
cd submission_v6

# Copy files
cp ../submission.py .
cp ../challenge1_tcn_competition_best.pth .
cp ../challenge2_tcn_competition_best.pth .

# Create zip
zip -r ../eeg2025_submission_v6.zip .

# Verify
cd ..
ls -lh eeg2025_submission_v6.zip
unzip -l eeg2025_submission_v6.zip
```

**Validation:**
- [ ] submission.py included
- [ ] challenge1_tcn_competition_best.pth included (2.4 MB)
- [ ] challenge2_tcn_competition_best.pth included (2.4 MB)
- [ ] Total size < 50 MB (expected ~5 MB)
- [ ] Zip extracts correctly
- [ ] No extra files included

---

#### ⏳ A7: Upload to Codabench
**Status:** PENDING  
**Depends On:** A6 completion

**Steps:**
1. Login to Codabench
2. Navigate to competition page
3. Upload eeg2025_submission_v6.zip
4. Wait for validation (1-2 hours)
5. Check for errors
6. Monitor leaderboard update

**URL:** https://www.codabench.org/competitions/4287/

**Validation:**
- [ ] Upload successful
- [ ] No validation errors
- [ ] Submission appears in history
- [ ] Score appears on leaderboard
- [ ] Score improves over previous submissions

**Expected Scores:**
- Challenge 1: NRMSE ~0.10 (65% improvement)
- Challenge 2: NRMSE 0.15-0.30 (15-50% improvement)
- Overall: Top 3-5 ranking

---

### Consistent System State

**Pre-conditions:**
- Challenge 1 TCN trained and ready
- Challenge 2 TCN training in progress
- submission.py has Challenge 1 integrated
- All dependencies installed
- Codabench account active

**Post-conditions:**
- Both TCN models integrated in submission.py
- Submission v6 uploaded to Codabench
- Leaderboard updated with new scores
- All checkpoints and logs preserved
- Documentation complete

**Invariants:**
- Model architectures match checkpoint files
- File size remains under 50 MB limit
- Both challenges use TCN_EEG architecture
- Training logs preserved for reproducibility

---

### Isolated Development

Each task can be developed/tested independently:

**A1-A2:** Challenge 1 (already complete, independent)  
**A3-A4:** Challenge 2 (in progress, independent)  
**A5:** Integration testing (depends on A2 + A4)  
**A6-A7:** Packaging and upload (depends on A5)

**Rollback Plan:**
- If Challenge 2 fails: Submit with only Challenge 1 TCN
- If integration fails: Use previous submission.py with new weights
- If upload fails: Fix issues and re-upload
- If leaderboard worse: Analyze and iterate

---

### Durable Changes

**Persistent Artifacts:**
1. **Model Checkpoints:**
   - `checkpoints/challenge1_tcn_competition_best.pth` (2.4 MB)
   - `checkpoints/challenge2_tcn_competition_best.pth` (2.4 MB, when complete)
   - History: `checkpoints/*_history.json`

2. **Training Logs:**
   - `logs/train_fixed_20251017_184601.log` (Challenge 1)
   - `logs/train_c2_tcn_20251017_221832.log` (Challenge 2)

3. **Submission Files:**
   - `submission.py` (updated with TCN)
   - `submission_old_attention.py` (backup)
   - `eeg2025_submission_v6.zip` (final package)

4. **Documentation:**
   - `memory-bank/` (complete project memory)
   - `TRAINING_UPDATE.md`
   - `TODO_SUBMISSION_V6.md`
   - `CHALLENGE2_TRAINING_STATUS.md`

**Git Tracking:**
```bash
git add memory-bank/
git add submission.py
git add scripts/train_challenge2_tcn.py
git commit -m "Submission v6: Dual TCN models"
```

---

## Timeline

**Day 1 (October 17):**
- ✅ 17:30 - Setup independent training
- ✅ 18:30 - Fix window indexing bug
- ✅ 18:46 - Complete Challenge 1 training
- ✅ 19:00 - Integrate Challenge 1 into submission.py
- ✅ 22:18 - Launch Challenge 2 training
- 🔄 22:35 - Challenge 2 epoch 4/100 (current)
- ⏳ 23:00-23:30 - Challenge 2 expected completion

**Day 2 (October 18):**
- ⏳ Morning - Integrate Challenge 2, test submission
- ⏳ Mid-day - Package and upload to Codabench
- ⏳ Afternoon - Monitor leaderboard results

---

## Success Metrics

### Technical Success
- [ ] Challenge 1 TCN: Val loss < 0.015 ✅ (achieved 0.010170)
- [ ] Challenge 2 TCN: Val loss < 0.30
- [ ] Submission size < 50 MB
- [ ] No validation errors on Codabench
- [ ] Both models load and predict correctly

### Performance Success
- [ ] Challenge 1 NRMSE < 0.15 (target: ~0.10)
- [ ] Challenge 2 NRMSE < 0.30 (target: 0.15-0.25)
- [ ] Overall ranking: Top 5
- [ ] Improvement over baseline: > 30% combined

### Process Success
- [x] Independent training survives crashes
- [x] Comprehensive documentation created
- [x] Reproducible training pipeline
- [ ] Clean submission package
- [ ] Timely upload (< 48 hours from training start)

---

## Risk Mitigation

### Risk 1: Challenge 2 Doesn't Improve
**Probability:** Medium  
**Impact:** High  
**Mitigation:**
- Monitor validation loss closely
- If not improving by epoch 10, adjust hyperparameters
- If still poor, submit with Challenge 1 TCN only
- Plan B: Use old Challenge 2 model with Challenge 1 TCN

### Risk 2: Integration Conflicts
**Probability:** Low  
**Impact:** Medium  
**Mitigation:**
- Test each model independently first
- Use same TCN_EEG class for both challenges
- Keep backups of working submission.py versions
- Test thoroughly before packaging

### Risk 3: Codabench Validation Fails
**Probability:** Low  
**Impact:** High  
**Mitigation:**
- Test submission locally first
- Follow exact API requirements
- Check file size and format
- Review previous successful submissions
- Allow time for multiple upload attempts

### Risk 4: Training Takes Too Long
**Probability:** Low (tmux handles this)  
**Impact:** Low  
**Mitigation:**
- Training runs in tmux (independent)
- Can monitor and adjust remotely
- Early stopping prevents excessive training
- Can submit partial results if needed

---

## Completion Checklist

**Before Marking Complete:**
- [ ] All 7 atomic tasks (A1-A7) complete
- [ ] System state consistent (all models integrated)
- [ ] Changes durable (committed to git)
- [ ] Success metrics achieved
- [ ] Documentation updated
- [ ] Memory bank current

**Sign-off Criteria:**
- Submission v6 uploaded to Codabench
- Validation passed
- Leaderboard updated
- Results analyzed and documented

