# TODO List - NeurIPS 2025 EEG Foundation Challenge

**Last Updated:** October 16, 2025 13:45  
**Competition Deadline:** [Check Codabench for exact date]

---

## 🔥 CRITICAL - Current Training (In Progress)

```
- [x] Fixed Challenge 1 metadata field bug (rt_from_stimulus)
- [x] Fixed Challenge 2 externalizing extraction
- [x] Restarted Challenge 1 v5 training
- [x] Verified Challenge 2 v6 has diverse targets
- [ ] Wait for Epoch 1 completion on Challenge 1 (~30 min)
- [ ] Monitor Challenge 2 training (currently Epoch 3/50)
- [ ] Verify both challenges show valid NRMSE (NOT 0.0)
```

**Expected Time:** 2-3 hours for training to complete

---

## 📊 Phase 1: Multi-Release Training (CURRENT PHASE)

### Training Monitoring
```
- [ ] Check Challenge 1 first epoch NRMSE
  - Expected: 0.5-2.0 range (NOT 0.0)
  - If 0.0: Debug metadata extraction again
  
- [ ] Monitor Challenge 2 convergence
  - Currently: Train 0.8494, Val 20M+ (high)
  - Watch for validation NRMSE to stabilize
  - If stays high: May need different approach
  
- [ ] Watch for early stopping (patience=15 epochs)
  - Both should complete in ~50 epochs or stop early
  
- [ ] Check logs every 30 minutes:
  ```bash
  tail -100 logs/challenge1_training_v5.log | grep -E "Epoch|NRMSE"
  tail -100 logs/challenge2_training_v6.log | grep -E "Epoch|NRMSE"
  ```
```

### After Training Completes (~3-4 hours)
```
- [ ] Check final results:
  ```bash
  grep "Best.*NRMSE" logs/challenge1_training_v5.log | tail -1
  grep "Best.*NRMSE" logs/challenge2_training_v6.log | tail -1
  ```
  
- [ ] Expected results:
  - Challenge 1: ~1.4 NRMSE (goal: < 2.0)
  - Challenge 2: ~0.5 NRMSE (goal: < 1.0)
  - Overall: ~0.8 NRMSE (goal: < 1.0 for competitive)
  
- [ ] If results are good, proceed to submission
- [ ] If results are poor (>2.0), proceed to Phase 2
```

---

## 📦 Phase 1: Submission Package (If Results Good)

### Test Submission Script
```
- [ ] Test submission.py with trained weights:
  ```bash
  source venv/bin/activate
  python3 submission.py --test
  ```
  
- [ ] Verify it loads both models successfully
- [ ] Check that predictions are generated
- [ ] No errors or crashes
```

### Create Submission Package
```
- [ ] Verify all required files exist:
  - [ ] submission.py (updated with Compact models)
  - [ ] weights_challenge_1_multi_release.pt
  - [ ] weights_challenge_2_multi_release.pt
  - [ ] METHODS_DOCUMENT.pdf (needs to be created from .md)
  
- [ ] Convert methods document to PDF:
  ```bash
  # Option 1: Using pandoc
  pandoc METHODS_DOCUMENT.md -o METHODS_DOCUMENT.pdf
  
  # Option 2: Using markdown-pdf (if installed)
  markdown-pdf METHODS_DOCUMENT.md
  
  # Option 3: Use online converter or VS Code extension
  ```
  
- [ ] Create submission zip:
  ```bash
  zip submission_multi_release_v1.zip \
      submission.py \
      weights_challenge_1_multi_release.pt \
      weights_challenge_2_multi_release.pt \
      METHODS_DOCUMENT.pdf
  ```
  
- [ ] Verify zip contents:
  ```bash
  unzip -l submission_multi_release_v1.zip
  ```
```

### Upload Submission
```
- [ ] Go to competition page:
  https://www.codabench.org/competitions/4287/
  
- [ ] Navigate to "My Submissions" tab
- [ ] Click "Submit" button
- [ ] Upload submission_multi_release_v1.zip
- [ ] Add submission description:
  "Multi-release training (R1-R4) with compact CNNs, 
   fixed target extraction bugs, cross-release validation"
  
- [ ] Wait for evaluation results (~1-2 hours)
- [ ] Check leaderboard position
```

---

## 🚀 Phase 2: Advanced Features (If Needed)

**Trigger:** If Phase 1 results are > 2.0 NRMSE overall

### Challenge 1: P300 Component Extraction
```
- [ ] Implement P300 detection:
  - [ ] Extract 250-500ms post-stimulus window
  - [ ] Use peak amplitude as additional feature
  - [ ] Add P300 latency as feature
  
- [ ] Create train_challenge1_p300.py
- [ ] Expected improvement: 1.4 → 1.0 NRMSE
- [ ] Training time: ~3 hours
```

### Challenge 2: Spectral Band Features
```
- [ ] Implement frequency band power extraction:
  - [ ] Delta (0.5-4 Hz)
  - [ ] Theta (4-8 Hz)
  - [ ] Alpha (8-13 Hz)
  - [ ] Beta (13-30 Hz)
  - [ ] Gamma (30-50 Hz)
  
- [ ] Create train_challenge2_spectral.py
- [ ] Use band power ratios as features
- [ ] Expected improvement: 0.5 → 0.35 NRMSE
- [ ] Training time: ~2 hours
```

### Ensemble Methods
```
- [ ] Train multiple models with different initializations
- [ ] Average predictions across models
- [ ] Expected improvement: 5-10% reduction in NRMSE
- [ ] Training time: ~6 hours (3 models × 2 hours)
```

---

## 📝 Documentation Updates

### Before Submission
```
- [x] METHODS_DOCUMENT.md created
- [ ] Convert to PDF
- [x] TRAINING_STATUS.md updated with all bugs
- [ ] Update with final results after training
- [ ] Add training time and convergence info
```

### After Submission
```
- [ ] Update README.md with:
  - [ ] Competition results
  - [ ] Final NRMSE scores
  - [ ] Leaderboard position
  - [ ] Key learnings
  
- [ ] Create FINAL_RESULTS.md with:
  - [ ] Submission scores
  - [ ] Comparison with previous submission
  - [ ] What worked / what didn't
  - [ ] Future improvements
  
- [ ] Clean up old files:
  - [ ] Remove old training logs (v1-v4)
  - [ ] Keep only final versions
  - [ ] Archive experimental code
```

---

## 🔍 Debugging Checklist (If Problems)

### If NRMSE = 0.0 (Constant Prediction)
```
- [ ] Check target extraction in __getitem__
- [ ] Print first 10 targets to verify diversity
- [ ] Verify metadata field names
- [ ] Check for NaN values in targets
- [ ] Ensure targets aren't all the same value
```

### If NRMSE Very High (>10.0)
```
- [ ] Check data normalization
- [ ] Verify input shape is correct
- [ ] Check for NaN or Inf in data
- [ ] Reduce learning rate (1e-3 → 1e-4)
- [ ] Increase regularization (dropout 0.5 → 0.6)
```

### If Training Crashes
```
- [ ] Check memory usage: htop or top
- [ ] Reduce batch size (32 → 16)
- [ ] Reduce num_workers (4 → 2)
- [ ] Check disk space: df -h
- [ ] Review crash logs in logs/challenge*_crash_*.log
```

### If Validation Much Worse Than Training
```
- [ ] Increase dropout (current: 0.3-0.5)
- [ ] Increase weight decay (1e-4 → 1e-3)
- [ ] Reduce model size further
- [ ] Add more data augmentation
- [ ] Use fewer epochs (overfitting)
```

---

## 📋 Competition Rules Compliance

```
- [x] Verified: Can only be part of ONE team
- [ ] Ensure no multiple accounts used
- [ ] Check submission format requirements
- [ ] Verify methods document includes all required sections
- [ ] Ensure code can run on competition environment
```

---

## ⏱️ Time Estimates

| Task | Duration | Dependencies |
|------|----------|--------------|
| **Current Training** | 2-3 hours | None |
| **Test submission.py** | 5 min | Training complete |
| **Create PDF** | 10 min | pandoc installed |
| **Create submission zip** | 2 min | All files ready |
| **Upload & wait for results** | 1-2 hours | Submission ready |
| **Phase 2 (if needed)** | 6-8 hours | Poor Phase 1 results |
| **TOTAL (best case)** | 3-4 hours | Good results first try |
| **TOTAL (with Phase 2)** | 10-12 hours | Need improvements |

---

## 🎯 Success Metrics

### Minimum Acceptable
```
- [ ] Both challenges NRMSE > 0.0 (not constant)
- [ ] Challenge 1 NRMSE < 2.5
- [ ] Challenge 2 NRMSE < 1.5
- [ ] Overall NRMSE < 1.5
- [ ] Better than previous submission (2.01)
```

### Competitive Target
```
- [ ] Challenge 1 NRMSE < 1.5
- [ ] Challenge 2 NRMSE < 0.6
- [ ] Overall NRMSE < 0.8
- [ ] Top 5 leaderboard position
```

### Stretch Goal
```
- [ ] Challenge 1 NRMSE < 1.0
- [ ] Challenge 2 NRMSE < 0.4
- [ ] Overall NRMSE < 0.5
- [ ] Top 3 leaderboard position
```

---

## 📞 Quick Commands Reference

### Monitor Training
```bash
# Check Challenge 1 progress
tail -f logs/challenge1_training_v5.log | grep -E "Epoch|NRMSE"

# Check Challenge 2 progress
tail -f logs/challenge2_training_v6.log | grep -E "Epoch|NRMSE"

# Count active processes
ps aux | grep "[p]ython3 scripts/train" | wc -l

# Check GPU/CPU usage
htop

# Check disk space
df -h
```

### Emergency Stop
```bash
# Kill all training processes
pkill -f "python3 scripts/train"

# Check if stopped
ps aux | grep "[p]ython3 scripts/train"
```

### Quick Test
```bash
# Activate venv and test submission
source venv/bin/activate
python3 submission.py --test

# Check weight files exist
ls -lh weights_challenge_*.pt
```

---

## 🏆 Final Checklist Before Submission

```
- [ ] Training completed successfully
- [ ] Both models saved (weights_*.pt files exist)
- [ ] Final NRMSE < 1.5 overall
- [ ] submission.py tested and working
- [ ] METHODS_DOCUMENT.pdf created
- [ ] All files in submission.zip
- [ ] Zip file < 100 MB (should be ~50MB)
- [ ] Methods document describes actual approach
- [ ] No placeholder text in methods
- [ ] Ready to upload to Codabench
```

---

**Priority:** Focus on Phase 1 completion, then decide on Phase 2 based on results.

**Next Action:** Monitor training progress for next 2-3 hours, then proceed to submission.

**Expected Completion:** October 16, 2025 16:30 (if all goes well)

