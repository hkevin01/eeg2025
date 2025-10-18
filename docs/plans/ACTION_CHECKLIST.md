# ✅ Action Checklist - Competition Training & Submission

**Date:** October 17, 2025  
**Deadline:** November 2, 2025 (16 days)

---

## 🔥 CURRENTLY RUNNING

```markdown
- [x] Set up crash-proof training system
- [x] Create train_tcn_competition_data.py (official data loader)
- [x] Start training on R1, R2, R3 (competition releases)
- [ ] Wait for training to complete (~2 hours, by 8:20 PM)
```

**Status:** ✅ Training actively running (PID: 105017)

---

## 📋 TODO: After Training Completes

### Phase 1: Validation & Integration (Tonight)

```markdown
- [ ] Check training completed successfully
      Command: ./scripts/monitor_training.sh
      
- [ ] Load and inspect best checkpoint
      File: checkpoints/challenge1_tcn_competition_best.pth
      Check: val_loss, epoch, correlation
      
- [ ] Test checkpoint loads correctly
      Python: torch.load('checkpoints/challenge1_tcn_competition_best.pth')
      
- [ ] Create updated submission.py with TCN model
      Replace: Challenge1Model class
      Load: TCN checkpoint in __init__
      Test: __call__ method returns correct shape
      
- [ ] Test locally with starter kit local_scoring.py
      Verify: Predictions work on sample data
      Check: No errors, reasonable outputs
      
- [ ] Create submission v6 ZIP
      Include: submission.py, TCN checkpoint, Challenge2 weights
      Verify: Under 50 MB size limit
      
- [ ] Upload v6 to Codabench
      URL: https://www.codabench.org/competitions/4287/
      Wait: 1-2 hours for results
      
- [ ] Compare v6 results with current (0.2832 NRMSE)
      Goal: Show improvement (expect 0.21-0.25 range)
```

---

## 📋 TODO: Challenge 2 Training (This Weekend)

### Phase 2: Externalizing Factor Prediction

```markdown
- [ ] Create train_tcn_challenge2.py
      Task: Predict p_factor from EEG
      Data: Multi-task (RestingState, etc.)
      Releases: R2, R3, R4 (train), R5 (test)
      
- [ ] Implement p_factor extraction from metadata
      Field: 'p_factor' in dataset descriptions
      Handle: Missing values, normalization
      
- [ ] Create fixed-length windows (not trial-locked)
      Use: create_fixed_length_windows from braindecode
      Length: 10-30 seconds per window
      
- [ ] Train TCN for regression on p_factor
      Loss: MAE (Mean Absolute Error)
      Metric: Pearson correlation
      
- [ ] Integrate into submission.py
      Update: Challenge2Model class
      Load: Challenge 2 TCN checkpoint
      
- [ ] Create submission v7 with both challenges updated
      Upload: Test combined performance
```

---

## 📋 TODO: Advanced Improvements (Next Week)

### Phase 3: Pre-training & Transfer Learning

```markdown
- [ ] Implement pre-training on passive tasks
      Tasks: RestingState, DespicableMe, ThePresent
      Method: Self-supervised or multi-task
      
- [ ] Fine-tune on active task (contrastChangeDetection)
      Strategy: Progressive unfreezing
      Compare: With vs without pre-training
      
- [ ] Test S4 State Space Model
      Implement: S4_EEG from improvements/all_improvements.py
      Train: On competition data
      Compare: TCN vs S4 performance
      
- [ ] Create ensemble model
      Combine: TCN + S4 + others
      Method: WeightedEnsemble from improvements
      Optimize: Ensemble weights on validation
      
- [ ] Add TTA (Test-Time Augmentation)
      Integrate: With TCN predictions
      Test: Multiple augmentation strategies
```

---

## 📋 TODO: Final Push (Week Before Deadline)

### Phase 4: Optimization & Final Submission

```markdown
- [ ] Hyperparameter optimization
      Tools: Optuna or grid search
      Parameters: Learning rate, dropout, architecture
      
- [ ] Train on full datasets (not limited to 50)
      Remove: max_datasets_per_release limit
      Use: All available competition data
      Time: May take 8-24 hours
      
- [ ] Cross-validation across releases
      Strategy: R1+R2→R3, R1+R3→R2, R2+R3→R1
      Select: Best model from all folds
      
- [ ] Create super-ensemble
      Combine: All best models from different approaches
      Weight: Based on validation performance
      
- [ ] Final submission testing
      Test: Multiple times locally
      Verify: No errors, consistent results
      Document: Model architecture and training
      
- [ ] Submit final version(s)
      Strategy: Submit 2-3 best models
      Compare: Choose best from leaderboard
      Timing: Leave 24h buffer before deadline
```

---

## 🎯 Key Milestones & Deadlines

| Milestone | Target Date | Status |
|-----------|------------|--------|
| TCN Challenge 1 training complete | Oct 17 (tonight) | 🔄 In Progress |
| v6 submission uploaded | Oct 17-18 | ⏳ Next |
| TCN Challenge 2 training | Oct 18-19 | ⏳ Weekend |
| v7 submission (both challenges) | Oct 20 | ⏳ Planned |
| Pre-training experiments | Oct 21-24 | ⏳ Next Week |
| Ensemble models | Oct 25-27 | ⏳ Week 2 |
| Hyperparameter optimization | Oct 28-29 | ⏳ Week 3 |
| Final training on full data | Oct 30-31 | ⏳ Final Week |
| Final submission(s) | Nov 1 | ⏳ Before Deadline |
| **Competition Deadline** | **Nov 2** | 🎯 **Goal** |

---

## 📊 Success Metrics

### Minimum Success (Baseline Improvement)
```markdown
- [ ] v6 submission works without errors
- [ ] Challenge 1 score better than v1 (2.013 NRMSE)
- [ ] Rank improves from #47
```

### Good Success (Top 20)
```markdown
- [ ] Combined NRMSE < 0.26
- [ ] Both challenges show improvement
- [ ] Rank in top 20
```

### Great Success (Top 10)
```markdown
- [ ] Combined NRMSE < 0.22
- [ ] Pre-training provides benefits
- [ ] Rank in top 10
```

### Excellent Success (Top 5)
```markdown
- [ ] Combined NRMSE < 0.18
- [ ] Ensemble outperforms single models
- [ ] Rank in top 5
```

### Perfect Success (Podium)
```markdown
- [ ] Combined NRMSE < 0.16
- [ ] Novel insights documented
- [ ] Rank: Top 3 🏆
```

---

## 🚨 Critical Reminders

### ❌ DON'T
- ❌ Train on R5 (that's the test set!)
- ❌ Use data outside competition releases
- ❌ Exceed 50 MB submission size
- ❌ Skip validation testing before submission
- ❌ Wait until last day to submit

### ✅ DO
- ✅ Use official EEGChallengeDataset loader
- ✅ Follow starter kit preprocessing
- ✅ Train on R1-R3, validate on R4, test via submission
- ✅ Save checkpoints frequently
- ✅ Test locally before uploading
- ✅ Submit multiple versions to compare
- ✅ Leave buffer time before deadline

---

## 📞 Emergency Contacts & Resources

### If Training Fails
```bash
# Check log for errors
cat logs/train_real_20251017_182023.log | tail -100

# Check if process died
ps -p 105017

# Restart if needed
./scripts/train_real_data_robust.sh
```

### Resources
- Competition: https://eeg2025.github.io
- Codabench: https://www.codabench.org/competitions/4287/
- Starter Kit: /home/kevin/Projects/eeg2025/starter_kit_integration/
- EEGDash Docs: https://eeglab.org/EEGDash/overview.html
- Braindecode: https://braindecode.org/

### Documentation
- COMPETITION_FOCUS.md - Full competition details
- TRAINING_STATUS_NOW.md - Current training status
- TCN_TRAINING_COMPLETE.md - Previous TCN results

---

## 🎉 Completion Tracking

### Today (Oct 17)
- [x] ~~Train TCN on synthetic data~~ ✅ Done (97% improvement!)
- [x] ~~Create crash-proof training system~~ ✅ Done
- [x] ~~Start training on competition data~~ 🔄 In Progress
- [ ] Complete first competition training run
- [ ] Create v6 submission

### This Week
- [ ] Upload v6 to Codabench
- [ ] Train Challenge 2 model
- [ ] Create v7 with both challenges
- [ ] Start pre-training experiments

### Next Week
- [ ] Advanced models (S4, ensemble)
- [ ] Hyperparameter optimization
- [ ] Multiple submission testing

### Final Week
- [ ] Full dataset training
- [ ] Final ensemble
- [ ] Submit best version

---

**Current Priority:** ⏳ Wait for training completion (~2 hours)

**Next Action:** Check training results, create v6 submission

**Status:** ✅ On track for competition success!

---

*Last Updated: October 17, 2025, 18:23 EDT*
