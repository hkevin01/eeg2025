# Submission v6 Todo List - TCN for Both Challenges

## ✅ COMPLETED

- [x] Install tmux for independent training
- [x] Create independent training launcher scripts
- [x] Train Challenge 1 TCN on competition data
  - [x] Fix window_ind indexing bug
  - [x] Complete training (17 epochs, val loss 0.010170)
  - [x] Save best checkpoint: challenge1_tcn_competition_best.pth
- [x] Fix monitor script to detect all log patterns
- [x] Integrate Challenge 1 TCN into submission.py
  - [x] Replace sparse attention CNN with TCN_EEG
  - [x] Fix TemporalBlock to match trained model (add BatchNorm)
  - [x] Load challenge1_tcn_competition_best.pth weights
  - [x] Test locally - predictions reasonable (1.88-1.97s)
- [x] Create Challenge 2 training script
  - [x] Challenge2Dataset class for RestingState EEG
  - [x] Same TCN architecture as Challenge 1
  - [x] Configuration: R1-R3 train, R4 validate
- [x] Fix dtype mismatch bug (Float64 → Float32)
- [x] Launch Challenge 2 training in tmux
  - [x] Session: eeg_both_challenges
  - [x] Successfully loading data (99K train, 63K val)
  - [x] Training started and progressing

## 🔄 IN PROGRESS

- [ ] Challenge 2 Training Completion
  - Status: Epoch 1/100, Batch ~1900/6192 (30% through epoch 1)
  - Expected: 30-60 minutes total
  - Completion ETA: ~23:00-23:30 tonight
  - Log: logs/train_c2_tcn_20251017_221832.log

## ⏳ PENDING (Tonight/Tomorrow)

- [ ] Review Challenge 2 training results
  - [ ] Check best validation loss
  - [ ] Compare to baseline (0.2917 NRMSE)
  - [ ] Verify early stopping behavior

- [ ] Integrate Challenge 2 TCN into submission.py
  - [ ] Replace CompactExternalizingCNN with TCN_EEG
  - [ ] Load challenge2_tcn_competition_best.pth weights
  - [ ] Update model initialization
  - [ ] Test locally with dummy data

- [ ] Final submission testing
  - [ ] Test Challenge 1 predictions (response times)
  - [ ] Test Challenge 2 predictions (externalizing scores)
  - [ ] Verify both models load correctly
  - [ ] Check prediction ranges are reasonable

- [ ] Package submission v6
  - [ ] Create submission zip with both TCN weights
  - [ ] Verify file size < 50 MB (expected ~5 MB)
  - [ ] Include: submission.py, challenge1_tcn*.pth, challenge2_tcn*.pth
  - [ ] Test zip extraction locally

- [ ] Upload to Codabench
  - [ ] Submit to competition: https://www.codabench.org/competitions/4287/
  - [ ] Wait for validation (1-2 hours)
  - [ ] Check leaderboard score
  - [ ] Compare to previous submissions

## 📊 Expected Results

### Challenge 1: Response Time
- **Old model:** Sparse attention CNN (846K params, NRMSE ~0.28)
- **New model:** TCN_EEG (196K params, val loss 0.010170)
- **Expected:** NRMSE ~0.10 (65% improvement)

### Challenge 2: Externalizing
- **Baseline:** 0.2917 NRMSE
- **New model:** TCN_EEG (196K params, training now)
- **Target:** NRMSE 0.15-0.20 (30-50% improvement)

### Overall
- **Current rank:** Unknown (haven't submitted v6 yet)
- **Target rank:** Top 3
- **Strategy:** TCN architecture for both challenges

## 🔧 Next Actions (Immediate)

1. **Monitor training** - Check progress every 10-15 minutes
2. **Wait for completion** - Training needs 30-60 min total
3. **Prepare integration** - Have submission.py update ready
4. **Test thoroughly** - Verify both models before upload

## 📝 Commands for Next Steps

```bash
# Monitor training progress
tail -f logs/train_c2_tcn_20251017_221832.log

# Check if training completed
ls -lh checkpoints/challenge2_tcn_competition_best.pth

# When complete, integrate into submission
# (Update submission.py to use Challenge 2 TCN)

# Test locally
python3 submission.py

# Package for upload
zip eeg2025_submission_v6.zip \
    submission.py \
    challenge1_tcn_competition_best.pth \
    challenge2_tcn_competition_best.pth

# Verify size
ls -lh eeg2025_submission_v6.zip
```

## 🎯 Success Criteria

- ✅ Challenge 1: Val loss 0.010170 achieved
- 🔄 Challenge 2: Training in progress
- ⏳ Challenge 2: Val loss < 0.20 target
- ⏳ Both models integrated and tested
- ⏳ Submission package < 50 MB
- ⏳ Codabench validation passes
- ⏳ Leaderboard improvement visible

