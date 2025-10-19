# 📋 TODO LIST - PART 3: Submission

**Date:** October 19, 2025, 6:20 PM EDT  
**Prerequisites:** Part 2 training must be complete  
**Deadline:** November 2, 2025 (13 days remaining)

---

## ⏳ PENDING - Pre-Submission Tasks

### Verify Both Challenges Ready
- [ ] Challenge 1 status check
  ```bash
  ls -lh checkpoints/challenge1_tcn_competition_best.pth
  ls -lh weights_challenge_1.pt
  ```

- [ ] Challenge 2 status check
  ```bash
  ls -lh checkpoints/challenge2_fast_best.pth
  ls -lh weights_challenge_2.pt
  ```

### Test Submission Locally
- [ ] Run test script
  ```bash
  python3 test_submission_verbose.py
  ```

- [ ] Expected output:
  ```
  ✅ Challenge 1: TCN loaded successfully
  ✅ Challenge 2: EEGNeX loaded successfully
  ✅ Challenge 1 prediction shape: (batch, 1)
  ✅ Challenge 2 prediction shape: (batch, 1)
  ✅ submission.py works correctly
  ```

---

## ⏳ PENDING - Repository Organization

### Clean Up Root Directory
- [ ] Move large log files to archive
  ```bash
  mkdir -p archive/logs_oct19
  mv logs/cache_creation.log archive/logs_oct19/ 2>/dev/null
  mv logs/cache_*.log archive/logs_oct19/ 2>/dev/null
  ```

- [ ] Keep essential files only
  ```bash
  # Keep in root:
  # - submission.py
  # - weights_challenge_1.pt
  # - weights_challenge_2.pt
  # - README.md
  # - requirements.txt
  # - setup.py
  ```

### Update Documentation
- [ ] Update README.md with final results
  ```bash
  # Add section:
  # - Challenge 1: Val loss 0.010170
  # - Challenge 2: Val loss [from database]
  # - Infrastructure: HDF5 cache + SQLite tracking
  ```

- [ ] Create final STATUS.md
  ```bash
  cat > STATUS.md << 'STATUS'
  # EEG2025 Foundation Challenge - Final Status
  
  **Date:** October 19, 2025
  **Status:** Ready for submission
  
  ## Challenge 1: Response Time Prediction ✅
  - Model: TCN (196K params)
  - Val loss: 0.010170
  - Status: READY
  
  ## Challenge 2: Externalizing Factor Prediction ✅
  - Model: EEGNeX
  - Val loss: [CHECK DATABASE]
  - Status: READY
  
  ## Infrastructure
  - HDF5 Cache: 50GB (10-15x faster loading)
  - Database: SQLite tracking (7 tables, 2 views)
  - Documentation: Complete
  STATUS
  ```

---

## ⏳ PENDING - Create Submission Package

### Build Submission ZIP
- [ ] Create submission package
  ```bash
  # Single-level ZIP (no folders)
  zip -j submission.zip \
    submission.py \
    weights_challenge_1.pt \
    weights_challenge_2.pt
  ```

- [ ] Verify ZIP contents
  ```bash
  unzip -l submission.zip
  # Should show:
  #   submission.py
  #   weights_challenge_1.pt
  #   weights_challenge_2.pt
  ```

- [ ] Check file sizes
  ```bash
  ls -lh submission.zip
  ls -lh weights_*.pt
  ```

---

## ⏳ PENDING - Final Validation

### Test Unpacked Submission
- [ ] Create test directory
  ```bash
  mkdir -p /tmp/submission_test
  cd /tmp/submission_test
  unzip ~/Projects/eeg2025/submission.zip
  ```

- [ ] Test in isolation
  ```bash
  python3 -c "
  from submission import Submission
  sub = Submission()
  print('✅ Submission class instantiated')
  print(f'Challenge 1 model: {type(sub.model_challenge1).__name__}')
  print(f'Challenge 2 model: {type(sub.model_challenge2).__name__}')
  "
  ```

- [ ] Clean up test
  ```bash
  rm -rf /tmp/submission_test
  cd ~/Projects/eeg2025
  ```

---

## ⏳ PENDING - Upload to Competition

### Submit to Platform
- [ ] Go to competition website
  ```
  https://eeg2025.github.io
  ```

- [ ] Upload submission.zip

- [ ] Wait for validation results

- [ ] Check leaderboard position

---

## ⏳ PENDING - Post-Submission

### Monitor Results
- [ ] Check email for results notification

- [ ] Review leaderboard scores
  - Challenge 1: NRMSE score
  - Challenge 2: L1 loss score

### Backup Everything
- [ ] Create final backup
  ```bash
  tar -czf eeg2025_final_backup_$(date +%Y%m%d).tar.gz \
    --exclude='data/' \
    --exclude='logs/' \
    --exclude='archive/' \
    --exclude='__pycache__' \
    .
  ```

- [ ] Store backup safely
  ```bash
  mv eeg2025_final_backup_*.tar.gz ~/Backups/
  ```

---

## 📊 Submission Checklist

```
Before Upload:
  ✅ Both models trained
  ✅ Both weights files created
  ✅ submission.py tested locally
  ✅ ZIP created correctly
  ✅ ZIP contents verified
  ✅ File sizes reasonable

After Upload:
  ⏳ Validation passed
  ⏳ Leaderboard updated
  ⏳ Results received
  ⏳ Backup created
```

---

## 🎯 Success Criteria

### Challenge 1
- NRMSE < 0.30 (target)
- Expected: 0.10-0.15 (based on val loss 0.010170)

### Challenge 2
- Minimize L1 loss
- Strong cross-subject generalization
- Expected: Competitive score (depends on final training)

---

## 🔗 Related Parts

- See `TODO_PART1_INFRASTRUCTURE.md` for cache status
- See `TODO_PART2_TRAINING.md` for training status

---

## 📁 Essential Files for Submission

```
submission.py              (Main submission file)
weights_challenge_1.pt     (Challenge 1 model weights)
weights_challenge_2.pt     (Challenge 2 model weights)
```

**Total ZIP size expected:** ~500MB - 1GB

---

**Last Updated:** October 19, 2025, 6:20 PM EDT  
**Status:** Waiting for training completion  
**Days Remaining:** 13
