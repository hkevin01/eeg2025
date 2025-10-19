# 📋 TODO LIST - PART 2: Training

**Date:** October 19, 2025, 6:20 PM EDT  
**Prerequisites:** Part 1 cache creation must be complete

---

## ⏳ PENDING - Training Setup

### Start Training in tmux
- [ ] Wait for cache creation to complete (see Part 1)
- [ ] Start training session
  ```bash
  tmux new -s training -d "cd /home/kevin/Projects/eeg2025 && \
    python3 train_challenge2_fast.py 2>&1 | tee logs/training_fast.log"
  ```

- [ ] Verify training started
  ```bash
  tmux ls  # Should show 'training' session
  tail -20 logs/training_fast.log
  ```

---

## ⏳ PENDING - Monitor Training

### Check Training Progress
- [ ] Attach to tmux session to watch live
  ```bash
  tmux attach -t training
  # Detach: Ctrl+B then D
  ```

- [ ] Check database for metrics
  ```bash
  # Latest training run
  sqlite3 data/metadata.db \
    'SELECT * FROM training_runs ORDER BY run_id DESC LIMIT 1;'
  
  # Epoch progress
  sqlite3 data/metadata.db \
    'SELECT epoch, train_loss, val_loss FROM epoch_history WHERE run_id=1;'
  
  # Best model
  sqlite3 data/metadata.db 'SELECT * FROM best_models;'
  ```

- [ ] Monitor log file
  ```bash
  tail -f logs/training_fast.log
  ```

---

## ⏳ PENDING - Training Completion

### Expected Training Behavior
- **Batch size:** 64
- **Max epochs:** 20
- **Early stopping:** Patience 5
- **Expected completion:** 5-10 epochs
- **Data loading:** ~10 seconds (vs 15-30 min before)
- **Time per epoch:** ~10-15 minutes

### Checkpoints
- [ ] Verify checkpoints being saved
  ```bash
  ls -lht checkpoints/challenge2_fast_*.pth | head -5
  ```

- [ ] Check best model saved
  ```bash
  ls -lh checkpoints/challenge2_fast_best.pth
  ```

---

## ⏳ PENDING - After Training Completes

### Verify Training Results
- [ ] Check final metrics in database
  ```bash
  sqlite3 data/metadata.db \
    "SELECT run_id, model_name, best_val_loss, total_epochs, status \
     FROM training_runs WHERE challenge=2 ORDER BY run_id DESC LIMIT 1;"
  ```

- [ ] Review best epoch
  ```bash
  sqlite3 data/metadata.db \
    "SELECT epoch, train_loss, val_loss \
     FROM epoch_history WHERE run_id=1 \
     ORDER BY val_loss ASC LIMIT 1;"
  ```

### Copy Weights for Submission
- [ ] Copy best weights to submission location
  ```bash
  cp checkpoints/challenge2_fast_best.pth weights_challenge_2.pt
  ls -lh weights_challenge_2.pt
  ```

- [ ] Verify weights loadable
  ```bash
  python3 -c "
  import torch
  weights = torch.load('weights_challenge_2.pt', map_location='cpu')
  print('✅ Weights loaded successfully')
  print(f'Keys: {list(weights.keys())[:5]}...')
  "
  ```

---

## 📊 Training Status Tracking

```
Stage 1: Data Loading
  └─ Expected: <10 seconds (with HDF5 cache)
  └─ Before: 15-30 minutes (without cache)

Stage 2: Training Loop
  └─ Epochs: 5-10 expected (early stopping)
  └─ Time/epoch: ~10-15 min
  └─ Total: 1-2 hours

Stage 3: Completion
  └─ Best model saved automatically
  └─ Metrics logged to database
  └─ Ready for submission
```

---

## 🔗 Related Parts

- See `TODO_PART1_INFRASTRUCTURE.md` for cache creation status
- See `TODO_PART3_SUBMISSION.md` for submission steps

---

## ⚠️ If Training Fails

### Check Errors
```bash
# Last 50 lines of log
tail -50 logs/training_fast.log

# Check for Python errors
grep -i "error\|exception\|traceback" logs/training_fast.log

# Check database for failed runs
sqlite3 data/metadata.db \
  "SELECT * FROM training_runs WHERE status='failed';"
```

### Restart Training
```bash
# Kill existing if needed
tmux kill-session -t training 2>/dev/null

# Start fresh
tmux new -s training -d \
  "python3 train_challenge2_fast.py 2>&1 | tee logs/training_fast_retry.log"
```

---

**Last Updated:** October 19, 2025, 6:20 PM EDT  
**Status:** Waiting for cache completion
