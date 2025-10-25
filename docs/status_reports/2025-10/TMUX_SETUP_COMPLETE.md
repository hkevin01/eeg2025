# ✅ Tmux Setup Complete - Crash-Resistant Training

**Date:** October 24, 2025 17:10 UTC  
**Status:** 🏃 Training running successfully in tmux  
**Problem Solved:** VSCode crashes no longer kill training!

---

## What Was Fixed

### The Problem
- **Issue:** VSCode crashed twice, killing training both times
- **Impact:** Lost training progress, had to restart from scratch
- **Root cause:** Training process tied to VSCode terminal session

### The Solution
- **Technology:** Tmux (terminal multiplexer)
- **Benefit:** Persistent sessions independent of VSCode
- **Result:** Training continues even if VSCode crashes!

---

## Current Status

### ✅ Training Running
```
Session:        eeg_training (tmux)
Started:        17:00 UTC
Loading:        111/150 subjects (74% complete)
GPU Usage:      99% (AMD RX 5600 XT)
Log File:       training_tmux.log
Experiment:     experiments/sam_full_run/20251024_165931/
```

### 📊 Progress
- **Data loading:** In progress (111/150 subjects loaded)
- **Training:** Will start after data loading completes
- **Expected duration:** 2-4 hours total
- **Crash resistance:** ✅ YES! Survives VSCode crashes

---

## Quick Start Guide

### Monitor Training
```bash
./monitor_training.sh          # Quick status
tail -f training_tmux.log      # Live streaming
tmux attach -t eeg_training    # Full access
```

### Detach Safely (Training Keeps Running!)
```
Ctrl+B then D
```

### If VSCode Crashes Again
1. Don't panic! Training is still running ✅
2. Reopen terminal or VSCode
3. Check status: `./monitor_training.sh`
4. Continue monitoring: `tail -f training_tmux.log`

---

## Files Created

### Scripts
1. **`start_training_tmux.sh`** - Start training in tmux session
2. **`monitor_training.sh`** - Check training status and GPU usage

### Documentation
3. **`TMUX_TRAINING_GUIDE.md`** - Complete tmux usage guide
4. **`TMUX_SETUP_COMPLETE.md`** (this file) - Setup summary

### Output
5. **`training_tmux.log`** - Complete training output (live)
6. **`experiments/sam_full_run/20251024_165931/`** - Experiment directory

---

## Key Benefits

### Before (Without Tmux)
- ❌ VSCode crash → Training dies
- ❌ Terminal close → Training dies  
- ❌ SSH disconnect → Training dies
- ❌ No way to detach/reattach
- ❌ Lost progress twice already

### After (With Tmux)
- ✅ VSCode crash → Training continues!
- ✅ Terminal close → Training continues!
- ✅ SSH disconnect → Training continues!
- ✅ Detach/reattach anytime
- ✅ Multiple windows/panes
- ✅ Full control + monitoring

---

## Training Configuration

### Model & Data
- **Architecture:** EEGNeX (62,353 parameters)
- **Datasets:** ds005506-bdf (150 subjects) + ds005507-bdf
- **Task:** Response time prediction (Challenge 1)
- **Format:** 129 channels × 200 timepoints

### Advanced Features
- **SAM Optimizer:** rho=0.05 (flatter minima)
- **Subject-level CV:** GroupKFold (no leakage)
- **Augmentation:** Scaling, dropout, noise
- **Early stopping:** Patience=15 epochs
- **Checkpointing:** Best + epoch saves

### Training Parameters
- **Epochs:** 100 (with early stopping)
- **Batch size:** 32
- **Learning rate:** 1e-3
- **Device:** CUDA (AMD ROCm 6.1.2)
- **Expected time:** 2-4 hours

---

## Monitoring Commands

### Quick Check
```bash
./monitor_training.sh
```
Shows:
- Session status
- Latest 30 lines of output
- GPU usage
- Helpful commands

### Live Streaming
```bash
tail -f training_tmux.log
```
Real-time output as training runs

### Full Interactive
```bash
tmux attach -t eeg_training
```
Full terminal access (detach with Ctrl+B D)

### GPU Monitoring
```bash
watch -n 2 rocm-smi
```
Updates every 2 seconds

---

## Expected Timeline

### Data Loading (Current Phase)
- **Started:** 17:00 UTC
- **Progress:** 111/150 subjects (74%)
- **Expected completion:** ~17:15 UTC (15 minutes)

### Training Phase
- **Start:** After data loading (~17:15 UTC)
- **Duration:** 2-4 hours
- **Completion:** ~19:15 - 21:15 UTC

### Results Available
- **Tonight:** 19:00 - 21:00 UTC
- **Tomorrow:** If running overnight

---

## What to Expect

### During Training
1. Data loading completes (74% done already)
2. Training begins (Epoch 1/100)
3. Progress printed every epoch
4. Best model saved automatically
5. Early stopping may trigger around epoch 40-60

### Output Format
```
Epoch   1/100 | Train Loss: X.XXXX | Val NRMSE: X.XXXX | LR: X.XXe-XX | ✨ BEST!
Epoch   2/100 | Train Loss: X.XXXX | Val NRMSE: X.XXXX | LR: X.XXe-XX |
...
```

### On Completion
```
================================================================================
✅ Training Complete!
================================================================================
   Best Val NRMSE: X.XXXX
   Experiment: experiments/sam_full_run/20251024_165931
   Best model: experiments/sam_full_run/20251024_165931/checkpoints/best_model.pt
```

---

## Success Metrics

### ✅ Setup Complete
- [x] Tmux session created
- [x] Training started successfully
- [x] Data loading in progress (74%)
- [x] GPU at 99% utilization
- [x] Log file being written
- [x] Monitoring scripts working
- [x] Documentation complete

### 🎯 Training Goals
- [ ] Data loading completes (15 min)
- [ ] Training begins and runs smoothly
- [ ] Validation NRMSE < 0.25
- [ ] Early stopping triggers appropriately
- [ ] Best model saved in checkpoints
- [ ] No crashes or interruptions

### 🏆 Ultimate Goals
- [ ] Test NRMSE < 1.0 (beat Oct 24 regression)
- [ ] Test NRMSE < 0.8 (beat Oct 16 baseline)
- [ ] Create submission package
- [ ] Upload to Codabench leaderboard

---

## Comparison: Previous Attempts vs Tmux

| Attempt | Method | Result | Issue |
|---------|--------|--------|-------|
| Oct 24 #1 | Direct terminal | ❌ Died | VSCode crash |
| Oct 24 #2 | Nohup | ❌ Died | VSCode crash |
| Oct 24 #3 | **Tmux** | ✅ **Running!** | Crash-resistant |

---

## Next Steps

### Immediate (While Training Runs)
1. ✅ Training running (no action needed)
2. ⏰ Monitor occasionally: `./monitor_training.sh`
3. ⏰ Check GPU: `watch -n 2 rocm-smi`
4. ⏰ Optional: Attach to watch live: `tmux attach -t eeg_training`

### After Training Completes (2-4 hours)
1. Check results: `grep "Best Val NRMSE" training_tmux.log`
2. Analyze training history: `cat experiments/sam_full_run/*/history.json`
3. If NRMSE < 1.0: Create submission
4. Upload to Codabench
5. Begin Phase 3 (Conformer architecture)

### If VSCode Crashes (Don't Worry!)
1. Training continues! ✅
2. Reopen VSCode or terminal
3. Run: `./monitor_training.sh`
4. Continue where you left off

---

## Pro Tips

### Keep VSCode Open (Optional)
- Not required! Training will continue even if closed
- But nice to have for other tasks
- Can safely close and reopen

### Monitor Without Interfering
- Use `./monitor_training.sh` for quick checks
- Use `tail -f training_tmux.log` for passive watching
- Don't attach unless you need to interact

### If You Need to Stop
```bash
# Option 1: Kill session
tmux kill-session -t eeg_training

# Option 2: Attach and Ctrl+C
tmux attach -t eeg_training
# Then Ctrl+C
```

---

## Emergency Recovery

### If Everything Crashes
1. Check if tmux session exists: `tmux ls`
2. If exists: `tmux attach -t eeg_training`
3. If not exists: Training died, check logs
4. Restart with: `./start_training_tmux.sh`

### If System Reboots
1. Tmux sessions don't survive reboot
2. Check if checkpoints exist
3. Resume from checkpoint (if implemented)
4. Or restart training fresh

---

## Lessons Learned

### What Didn't Work
- ❌ Regular terminal: Tied to VSCode
- ❌ Nohup: Still vulnerable to VSCode crashes
- ❌ Background jobs: Hard to monitor

### What Works
- ✅ Tmux: Perfect for long-running tasks
- ✅ Independent sessions: Survives crashes
- ✅ Easy monitoring: Multiple ways to check
- ✅ Interactive access: Can attach anytime

---

**Status:** 🎉 Problem Solved!  
**Training:** 🏃 Running smoothly in tmux  
**Crash Resistance:** ✅ VSCode crashes won't affect training  
**Monitoring:** 📊 Easy with `./monitor_training.sh`  

**Last Updated:** October 24, 2025 17:10 UTC  
**Next Check:** When training completes (2-4 hours)
