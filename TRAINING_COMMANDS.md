# Training Commands - Quick Reference

## 1. Preprocessing (Run First!)

### Memory-Safe Preprocessing
```bash
# Process all releases (R1-R4)
python scripts/preprocessing/cache_challenge1_windows_safe.py

# Process specific releases
python scripts/preprocessing/cache_challenge1_windows_safe.py --releases R1 R2

# Test with mini dataset first
python scripts/preprocessing/cache_challenge1_windows_safe.py --mini
```

**Expected:**
- Creates `data/cached/challenge1_R{1,2,3,4}_windows.h5`
- Each file: 2-4GB
- Total: ~12GB
- Memory usage: < 4GB RAM

**If it crashes:**
- Check logs: `logs/preprocessing/cache_safe_*.log`
- Resume: Script automatically skips already-processed releases
- Reduce memory: Process one release at a time

## 2. Training with Crash Protection

### Launch Safe Training
```bash
# Start training in tmux with memory monitoring
./train_safe_tmux.sh
```

**What it does:**
- Creates tmux session `eeg_train_safe`
- Left pane: Training output
- Right pane: Memory monitor (updates every 5s)
- Auto-logs to `logs/training_comparison/training_safe_*.log`

### Monitor Training
```bash
# Attach to session
tmux attach -t eeg_train_safe

# Detach: Ctrl+b then d

# Watch log in separate terminal
tail -f logs/training_comparison/training_safe_*.log

# Check if running
ps aux | grep train_challenge

# Check memory
free -h
```

### Stop Training
```bash
# Kill tmux session
tmux kill-session -t eeg_train_safe

# Or inside tmux: Ctrl+c
```

## 3. Challenge 2 Training

### Memory-Safe Challenge 2
```bash
# The Challenge 2 script already has memory safety
python scripts/training/challenge2/train_challenge2_multi_release.py
```

## 4. Monitoring Commands

### Memory Status
```bash
# Current memory
free -h

# Continuous monitoring
watch -n 2 'free -h'

# Top memory users
ps aux --sort=-%mem | head -10
```

### Training Status
```bash
# Check processes
ps aux | grep python | grep train

# Check GPU (if using)
nvidia-smi

# Disk space
df -h

# Check cached files
ls -lh data/cached/
du -sh data/cached/
```

### Logs
```bash
# Latest preprocessing log
ls -lth logs/preprocessing/ | head -3

# Latest training log
ls -lth logs/training_comparison/ | head -3

# Tail training log
tail -f logs/training_comparison/training_safe_*.log

# Search for errors
grep -i "error\|exception\|crash" logs/training_comparison/*.log
```

## 5. Troubleshooting

### If Preprocessing Crashes
```bash
# 1. Check log
tail -100 logs/preprocessing/cache_safe_*.log

# 2. Check memory
free -h

# 3. Resume (skips completed)
python scripts/preprocessing/cache_challenge1_windows_safe.py

# 4. Process one at a time
python scripts/preprocessing/cache_challenge1_windows_safe.py --releases R1
python scripts/preprocessing/cache_challenge1_windows_safe.py --releases R2
```

### If Training Crashes
```bash
# 1. Check log
tail -100 logs/training_comparison/training_safe_*.log

# 2. Check memory
free -h

# 3. Check if model saved
ls -lth weights_*.pt | head -3

# 4. Restart
./train_safe_tmux.sh
```

### If Memory Issues
```bash
# 1. Close other programs

# 2. Check memory hogs
ps aux --sort=-%mem | head -10

# 3. Clear cache
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

# 4. Use smaller batch size (edit training script)
# Change: batch_size=32 → batch_size=16
```

## 6. Post-Training

### Check Results
```bash
# Check if training completed
grep "TRAINING COMPLETE\|Best validation" logs/training_comparison/training_safe_*.log

# Check model weights
ls -lh weights_*.pt

# Compare to baseline
echo "Baseline Challenge 1: 1.00 NRMSE"
echo "Target: < 0.85 NRMSE"
```

### Clean Up
```bash
# Remove old logs (keep last 10)
cd logs/training_comparison && ls -t | tail -n +11 | xargs rm -f

# Remove tmux session
tmux kill-session -t eeg_train_safe
```

## 7. Complete Workflow

```bash
# Step 1: Preprocessing (30-60 min)
python scripts/preprocessing/cache_challenge1_windows_safe.py

# Step 2: Verify cached files
ls -lh data/cached/
du -sh data/cached/

# Step 3: Test HDF5 loading
python src/utils/hdf5_dataset.py

# Step 4: Start training
./train_safe_tmux.sh

# Step 5: Monitor (in another terminal)
watch -n 10 'tail -20 logs/training_comparison/training_safe_*.log'

# Step 6: Check results after training
grep "Best validation" logs/training_comparison/training_safe_*.log
```

## 8. Emergency Commands

### Kill Everything
```bash
# Kill all Python training
pkill -f train_challenge

# Kill tmux
tmux kill-session -t eeg_train_safe

# Check nothing running
ps aux | grep python
```

### System Recovery
```bash
# If system frozen
# Press: Ctrl+Alt+F3
# Login
# Kill processes: pkill -9 python
# Reboot: sudo reboot

# After reboot, resume:
python scripts/preprocessing/cache_challenge1_windows_safe.py  # Resumes
```

---

**Quick Start:**
```bash
# 1. Preprocess
python scripts/preprocessing/cache_challenge1_windows_safe.py

# 2. Train
./train_safe_tmux.sh

# 3. Monitor
tmux attach -t eeg_train_safe
```

**Help:**
- Preprocessing log: `logs/preprocessing/cache_safe_*.log`
- Training log: `logs/training_comparison/training_safe_*.log`
- Memory safe: Stops at 85% RAM
- Resume capable: Skips completed work
