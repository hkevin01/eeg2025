# Training Scripts

This folder contains all Challenge 2 training scripts.

## Active Training Scripts

### Current (In Use)
- `train_challenge2_r1r2.py` - **CURRENTLY RUNNING** in tmux session 'training'
  - Training with R1+R2 cache only (23GB)
  - EEGNeX model, batch size 64
  - Early stopping patience=5

### Full Dataset Training
- `train_challenge2_fast.py` - Fast training with full R1-R5 cache
  - Requires all cache files (R1-R5)
  - Uses HDF5 for ultra-fast loading
  - Database logging enabled

- `train_challenge2_correct.py` - Original robust training script
  - Tested and validated
  - CPU-based for stability

## Support Scripts

### Tmux Management
- `start_training_tmux.sh` - Start training in tmux session
- `launch_training_tmux.sh` - Alternative launcher
- `restart_challenge2_training.sh` - Restart training

### Monitoring
- `watchdog_challenge2.sh` - Watchdog for training process
- `manage_watchdog.sh` - Manage watchdog service

## Subdirectories

- `challenge2/` - Various Challenge 2 training variants
- `foundation/` - Foundation model training
- `optimized/` - GPU-optimized training scripts

## Usage

**Start training:**
```bash
tmux new -s training "python3 scripts/training/train_challenge2_r1r2.py"
```

**Monitor:**
```bash
tmux attach -t training
tail -f logs/training_r1r2.log
```

---
*Organized: October 19, 2025*
