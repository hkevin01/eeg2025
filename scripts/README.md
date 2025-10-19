# Scripts Directory

Active scripts for the EEG 2025 competition.

## Monitoring Scripts (`monitoring/`)

Scripts for monitoring Challenge 2 training in real-time.

### `watchdog_challenge2.sh` 🐕
**Purpose:** Automated crash and freeze detection system  
**Status:** Currently running (PID in watchdog logs)  
**Features:**
- Detects training crashes (process dies)
- Detects freezes (no log updates for 5+ minutes)
- Scans for errors in logs
- Monitors memory usage (warns at 90%)
- Visual + audio alerts
- Automatic completion detection

**Usage:**
```bash
# Managed by manage_watchdog.sh - don't run directly
```

### `manage_watchdog.sh`
**Purpose:** Control interface for the watchdog  
**Features:**
- Start/stop/restart watchdog
- Check watchdog status
- View watchdog logs
- Follow live output

**Usage:**
```bash
./manage_watchdog.sh status    # Check status
./manage_watchdog.sh start     # Start watchdog
./manage_watchdog.sh stop      # Stop watchdog
./manage_watchdog.sh restart   # Restart watchdog
./manage_watchdog.sh logs      # View logs
./manage_watchdog.sh follow    # Follow live output
```

### `monitor_challenge2.sh`
**Purpose:** Full-featured training monitor with auto-refresh  
**Features:**
- Process status tracking (PID, CPU, memory, runtime)
- Training phase detection (data loading, window creation, training)
- Epoch and batch progress
- Color-coded log entries
- GPU monitoring (if available)
- Next steps guidance
- Auto-refresh every 30 seconds

**Usage:**
```bash
./monitor_challenge2.sh
# Press Ctrl+C to exit
```

### `quick_training_status.sh`
**Purpose:** Quick training status snapshot (no auto-refresh)  
**Features:**
- Instant progress check
- Current epoch and batch
- Recent loss average
- Last 3 log lines

**Usage:**
```bash
./quick_training_status.sh
```

## Training Scripts (`training/`)

### `train_challenge2_correct.py` ⭐
**Purpose:** Challenge 2 training script (following official starter kit)  
**Status:** Currently running in background  
**Features:**
- Uses contrastChangeDetection task (correct!)
- Predicts p_factor (externalizing factor)
- 4-second windows with 2-second random crops
- EEGNeX model (generalization-focused)
- L1 loss (robust to outliers)
- Adamax optimizer (lr=0.002)
- Early stopping (patience=5)
- Trains on R1-R4, validates on R5

**Usage:**
```bash
# Run in background (recommended)
nohup python -u train_challenge2_correct.py > logs/challenge2_correct_training.log 2>&1 &

# Or run directly
python train_challenge2_correct.py
```

**Monitoring:**
- Log: `logs/challenge2_correct_training.log`
- Saves best model to: `weights_challenge_2_correct.pt`
- Monitored by: `watchdog_challenge2.sh`

## Quick Reference

**Check training status:**
```bash
cd scripts/monitoring
./quick_training_status.sh
```

**Monitor training (auto-refresh):**
```bash
cd scripts/monitoring
./monitor_challenge2.sh
```

**Manage watchdog:**
```bash
cd scripts/monitoring
./manage_watchdog.sh status
```

**View logs:**
```bash
tail -f logs/challenge2_correct_training.log  # Training
tail -f logs/watchdog.log                     # Watchdog
```

## Notes

- All monitoring scripts can be run from project root or scripts/monitoring/
- Training script should be run from project root
- Watchdog automatically monitors the training script
- See `../WATCHDOG_QUICK_REFERENCE.md` for watchdog details
- See `../CHALLENGE2_TRAINING_STATUS.md` for training details

**Last Updated:** October 19, 2025
