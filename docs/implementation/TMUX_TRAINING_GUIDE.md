# 🎯 Tmux Training Guide

## Why Tmux?

Tmux ensures training continues even if:
- Terminal disconnects
- SSH connection drops
- You close your laptop
- You need to logout

The training process runs **independently** of your terminal session.

---

## Quick Start

### Start Training
```bash
cd /home/kevin/Projects/eeg2025
./train_in_tmux.sh
```

This starts Challenge 1 in a tmux session called `eeg_train_c1`.

### Check Status
```bash
./check_training_simple.sh
```

### Start Challenge 2 (After Challenge 1 Completes)
```bash
./train_challenge2_tmux.sh
```

---

## Tmux Commands

### View Training Progress (Attach)
```bash
# Attach to Challenge 1
tmux attach -t eeg_train_c1

# Attach to Challenge 2
tmux attach -t eeg_train_c2
```

**Inside tmux:**
- You'll see live training output
- Scroll up: `Ctrl+B` then `[` (use arrow keys, press `q` to exit scroll mode)
- **Detach (leave running):** `Ctrl+B` then `D`

### List All Sessions
```bash
tmux ls
```

Output example:
```
eeg_train_c1: 1 windows (created Sat Oct 18 15:50:48 2025)
eeg_train_c2: 1 windows (created Sat Oct 18 18:30:00 2025)
```

### Kill a Session (Stop Training)
```bash
# Kill Challenge 1
tmux kill-session -t eeg_train_c1

# Kill Challenge 2
tmux kill-session -t eeg_train_c2
```

---

## Monitoring Without Attaching

### Watch Log Files
```bash
# Challenge 1
tail -f logs/training_comparison/challenge1_improved_*.log

# Challenge 2
tail -f logs/training_comparison/challenge2_improved_*.log

# Press Ctrl+C to stop watching
```

### Check Epochs
```bash
# See recent epochs (Challenge 1)
grep -E "Epoch|NRMSE" logs/training_comparison/challenge1_improved_*.log | tail -20

# Count epochs completed
grep "Epoch" logs/training_comparison/challenge1_improved_*.log | wc -l
```

### Best Validation Score
```bash
grep "Best Val NRMSE" logs/training_comparison/challenge1_improved_*.log | tail -1
```

---

## Workflow

### 1. Start Challenge 1
```bash
./train_in_tmux.sh
```

### 2. Monitor Progress
```bash
# Option A: Quick status (every 30 seconds)
watch -n 30 './check_training_simple.sh'

# Option B: Watch log live
tail -f logs/training_comparison/challenge1_improved_*.log

# Option C: Attach to tmux session
tmux attach -t eeg_train_c1
# (Detach with Ctrl+B then D)
```

### 3. Wait for Completion (~3 hours)
Challenge 1 will train for 50 epochs.

### 4. Start Challenge 2
```bash
./train_challenge2_tmux.sh
```

### 5. Wait for Completion (~3 hours)
Challenge 2 will train for 50 epochs.

### 6. Review Results
```bash
# Check final scores
grep "Best Val NRMSE" logs/training_comparison/challenge1_improved_*.log
grep "Best Val NRMSE" logs/training_comparison/challenge2_improved_*.log
```

---

## Troubleshooting

### Training Appears Stuck
```bash
# Attach to see what's happening
tmux attach -t eeg_train_c1

# Check if process is running
ps aux | grep train_challenge1
```

### Session Not Found
```bash
# List all sessions
tmux ls

# If empty, training stopped - check logs for errors
tail -100 logs/training_comparison/challenge1_improved_*.log | grep -i error
```

### Need to Restart
```bash
# Kill old session
tmux kill-session -t eeg_train_c1

# Start fresh
./train_in_tmux.sh
```

### Can't Detach from Tmux
Press: `Ctrl+B` (release both keys) then `D`

### Want to See Scrollback in Tmux
1. Press: `Ctrl+B` then `[`
2. Use arrow keys or Page Up/Down
3. Press `q` to exit scroll mode

---

## Log Files

All logs are in: `logs/training_comparison/`

Format: `challenge{1|2}_improved_YYYYMMDD_HHMMSS.log`

Example:
- `challenge1_improved_20251018_155048.log`
- `challenge2_improved_20251018_183045.log`

---

## Current Training Status

**Challenge 1:**
- Session: `eeg_train_c1`
- Started: October 18, 2025 at 15:50
- Expected completion: ~18:30
- Log: `logs/training_comparison/challenge1_improved_20251018_155048.log`

**Challenge 2:**
- Will start after Challenge 1 completes
- Run: `./train_challenge2_tmux.sh`
- Expected completion: ~21:30

---

## Tmux Cheat Sheet

| Action | Command |
|--------|---------|
| Start C1 | `./train_in_tmux.sh` |
| Start C2 | `./train_challenge2_tmux.sh` |
| List sessions | `tmux ls` |
| Attach to C1 | `tmux attach -t eeg_train_c1` |
| Attach to C2 | `tmux attach -t eeg_train_c2` |
| Detach | `Ctrl+B` then `D` |
| Kill C1 | `tmux kill-session -t eeg_train_c1` |
| Kill C2 | `tmux kill-session -t eeg_train_c2` |
| Scroll mode | `Ctrl+B` then `[` |
| Exit scroll | `q` |

---

## Benefits of Tmux

✅ **Persistent:** Training continues even if you disconnect
✅ **Resumable:** Attach anytime to see progress
✅ **Safe:** Logs are saved even if session crashes
✅ **Convenient:** Can logout and come back later
✅ **Reliable:** No accidental Ctrl+C killing training

---

## Next Steps

1. ✅ Training running in tmux (Challenge 1)
2. 📋 Monitor with `./check_training_simple.sh`
3. 📋 Start Challenge 2 when Challenge 1 completes (~18:30)
4. 📋 Review results when both complete (~21:30)
5. 📋 Compare to baseline (C1: 1.00 → 0.75, C2: 1.46 → 1.30)

**If results are good (>15% improvement):** Create submission with `python submission.py`
