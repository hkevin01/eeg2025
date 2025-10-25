# 🖥️ Tmux Training Guide - Crash-Resistant Training

**Created:** October 24, 2025 17:06 UTC  
**Status:** ✅ Training running in persistent tmux session

---

## Why Tmux?

**Problem:** VSCode crashed twice, killing the training process each time.

**Solution:** Tmux creates a persistent terminal session that survives:
- ✅ VSCode crashes
- ✅ Terminal closures
- ✅ SSH disconnections
- ✅ System restarts (with tmux-resurrect plugin)

---

## Quick Reference

### Start Training
```bash
./start_training_tmux.sh
```

### Monitor Training
```bash
./monitor_training.sh          # Quick status check
tail -f training_tmux.log      # Live log streaming
tmux attach -t eeg_training    # Full interactive access
```

### Attach to Session
```bash
tmux attach -t eeg_training
# Press Ctrl+B then D to detach (training keeps running!)
```

### Check Status
```bash
tmux ls                        # List all sessions
tmux capture-pane -t eeg_training -p | tail -20  # Peek at output
```

### Stop Training
```bash
tmux kill-session -t eeg_training
```

---

## Current Training Session

**Session Name:** `eeg_training`  
**Started:** October 24, 2025 17:00 UTC  
**Log File:** `training_tmux.log`  
**Experiment:** `experiments/sam_full_run/20251024_165931/`

**Configuration:**
- Epochs: 100
- Batch size: 32
- Learning rate: 1e-3
- SAM rho: 0.05
- Device: CUDA (AMD RX 5600 XT)
- Early stopping: 15 epochs

---

## Tmux Cheat Sheet

### Session Management
```bash
tmux new -s <name>            # Create new session
tmux attach -t <name>         # Attach to session
tmux ls                       # List sessions
tmux kill-session -t <name>   # Kill session
```

### Inside Tmux (Key Bindings)
```
Ctrl+B then D                 # Detach from session
Ctrl+B then [                 # Enter scroll mode (q to exit)
Ctrl+B then PgUp/PgDn        # Scroll up/down
Ctrl+B then ?                 # Show help
```

### Window Management
```
Ctrl+B then C                 # Create new window
Ctrl+B then N                 # Next window
Ctrl+B then P                 # Previous window
Ctrl+B then <number>          # Switch to window number
```

### Pane Management
```
Ctrl+B then %                 # Split vertically
Ctrl+B then "                 # Split horizontally
Ctrl+B then Arrow            # Navigate panes
```

---

## Training Workflow with Tmux

### 1. Start Training
```bash
cd /home/kevin/Projects/eeg2025
./start_training_tmux.sh
```

### 2. Monitor Progress
```bash
# Option A: Quick check
./monitor_training.sh

# Option B: Live streaming
tail -f training_tmux.log

# Option C: Full interactive
tmux attach -t eeg_training
```

### 3. Detach Safely
```
# While attached to tmux:
Press Ctrl+B, then press D
# Training continues running!
```

### 4. Check Results Later
```bash
# After training completes
cat training_tmux.log | grep "Best Val NRMSE"

# Check experiment outputs
ls -lh experiments/sam_full_run/*/checkpoints/
```

---

## Advantages Over Nohup

| Feature | Nohup | Tmux |
|---------|-------|------|
| Survives logout | ✅ | ✅ |
| Interactive access | ❌ | ✅ |
| Real-time monitoring | ❌ | ✅ |
| Multiple windows | ❌ | ✅ |
| Detach/reattach | ❌ | ✅ |
| Copy/paste | ❌ | ✅ |
| Split panes | ❌ | ✅ |

---

## Common Issues

### "Session not found"
```bash
# Check if session exists
tmux ls

# If not listed, training isn't running
# Restart with:
./start_training_tmux.sh
```

### "Session already exists"
```bash
# Attach to existing session
tmux attach -t eeg_training

# Or kill and restart
tmux kill-session -t eeg_training
./start_training_tmux.sh
```

### Can't see output
```bash
# Capture current pane content
tmux capture-pane -t eeg_training -p | tail -50

# Or check log file
tail -100 training_tmux.log
```

---

## Best Practices

### ✅ DO
- Use tmux for long-running training jobs
- Detach (Ctrl+B D) instead of closing terminal
- Check logs with `tail -f` for monitoring
- Create separate tmux sessions for different experiments

### ❌ DON'T
- Don't use Ctrl+C in tmux (will kill training!)
- Don't close terminal with tmux attached (detach first)
- Don't create multiple sessions with same name
- Don't forget to check GPU usage (`rocm-smi`)

---

## Automation Scripts

### Start Training in Tmux
**File:** `start_training_tmux.sh`
- Creates session `eeg_training`
- Runs training command
- Pipes output to `training_tmux.log`

### Monitor Training
**File:** `monitor_training.sh`
- Shows session status
- Displays recent log output
- Shows GPU usage
- Provides helpful commands

---

## Recovery from Crashes

### If VSCode Crashes
1. ✅ Training continues running (in tmux!)
2. Reopen terminal or VSCode
3. Check status: `./monitor_training.sh`
4. Resume monitoring: `tmux attach -t eeg_training`

### If Training Crashes
1. Check log file: `cat training_tmux.log | tail -100`
2. Check error messages
3. Fix issues in training script
4. Restart: `./start_training_tmux.sh`

### If System Restarts
1. Training will be lost (tmux sessions don't survive reboot)
2. Use checkpoints to resume
3. Restart training with `--resume` flag (if implemented)

---

## Pro Tips

### Multi-Window Setup
```bash
# Create windows for different tasks
tmux new -s monitoring
tmux rename-window -t monitoring logs
tmux new-window -n gpu
tmux send-keys -t monitoring:gpu "watch -n 2 rocm-smi" C-m
tmux new-window -n results
```

### Log Analysis
```bash
# Extract epoch results
grep "Epoch" training_tmux.log > epoch_results.txt

# Find best NRMSE
grep "BEST" training_tmux.log

# Count epochs completed
grep "Epoch.*/" training_tmux.log | wc -l
```

### Save Tmux Layout
```bash
# Save current session layout
tmux list-windows -t eeg_training > tmux_layout.txt

# Restore layout (manual recreation needed)
```

---

## Integration with Project

### Files Created
- `start_training_tmux.sh` - Start training in tmux
- `monitor_training.sh` - Check training status
- `TMUX_TRAINING_GUIDE.md` (this file) - Complete guide

### Log Files
- `training_tmux.log` - Complete training output
- `experiments/sam_full_run/*/history.json` - Training metrics

### Commands Added to Workflow
```bash
# Start
./start_training_tmux.sh

# Monitor
./monitor_training.sh

# Attach
tmux attach -t eeg_training

# Detach
Ctrl+B then D
```

---

## Success Metrics

### ✅ Achievements
- [x] Tmux session created successfully
- [x] Training started in persistent session
- [x] Output logged to file
- [x] Monitoring scripts created
- [x] Documentation complete

### 🎯 Goals
- [ ] Training completes without interruption
- [ ] Survive potential VSCode crashes
- [ ] Easy monitoring and management
- [ ] Best model saved in checkpoints

---

**Status:** ✅ Training running in tmux  
**Session:** `eeg_training`  
**Monitor:** `./monitor_training.sh`  
**Log:** `tail -f training_tmux.log`  

**Last Updated:** October 24, 2025 17:06 UTC
