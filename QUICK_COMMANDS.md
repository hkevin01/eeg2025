# Quick Reference Commands

## Monitor Preprocessing
```bash
# Check if running
ps aux | grep cache_challenge

# Watch log
tail -f logs/preprocessing/cache_safe_*.log

# Check memory
watch -n 5 'free -h'

# Check output
ls -lh data/cached/*.h5
```

## Start Training (After Preprocessing)
```bash
# Verify preprocessing done
ls -lh data/cached/challenge1_*.h5

# Launch training
./train_safe_tmux.sh

# Attach to session
tmux attach -t eeg_train_safe

# Detach: Ctrl+b then d
```

## Emergency
```bash
# Stop everything
pkill -f cache_challenge
pkill -f train_challenge
tmux kill-session -t eeg_train_safe

# Check nothing running
ps aux | grep python
```

## Files to Know
- **START_TRAINING_NOW.md** - Complete guide
- **TRAINING_COMMANDS.md** - Full reference
- **SESSION_SUMMARY_PART*.md** - What happened this session

