# Quick Reference - EEG2025 Project

## 🏃 Quick Commands

### Monitor Training
```bash
./monitor_training.sh
```

### Check Training Log
```bash
tail -f logs/foundation_cpu_*.log
```

### Check if Training is Running
```bash
ps aux | grep train_foundation_cpu | grep -v grep
```

### Stop Training (if needed)
```bash
pkill -f train_foundation_cpu
```

### List Checkpoints
```bash
ls -lht checkpoints/
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `scripts/train_foundation_cpu.py` | Main training script (ACTIVE) |
| `scripts/models/eeg_dataset_simple.py` | Dataset loader |
| `monitor_training.sh` | Monitor training progress |
| `checkpoints/foundation_best.pth` | Best model (after training) |
| `logs/foundation_cpu_*.log` | Training logs |
| `PROGRESS_UPDATE.md` | Current progress |
| `NEXT_PHASE.md` | Next steps plan |

## 🎯 Current Status

**Training**: ACTIVE
**Device**: CPU (GPU incompatible - safeguarded)
**Model**: FoundationTransformer (128 hidden, 8 heads, 4 layers)
**Data**: 12 HBN subjects (~3000+ windows)
**Duration**: ~2-4 hours for 20 epochs

## 📊 What to Expect

1. **Data Loading**: 2-5 minutes (loading all EEG files)
2. **Epoch 1**: ~10-15 minutes (first epoch)
3. **Subsequent Epochs**: Similar time
4. **Total**: ~2-4 hours

## 🔔 When Training Finishes

1. Check final log: `tail -100 logs/foundation_cpu_*.log`
2. Find best model: `ls -lh checkpoints/foundation_best.pth`
3. Review history: `cat logs/foundation_history_*.json`
4. Move to challenges (see PROGRESS_UPDATE.md)

## 🆘 Troubleshooting

### Training Stopped?
```bash
# Check if process exists
ps aux | grep train_foundation_cpu

# Check last lines of log
tail -20 logs/foundation_cpu_*.log

# Restart if needed
nohup python3 -u scripts/train_foundation_cpu.py > logs/foundation_cpu_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### VS Code Crashed?
- Training continues in background!
- Reopen VS Code
- Run `./monitor_training.sh` to check status

### Out of Memory?
- Training uses ~6GB RAM
- If OOM, reduce `batch_size` in script (line 32: change from 16 to 8)

## 📈 Expected Results

- **Training Loss**: Should decrease from ~0.7 to ~0.3-0.5
- **Validation Loss**: Should follow training loss
- **Accuracy**: Random start (50%) → ~60-70% (dummy task)
- **Checkpoints**: Saved every 2 epochs + best model

## ✅ Success Criteria

- ✅ Training completes without crashes
- ✅ Loss decreases over epochs  
- ✅ Checkpoints saved successfully
- ✅ Can load best model for challenges

---
**Next**: After training completes, implement competition challenges!
