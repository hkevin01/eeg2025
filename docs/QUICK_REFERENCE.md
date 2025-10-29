# 🚀 Quick Reference - Training Commands

## 📊 Check Training Status

```bash
# Quick status check
bash monitor_training.sh

# Watch live (press Ctrl+B then D to detach)
tmux attach -t eeg_training

# Check latest log entries
tail -30 logs/train_all_rsets_20251028_145812.log

# Follow log in real-time
tail -f logs/train_all_rsets_20251028_145812.log
```

## 🎯 Current Mission

- **Goal:** C1 score < 0.93 (top 3 placement)
- **Current:** C1 = 1.0015 (baseline, 8% behind leaders)
- **Status:** Training ALL R-sets (41,071 samples) epoch 9/30
- **ETA:** Complete by 6:30 PM, test results by 7:15 PM

## 📂 Key Files

- **Training script:** `scripts/experiments/train_c1_all_rsets.py`
- **Live log:** `logs/train_all_rsets_20251028_145812.log`
- **Best checkpoint:** `checkpoints/compact_cnn_all_rsets_best.pth`
- **Status docs:** 
  - `STATUS_SUMMARY_OCT28_3PM.md` (concise)
  - `TRAINING_IN_PROGRESS_OCT28.md` (detailed)

## ⚠️ Important

- **DON'T trust validation metrics** (previous val 0.16 → test 1.14!)
- **ONLY test score matters** (submit to know if it works)
- **Safety net:** quick_fix (1.0065) already submitted
- **Deadline:** Nov 2 (4 days remaining)

## 🎓 What We're Testing

**Hypothesis:** Test set is random mixture of R1-R4
- Previous: R4-only → 1.0020 (failed)
- Previous: R1-R3 → 1.1398 (disaster!)
- **Current: ALL R-sets random → TBD** ⬅️

## 🏆 Success Criteria

- **Minimum:** C1 < 0.98 → Proves approach works
- **Good:** C1 < 0.95 → Significant progress
- **Target:** C1 < 0.93 → **TOP 3!** 🏆
- **Stretch:** C1 < 0.92 → Top 2!

---

**Next check:** 4:00 PM  
**Complete:** 6:30 PM  
**Results:** 7:15 PM  

🎯 **GOAL: TOP 3 PLACEMENT!**
