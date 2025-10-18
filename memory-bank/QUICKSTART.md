# Quick Start Guide - Resume EEG 2025 Challenge

This document helps you quickly resume the project from any state.

## 🚀 Immediate Status Check

```bash
# Check if Challenge 2 training is still running
tmux list-sessions | grep eeg_both

# Quick training progress
./check_c2_training.sh

# View latest training results
tail -30 logs/train_c2_tcn_*.log
```

## 📊 Current Project State (as of Oct 17, 2025 22:35)

### ✅ COMPLETE
- **Challenge 1 TCN:** Trained and integrated (val loss 0.010170, 65% improvement)
- **Submission.py:** Updated with Challenge 1 TCN
- **Training System:** Independent tmux-based (survives crashes)
- **Documentation:** Complete memory bank created

### 🔄 IN PROGRESS
- **Challenge 2 TCN:** Training (epoch 4/100, val loss 0.668 best so far)
- **Tmux Session:** eeg_both_challenges (active)
- **Log File:** logs/train_c2_tcn_20251017_221832.log

### ⏳ PENDING
- Challenge 2 integration into submission.py
- Final submission testing
- Package submission v6
- Upload to Codabench

## 📁 Key Files & Locations

### Models & Checkpoints
```
checkpoints/challenge1_tcn_competition_best.pth   # Challenge 1 (2.4 MB) ✅
checkpoints/challenge2_tcn_competition_best.pth   # Challenge 2 (training) 🔄
```

### Code
```
submission.py                           # Competition submission (Challenge 1 ready)
scripts/train_challenge2_tcn.py         # Challenge 2 training script
scripts/monitoring/                     # Training monitors
```

### Documentation
```
memory-bank/
  ├── app-description.md                # Project overview
  ├── change-log.md                     # All changes with dates
  ├── QUICKSTART.md                     # This file
  ├── implementation-plans/
  │   └── submission-v6.md              # ACID plan for submission v6
  └── architecture-decisions/
      └── tcn-choice.md                 # Why we chose TCN
```

### Logs & Monitoring
```
logs/train_fixed_20251017_184601.log    # Challenge 1 complete
logs/train_c2_tcn_20251017_221832.log   # Challenge 2 active
check_c2_training.sh                    # Quick progress checker
```

## 🎯 Next Actions (Based on Current State)

### If Challenge 2 Still Training (Most Likely)

```bash
# 1. Check progress
./check_c2_training.sh

# 2. View real-time training
tail -f logs/train_c2_tcn_20251017_221832.log

# 3. Check validation results
grep -E "^Epoch|Train Loss:|Val Loss:" logs/train_c2_tcn_20251017_221832.log

# 4. Wait for completion (ETA: 30-60 min from 22:18 start)
# Training will auto-stop when optimal or max epochs reached
```

### If Challenge 2 Completed

```bash
# 1. Check final results
grep "Best model" logs/train_c2_tcn_20251017_221832.log

# 2. Verify checkpoint exists
ls -lh checkpoints/challenge2_tcn_competition_best.pth

# 3. Get final validation loss
grep "Val Loss:" logs/train_c2_tcn_20251017_221832.log | tail -1

# 4. Proceed to integration (see submission-v6.md task A4)
```

### If Training Failed

```bash
# 1. Check for errors in log
tail -100 logs/train_c2_tcn_20251017_221832.log | grep -i error

# 2. Check tmux session status
tmux list-sessions

# 3. Restart if needed
tmux kill-session -t eeg_both_challenges
./scripts/train_both_challenges.sh

# 4. If problems persist, review change-log.md for fixes
```

## 🔧 Common Recovery Scenarios

### Scenario 1: Lost Chat Context
**You Are Here!** This memory bank is designed for this exact scenario.

**Actions:**
1. Read `memory-bank/app-description.md` - Get project overview
2. Read `memory-bank/implementation-plans/submission-v6.md` - See current plan
3. Check `memory-bank/change-log.md` - Review recent changes
4. Run status checks above - Assess current state
5. Continue from next pending task in submission-v6.md

### Scenario 2: VS Code Crashed
**Training continues!** We use tmux for this reason.

**Actions:**
```bash
# Reattach to training session
tmux attach -t eeg_both_challenges

# Or just check progress
./check_c2_training.sh

# Training logs are safe
tail -f logs/train_c2_tcn_*.log
```

### Scenario 3: System Reboot
**Check if training session survived:**

```bash
# List tmux sessions
tmux list-sessions

# If session exists, reattach
tmux attach -t eeg_both_challenges

# If session gone, check last log state
tail -100 logs/train_c2_tcn_20251017_221832.log

# Restart if needed (from last checkpoint)
# Note: Training script auto-saves best model, can resume manually if needed
```

### Scenario 4: Need to Submit NOW
**Emergency submission with just Challenge 1:**

```bash
# Challenge 1 is ready, submit without waiting for Challenge 2
mkdir -p submission_emergency
cd submission_emergency

# Use current submission.py (has Challenge 1 TCN)
cp ../submission.py .
cp ../challenge1_tcn_competition_best.pth .

# Keep old Challenge 2 model
cp ../weights_challenge_2_multi_release.pt .

# Package
zip -r ../eeg2025_submission_emergency.zip .

# Upload to: https://www.codabench.org/competitions/4287/
```

## 📚 Understanding the Architecture

### Quick Architecture Overview

**Model: TCN_EEG (Temporal Convolutional Network)**

```
Input: (batch, 129 channels, 200 timepoints)
  ↓
TemporalBlock (dilation=1) → 48 filters
  ↓
TemporalBlock (dilation=2) → 48 filters
  ↓
TemporalBlock (dilation=4) → 48 filters
  ↓
TemporalBlock (dilation=8) → 48 filters
  ↓
TemporalBlock (dilation=16) → 48 filters
  ↓
Global Average Pool
  ↓
Linear: 48 → 1
  ↓
Output: Single value (response time or externalizing score)
```

**Parameters:** 196,225 (very efficient!)  
**Receptive Field:** 127 timepoints (63% of input)

**Why TCN?**
- Captures long-range temporal dependencies
- Efficient (77% fewer params than attention)
- Fast training & inference
- Proven: 65% improvement on Challenge 1

See `memory-bank/architecture-decisions/tcn-choice.md` for full rationale.

## 🎓 Reading Order for Deep Dive

1. **QUICKSTART.md** (you are here) - Immediate orientation
2. **app-description.md** - Project goals, datasets, technical stack
3. **implementation-plans/submission-v6.md** - Current work breakdown
4. **change-log.md** - What's been done, when, and why
5. **architecture-decisions/tcn-choice.md** - Model design rationale

## 💡 Useful Commands

### Training Monitoring
```bash
# Quick status
./check_c2_training.sh

# Watch live
tail -f logs/train_c2_tcn_20251017_221832.log

# Attach to training session
tmux attach -t eeg_both_challenges
# (Ctrl+B then D to detach)

# Get all epoch results
grep -E "^Epoch|Val Loss:" logs/train_c2_tcn_*.log
```

### Model Testing
```bash
# Test current submission
python3 submission.py

# Check model size
ls -lh checkpoints/*.pth

# Verify model architecture
python3 -c "import torch; from submission import TCN_EEG; \
    model = TCN_EEG(); \
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')"
```

### Data Inspection
```bash
# Check dataset access
python3 -c "from eegdash import EEGChallengeDataset; \
    ds = EEGChallengeDataset(release='R1', query=dict(task='VisualStimulus')); \
    print(f'R1 VisualStimulus: {len(ds.datasets)} recordings')"
```

### Git Status
```bash
# Check what's changed
git status

# Review recent commits
git log --oneline -10

# Stage memory bank
git add memory-bank/

# Commit documentation
git commit -m "Add comprehensive memory bank for project recovery"
```

## ⚠️ Important Notes

1. **Training is Independent:** Uses tmux, survives IDE crashes
2. **Checkpoints Auto-Save:** Best model saved automatically
3. **Early Stopping:** Will stop training when optimal (patience 15)
4. **Logs Preserved:** All training logs kept for reproducibility
5. **Documentation First:** Always check memory-bank/ before making changes

## 🆘 Troubleshooting

### "I don't know what's happening"
→ Run `./check_c2_training.sh` and read the output

### "Training seems stuck"
→ Check `tail -20 logs/train_c2_tcn_*.log` - if batch numbers increasing, it's running

### "I need to make changes to training"
→ DON'T! Training is in progress. Wait for completion, then retrain if needed.

### "How do I know if Challenge 2 is good enough?"
→ Check val loss: Target < 0.30 (NRMSE). Baseline is 0.2917.  
→ Current best: 0.668 (NRMSE 0.817) - Still improving

### "What if Challenge 2 never improves?"
→ Submit with Challenge 1 TCN + old Challenge 2 model (Plan B in submission-v6.md)

## 🏁 Success Criteria

**Ready to Submit When:**
- [ ] Challenge 2 training complete (or decided to use old model)
- [ ] Both models integrated in submission.py
- [ ] Local testing passes (both predictions reasonable)
- [ ] Submission package < 50 MB (expected ~5 MB)
- [ ] All documentation updated

**Competition Success:**
- [ ] Challenge 1 NRMSE < 0.15 (target: ~0.10)
- [ ] Challenge 2 NRMSE < 0.30 (target: 0.15-0.25)
- [ ] Overall ranking: Top 5
- [ ] No validation errors on Codabench

## �� Where to Get More Info

- **Project Overview:** `memory-bank/app-description.md`
- **Current Plan:** `memory-bank/implementation-plans/submission-v6.md`
- **Change History:** `memory-bank/change-log.md`
- **Architecture Details:** `memory-bank/architecture-decisions/tcn-choice.md`
- **Competition:** https://www.codabench.org/competitions/4287/

---

**Last Updated:** October 17, 2025 22:35  
**Status:** Challenge 2 training in progress (epoch 4/100)  
**Next Milestone:** Challenge 2 completion (~30-60 min)

