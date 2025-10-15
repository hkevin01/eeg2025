# TODO Checklist - EEG2025 Project

**Last Updated:** October 14, 2025, 18:45

---

## ✅ COMPLETED

```markdown
- [x] VS Code optimization (removed heavy extensions)
- [x] GPU safety system (comprehensive documentation)
- [x] Training scripts created (minimal & full)
- [x] Dataset loader tested and working
- [x] System resources verified (24GB RAM free, 120GB disk)
- [x] Removed Pylint, Mypy, Pylance, Flake8, Autopep8, Black, Isort
- [x] Created .vscode/settings.json for performance
- [x] Killed duplicate training processes
```

---

## 🎯 PRIORITY 1: Foundation Training (CRITICAL) ⭐⭐⭐

**Status:** ⚠️ NOT STARTED  
**Blocker:** Training keeps getting interrupted  
**Solution:** Run minimal training first (quick, reliable)

### Step 1.1: Run Minimal Training (10-15 min)
```markdown
- [ ] Open terminal
- [ ] cd /home/kevin/Projects/eeg2025
- [ ] Run: python3 scripts/train_minimal.py | tee logs/minimal_$(date +%Y%m%d_%H%M%S).log
- [ ] Wait for completion (10-15 minutes)
- [ ] Verify checkpoint created: checkpoints/minimal_best.pth
```

**Command to run:**
```bash
cd /home/kevin/Projects/eeg2025
python3 scripts/train_minimal.py | tee logs/minimal_$(date +%Y%m%d_%H%M%S).log
```

### Step 1.2: Verify Training Completed
```markdown
- [ ] Check checkpoint exists: ls -lh checkpoints/minimal_best.pth
- [ ] Check history exists: ls -lh logs/minimal_history.json
- [ ] Load checkpoint in Python to verify it works
```

---

## �� PRIORITY 2: Challenge 1 Implementation ⭐⭐⭐

**Status:** ⭕ WAITING (needs trained model from Priority 1)  
**Time Estimate:** 30 minutes  
**Goal:** Create age prediction submission

### Step 2.1: Verify Challenge 1 Script
```markdown
- [ ] Check script exists: ls scripts/train_challenge1.py
- [ ] Review script configuration
- [ ] Ensure uses checkpoint from Priority 1
```

### Step 2.2: Run Challenge 1 Training
```markdown
- [ ] Run: python3 scripts/train_challenge1.py
- [ ] Monitor training progress
- [ ] Wait for completion
```

### Step 2.3: Verify Submission Created
```markdown
- [ ] Check submission: ls -lh submissions/challenge1_predictions.csv
- [ ] Verify format: participant_id, age_prediction
- [ ] Check metrics: Pearson r > 0.3, AUROC > 0.7
```

---

## 🎯 PRIORITY 3: Challenge 2 Implementation (OPTIONAL) ⭐⭐

**Status:** ⭕ NOT STARTED  
**Time Estimate:** 30 minutes  
**Goal:** Create sex classification submission

### Step 3.1: Create Challenge 2 Script
```markdown
- [ ] Copy train_challenge1.py to train_challenge2.py
- [ ] Modify for sex classification (binary)
- [ ] Update submission format
```

### Step 3.2: Run Challenge 2 Training
```markdown
- [ ] Run: python3 scripts/train_challenge2.py
- [ ] Monitor training progress
- [ ] Wait for completion
```

### Step 3.3: Verify Submission Created
```markdown
- [ ] Check submission: ls -lh submissions/challenge2_predictions.csv
- [ ] Verify format: participant_id, sex_prediction
- [ ] Check accuracy metrics
```

---

## 🎯 PRIORITY 4: Competition Submission (STRETCH) ⭐

**Status:** ⭕ NOT STARTED  
**Depends On:** Priority 2 (at minimum)

```markdown
- [ ] Test Challenge 1 submission locally
- [ ] Upload Challenge 1 to competition platform
- [ ] (Optional) Test Challenge 2 submission locally
- [ ] (Optional) Upload Challenge 2 to competition platform
- [ ] Monitor leaderboard
```

---

## 🎯 OPTIONAL: Full Training (STRETCH)

**Status:** ⭕ NOT STARTED  
**Time Estimate:** 2-4 hours  
**When:** After minimal training succeeds

```markdown
- [ ] Run: nohup python3 scripts/train_simple.py > logs/full_$(date +%Y%m%d_%H%M%S).log 2>&1 &
- [ ] Monitor: tail -f logs/full_*.log
- [ ] Wait for completion (2-4 hours)
- [ ] Replace minimal checkpoint with full checkpoint
- [ ] Re-run Challenge 1 with better model
- [ ] Re-run Challenge 2 with better model
```

---

## 📊 Progress Summary

| Task | Status | Time Estimate | Priority |
|------|--------|---------------|----------|
| VS Code Optimization | ✅ Done | - | Completed |
| Minimal Training | ⚠️ Next | 10-15 min | 🔴 CRITICAL |
| Challenge 1 | ⭕ Waiting | 30 min | 🔴 HIGH |
| Challenge 2 | ⭕ Optional | 30 min | 🟡 MEDIUM |
| Full Training | ⭕ Optional | 2-4 hours | 🟢 LOW |
| Competition Submit | ⭕ Stretch | 15 min | 🟢 LOW |

---

## 🚀 Quick Start (DO THIS NOW)

**Step 1: Run Minimal Training**
```bash
cd /home/kevin/Projects/eeg2025
python3 scripts/train_minimal.py | tee logs/minimal_$(date +%Y%m%d_%H%M%S).log
```

**Step 2: After it completes (10-15 min), run Challenge 1**
```bash
python3 scripts/train_challenge1.py
```

**Step 3: Check your submission**
```bash
ls -lh submissions/challenge1_predictions.csv
cat submissions/challenge1_predictions.csv | head
```

---

## 🎯 Definition of Done

### Minimum Success:
- [x] VS Code optimized
- [ ] **Minimal training completed** ← DO THIS NOW
- [ ] **Challenge 1 submission created**
- [ ] Checkpoint saved and loadable

### Full Success:
- [ ] Full training completed
- [ ] Both challenges submitted
- [ ] Competition submission uploaded
- [ ] Leaderboard position recorded

---

## 📝 Notes

- **Current blocker:** Need to complete one successful training run
- **Solution:** Run minimal training (quick & reliable)
- **VS Code:** Should no longer crash (extensions removed)
- **GPU:** Not using (unstable), CPU is safe
- **Time required:** ~45 minutes total (15 min training + 30 min Challenge 1)

---

## ⚠️ IMPORTANT

**You MUST complete Priority 1 (minimal training) before anything else can proceed.**

**Command to run RIGHT NOW:**
```bash
cd /home/kevin/Projects/eeg2025 && python3 scripts/train_minimal.py | tee logs/minimal_$(date +%Y%m%d_%H%M%S).log
```

This will take 10-15 minutes. Let it complete without interruption.

---

**Ready to start? Run the command above! 🚀**
