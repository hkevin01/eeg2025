# ✅ Daily Checklist Template

**Date**: _____________  
**Week**: ___ / 4  
**Time Available**: _______ hours

---

## 🌅 Morning Routine (30 min)

- [ ] Check overnight training (if any)
- [ ] Review yesterday's progress
- [ ] Set today's 3 main goals
- [ ] Check for blockers/issues

**Today's Top 3 Goals**:
1. ___________________________
2. ___________________________
3. ___________________________

---

## 💻 Work Session 1 (90 min)

**Focus**: Data / Code / Testing

- [ ] Task 1: ___________________________
- [ ] Task 2: ___________________________
- [ ] Task 3: ___________________________

**Notes**:
```
[Write any observations, errors, or insights here]
```

---

## �� Work Session 2 (90 min)

**Focus**: Training / Evaluation / Optimization

- [ ] Task 1: ___________________________
- [ ] Task 2: ___________________________
- [ ] Task 3: ___________________________

**Metrics Tracked**:
- Challenge 1 Score: _______
- Challenge 2 Score: _______
- Inference Time: _______ ms

---

## 🌙 Evening Wrap-Up (30 min)

### What Got Done Today ✓
- ___________________________
- ___________________________
- ___________________________

### What's Blocked ⚠️
- ___________________________
- ___________________________

### Tomorrow's Plan 📋
- ___________________________
- ___________________________
- ___________________________

---

## 📊 Quick Status Check

Run this command to see your progress:

```bash
# Quick daily status
python << 'PYTHON'
import os
from datetime import datetime

print(f"\n{'='*50}")
print(f"📊 Daily Status - {datetime.now().strftime('%Y-%m-%d')}")
print(f"{'='*50}\n")

# Check data
data_exists = os.path.exists('data/raw/hbn')
print(f"✓ Data Ready: {'Yes' if data_exists else 'No'}")

# Check models
models = len([f for f in os.listdir('checkpoints') if f.endswith('.pt')]) if os.path.exists('checkpoints') else 0
print(f"✓ Models Trained: {models}")

# Check tests
tests = len([f for f in os.listdir('tests') if f.startswith('test_')]) if os.path.exists('tests') else 0
print(f"✓ Tests Written: {tests}")

print(f"\n{'='*50}\n")
PYTHON
```

---

## 🎯 Weekly Progress Tracker

| Day | Data | Models | Tests | Score |
|-----|------|--------|-------|-------|
| Mon |  ⭕  |   ⭕   |  ⭕   |   -   |
| Tue |  ⭕  |   ⭕   |  ⭕   |   -   |
| Wed |  ⭕  |   ⭕   |  ⭕   |   -   |
| Thu |  ⭕  |   ⭕   |  ⭕   |   -   |
| Fri |  ⭕  |   ⭕   |  ⭕   |   -   |
| Sat |  ⭕  |   ⭕   |  ⭕   |   -   |
| Sun |  ⭕  |   ⭕   |  ⭕   |   -   |

**Legend**: ⭕ Not Started | 🟡 In Progress | ✅ Complete

---

## 💡 Today's Learnings

**What I Learned**:
```
[Key insights, bugs fixed, new techniques discovered]
```

**What I'll Do Differently Tomorrow**:
```
[Process improvements, time management, etc.]
```

---

## 🚀 Quick Commands

Keep these handy:

```bash
# Run tests
pytest tests/ -v --tb=short

# Check model
python scripts/check_model_status.py

# Quick training (10 epochs)
python src/training/train_cross_task.py --epochs 10 --quick

# Validate submission
python scripts/validate_submission.py --challenge challenge1
```

---

**Remember**: 
- ✅ Small progress daily > Big bursts occasionally
- 🎯 Focus on one thing at a time
- 📝 Document everything (Future You will thank you!)
- 🛑 Stop if stuck for >30 min, ask for help

---

**End of Day Score**: ___ / 10  
**Energy Level**: Low / Medium / High  
**Tomorrow's First Task**: _______________________
