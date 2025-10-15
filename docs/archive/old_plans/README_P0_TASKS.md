# 📋 P0 Critical Tasks - Complete Guide

**Status**: All documentation created ✅  
**Ready**: To start execution  
**Goal**: Complete before training begins

---

## 🎯 Quick Navigation

**Want to start NOW?**  
→ Read: `START_HERE_P0.md`  
→ Run: `./check_p0_status.sh`

**Want full details?**  
→ Read: `CRITICAL_TASKS_P0.md`

**Want data guide?**  
→ Read: `docs/DATA_ACQUISITION_GUIDE.md`

---

## 📂 All P0 Documentation

### Root Directory Files
| File | Purpose | Use When |
|------|---------|----------|
| `START_HERE_P0.md` | Quick start guide | You want to begin now |
| `CRITICAL_TASKS_P0.md` | Detailed plan | You need full context |
| `check_p0_status.sh` | Progress checker | Check your status |
| `README_P0_TASKS.md` | This file | Finding documentation |

### Documentation Files (in `docs/`)
| File | Purpose | Use When |
|------|---------|----------|
| `DATA_ACQUISITION_GUIDE.md` | Complete tutorial | First time with data |
| `QUICK_START_DATA_TODO.md` | Step-by-step list | You want a checklist |
| `WEEK_BY_WEEK_PLAN.md` | Competition schedule | Long-term planning |
| `DAILY_CHECKLIST.md` | Daily template | Ongoing work tracking |
| `DATA_ACQUISITION_INDEX.md` | Master index | Finding resources |

### Scripts (in `scripts/`)
| Script | Purpose | Command |
|--------|---------|---------|
| `download_hbn_data.py` | Download EEG data | Already exists ✅ |
| `verify_data_structure.py` | Validate BIDS | Already exists ✅ |
| `validate_data_statistics.py` | Check quality | Already exists ✅ |

---

## 🚀 The 4 Critical Tasks

### 1️⃣ Acquire HBN Dataset
**Time**: 1-2 days  
**Command**: `python scripts/download_hbn_data.py --subjects 2`  
**Done When**: Can load 2+ subjects

### 2️⃣ Write Core Tests
**Time**: 2-3 days  
**Goal**: 15+ tests covering data, models, metrics  
**Done When**: All tests pass, CI green

### 3️⃣ Validate Pipeline
**Time**: 1 day  
**Command**: `python scripts/verify_data_structure.py`  
**Done When**: All validations pass

### 4️⃣ Measure Inference Speed
**Time**: 4 hours  
**Command**: `pytest tests/test_inference_speed.py`  
**Done When**: <50ms confirmed

---

## 📅 This Week's Schedule

| Day | Focus | Time | Deliverable |
|-----|-------|------|-------------|
| Mon | Data setup | 2h | 2 subjects downloaded |
| Tue | Data + tests | 3h | 50 subjects + 5 tests |
| Wed | More tests | 3h | 15 total tests passing |
| Thu | Validation | 3h | All checks pass + speed test |
| Fri | Buffer | 2h | Fix issues, document |

---

## ✅ Progress Tracking

### Quick Check
```bash
# Run anytime to see status
./check_p0_status.sh
```

### Manual Check
```bash
# Check data
ls data/raw/hbn/sub-*/

# Check tests
find tests/ -name "test_*.py" | wc -l

# Check scripts
ls scripts/verify_*.py
```

### Full Status
```python
python << 'EOF'
import os
from pathlib import Path

tasks = {
    "Data Acquired": Path("data/raw/hbn").exists() and any(Path("data/raw/hbn").iterdir()),
    "Core Tests (15+)": len(list(Path("tests").glob("test_*.py"))) >= 15 if Path("tests").exists() else False,
    "Validation Scripts": Path("scripts/verify_data_structure.py").exists(),
    "Speed Benchmark": Path("tests/test_inference_speed.py").exists(),
}

print("\n🔴 P0 Tasks Status\n")
for name, done in tasks.items():
    print(f"{'✅' if done else '⭕'} {name}")

completed = sum(tasks.values())
print(f"\n📊 Progress: {completed}/4 ({completed*25}%)\n")
