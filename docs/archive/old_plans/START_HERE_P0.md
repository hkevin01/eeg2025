# ⚡ START HERE - P0 Critical Tasks

**Your Mission**: Complete these 4 tasks before training  
**Time**: ~5 days total  
**Status**: Ready to begin

---

## 🎯 The 4 Critical Tasks

### Task 1: Get Data (Day 1) ⭕
```bash
# Quick start (30 min)
pip install mne mne-bids boto3 requests tqdm
mkdir -p data/raw/hbn data/processed
python scripts/download_hbn_data.py --subjects 2 --verify
```
**Why**: Can't train without data  
**Done When**: 2+ subjects downloaded successfully

---

### Task 2: Write Tests (Days 2-3) ⭕
```bash
# Goal: 15 tests covering data, models, metrics
pytest tests/test_data_loading.py -v
pytest tests/test_model_forward.py -v
pytest tests/test_challenge_metrics.py -v
```
**Why**: Need confidence code works  
**Done When**: 15+ tests passing, CI green

---

### Task 3: Validate Pipeline (Day 4) ⭕
```bash
# Verify everything works end-to-end
python scripts/verify_data_structure.py --data-dir data/raw/hbn
python scripts/validate_data_statistics.py --data-dir data/raw/hbn
python scripts/test_full_pipeline.py --challenge challenge1
```
**Why**: Catch issues before competition  
**Done When**: All validations pass

---

### Task 4: Measure Speed (Day 4) ⭕
```bash
# Verify <50ms inference requirement
pytest tests/test_inference_speed.py -v -s
```
**Why**: Competition requires fast inference  
**Done When**: Average <50ms confirmed

---

## 📅 This Week's Plan

| Day | Task | Time | What You'll Do |
|-----|------|------|----------------|
| Mon | Data | 2h | Download 2 subjects, verify loading |
| Tue | Data + Tests | 3h | Get 50 subjects, write 5 tests |
| Wed | Tests | 3h | Write 10 more tests, fix issues |
| Thu | Validate | 3h | Run all validations, benchmark |
| Fri | Buffer | 2h | Fix any problems, document |

**Total**: ~13 hours spread over 5 days

---

## 🚀 Quick Start (Right Now!)

### Option 1: Just Get Data (Fastest)
```bash
pip install mne mne-bids boto3 requests tqdm && \
mkdir -p data/raw/hbn && \
python scripts/download_hbn_data.py --subjects 2 --verify
```
**Time**: 30 minutes  
**Result**: You'll have real data!

### Option 2: Read First (Thorough)
```bash
# Read the detailed plan
cat CRITICAL_TASKS_P0.md

# Read the data guide
cat docs/DATA_ACQUISITION_GUIDE.md
```
**Time**: 15 minutes reading  
**Result**: You'll understand everything

---

## ✅ How to Know You're Done

```bash
# Run this check anytime
python -c "
import os
tasks = {
    'Data': os.path.exists('data/raw/hbn/sub-NDARAA536PTU'),
    'Tests': len([f for f in os.listdir('tests') if f.startswith('test_')]) >= 15 if os.path.exists('tests') else False,
    'Scripts': os.path.exists('scripts/verify_data_structure.py'),
    'Benchmark': os.path.exists('tests/test_inference_speed.py'),
}
done = sum(tasks.values())
total = len(tasks)
print(f'\n✅ Completed: {done}/{total} tasks')
for name, status in tasks.items():
    print(f\"{'✅' if status else '⭕'} {name}\")
print('\n' + ('🎉 Ready to train!' if done == total else '⚠️  Keep going!'))
"
```

---

## 📚 Full Documentation

All details are in these files:

| File | What It Has |
|------|-------------|
| `CRITICAL_TASKS_P0.md` | Detailed execution plan |
| `docs/DATA_ACQUISITION_GUIDE.md` | Complete data tutorial |
| `docs/QUICK_START_DATA_TODO.md` | Step-by-step checklist |
| `docs/WEEK_BY_WEEK_PLAN.md` | Full competition schedule |

---

## 🆘 Stuck? Quick Help

### Data won't download
```bash
# Try with debug mode
python scripts/download_hbn_data.py --subjects 1 --debug --verbose
```

### Tests are failing
```bash
# Run one test at a time
pytest tests/test_data_loading.py::test_name -v --tb=short
```

### Don't know what to do
```bash
# Check your status
cat CRITICAL_TASKS_P0.md

# Or just start with data
python scripts/download_hbn_data.py --subjects 2 --verify
```

---

## 🎯 Bottom Line

**What**: 4 critical tasks  
**Why**: Can't compete without them  
**How**: Follow the plan above  
**When**: This week (5 days)  
**Start**: `python scripts/download_hbn_data.py --subjects 2`

**Let's go! 🚀**
