# 📚 Data Acquisition & Validation - Complete Guide

**Created**: October 14, 2025  
**Purpose**: Get real EEG data and validate the pipeline  
**Status**: Ready to Use

---

## 🎯 Quick Start (Choose Your Path)

### Path A: I Want Step-by-Step Instructions
👉 **Start Here**: `docs/DATA_ACQUISITION_GUIDE.md`
- Complete walkthrough with screenshots
- Troubleshooting for common issues
- Expected outputs at each step

### Path B: I Want To Download Data Now
👉 **Start Here**: `scripts/download_hbn_data.py`
- Run: `python scripts/download_hbn_data.py --subjects 2`
- Downloads sample data in ~15 minutes
- Automated verification included

### Path C: I Want a Daily TODO List
👉 **Start Here**: `docs/QUICK_START_DATA_TODO.md`
- 4-phase checklist (Environment → Download → Validation → Integration)
- Time estimates for each step
- Success criteria clearly defined

### Path D: I Need a Long-Term Plan
👉 **Start Here**: `docs/WEEK_BY_WEEK_PLAN.md`
- 4-week schedule to competition
- Daily time budgets
- Risk mitigation strategies

---

## 📁 All Available Resources

### 1. **Documentation**
Located in `docs/` folder:

| File | Purpose | When to Use |
|------|---------|-------------|
| `DATA_ACQUISITION_GUIDE.md` | Complete tutorial | First time setup |
| `QUICK_START_DATA_TODO.md` | Quick checklist | When you need speed |
| `WEEK_BY_WEEK_PLAN.md` | Long-term schedule | Planning competition |
| `DAILY_CHECKLIST.md` | Daily task template | Every day |
| `DATA_ACQUISITION_INDEX.md` | This file! | Finding resources |

### 2. **Scripts**
Located in `scripts/` folder:

| Script | Purpose | Command |
|--------|---------|---------|
| `download_hbn_data.py` | Download EEG data | `python scripts/download_hbn_data.py` |
| `verify_data_structure.py` | Validate BIDS format | `python scripts/verify_data_structure.py` |
| `validate_data_statistics.py` | Check data quality | `python scripts/validate_data_statistics.py` |

### 3. **Configuration**
Located in `config/` folder:

| File | Purpose | Edit When |
|------|---------|-----------|
| `data_sources.yaml` | Data URLs and paths | Changing data source |

---

## 🚀 Recommended Workflow

### Day 1: Setup (2 hours)
```bash
# 1. Read the guide
cat docs/DATA_ACQUISITION_GUIDE.md

# 2. Install dependencies
pip install mne mne-bids boto3 requests tqdm

# 3. Create directories
mkdir -p data/{raw,processed,cache}

# 4. Test with 1 subject
python scripts/download_hbn_data.py --subjects 1 --verify
```

### Day 2: Validation (1 hour)
```bash
# 1. Verify structure
python scripts/verify_data_structure.py --data-dir data/raw/hbn

# 2. Check statistics
python scripts/validate_data_statistics.py --data-dir data/raw/hbn

# 3. Test loading
python -c "from src.dataio.hbn_dataset import HBNDataset; print('✓ OK')"
```

### Day 3: Integration (1 hour)
```bash
# 1. Run data loading tests
pytest tests/test_data_loading.py -v

# 2. Test preprocessing
python scripts/test_preprocessing_pipeline.py --subject NDARAA536PTU

# 3. Validate end-to-end
python scripts/test_full_pipeline.py --challenge challenge1
```

---

## 🎯 Success Criteria

After following this guide, you should have:

### ✅ Environment
- [x] All Python packages installed
- [x] Data directories created
- [x] Scripts executable

### ✅ Data
- [x] At least 2 subjects downloaded
- [x] BIDS structure validated
- [x] Data statistics checked

### ✅ Pipeline
- [x] Can load EEG data
- [x] Preprocessing works
- [x] Model inputs correct shape

### ✅ Testing
- [x] Data loading tests pass
- [x] Preprocessing tests pass
- [x] Integration tests pass

---

## ⚠️ Troubleshooting Quick Reference

### Problem: Download fails
**Solution**: Check `docs/DATA_ACQUISITION_GUIDE.md` Section 6.1

### Problem: MNE can't read files
**Solution**: Check `docs/DATA_ACQUISITION_GUIDE.md` Section 6.2

### Problem: Out of memory
**Solution**: Check `docs/DATA_ACQUISITION_GUIDE.md` Section 6.3

### Problem: Wrong data shape
**Solution**: Check `docs/DATA_ACQUISITION_GUIDE.md` Section 6.4

---

## 📊 Progress Tracking

Use this simple command to check your status:

```bash
# Daily status check
python << 'EOF'
import os
print("\n=== Data Acquisition Status ===\n")

checks = {
    "Dependencies": os.system("python -c 'import mne, boto3' 2>/dev/null") == 0,
    "Data Directory": os.path.exists("data/raw/hbn"),
    "Scripts Ready": os.path.exists("scripts/download_hbn_data.py"),
    "Tests Written": os.path.exists("tests/test_data_loading.py"),
}

for name, status in checks.items():
    icon = "✅" if status else "❌"
    print(f"{icon} {name}")

print("\n" + "="*32 + "\n")
