# Project Directory Structure

**NeurIPS 2025 EEG Foundation Challenge**  
**Last Updated:** October 16, 2025

---

## 📁 Root Directory

**Active Files:**
- `README.md` - Main project documentation
- `TODO.md` - Current task list and action items
- `METHODS_DOCUMENT.md` - Competition methods description (source)
- `METHODS_DOCUMENT.pdf` - **PDF for submission** (generated from .md)
- `submission.py` - Main submission script
- `requirements.txt` - Python dependencies

**Configuration:**
- `pyproject.toml` - Project metadata
- `setup.py` - Package setup
- `Makefile` - Build automation

---

## 📂 Key Directories

### `/scripts` - Training Scripts
```
scripts/
├── train_challenge1_multi_release.py  ← Challenge 1 (Response Time) - ACTIVE
├── train_challenge2_multi_release.py  ← Challenge 2 (Externalizing) - ACTIVE
├── train_challenge1_response_time.py  (old, single-release)
├── train_challenge2_externalizing.py  (old, single-release)
└── ...other experimental scripts
```

**Current Training:**
- `train_challenge1_multi_release.py` - R1-R3 train, R4 validation
- `train_challenge2_multi_release.py` - R1-R3 train, R4 validation

### `/weights` - Model Weights
```
weights/
├── weights_challenge_1_multi_release.pt  ← Current Challenge 1 weights
├── weights_challenge_2_multi_release.pt  ← Current Challenge 2 weights
└── archive/
    ├── weights_challenge_1.pt           (old, single-release)
    └── weights_challenge_2.pt           (old, single-release)
```

**For Submission:**
- Use `weights_challenge_*_multi_release.pt` files

### `/logs` - Training Logs
```
logs/
├── challenge1_training_v7_R4val.log  ← Current Challenge 1 log
├── challenge2_training_v8_R4val.log  ← Current Challenge 2 log
└── ...previous versions (v1-v6)
```

**Monitor Active Training:**
```bash
tail -f logs/challenge1_training_v7_R4val.log
tail -f logs/challenge2_training_v8_R4val.log
```

### `/docs` - Documentation
```
docs/
├── DIRECTORY_STRUCTURE.md          ← This file
├── CRITICAL_ISSUE_VALIDATION.md    ← Key bug discovery
├── FINAL_STATUS_FIXED.md           ← Current training status
├── plans/
│   └── PHASE2_TASK_SPECIFIC_PLAN.md  ← Advanced features plan
├── archive/
│   └── ...old status documents
└── ...other guides
```

**Important Docs:**
- `CRITICAL_ISSUE_VALIDATION.md` - Why we changed from R5 to R4 validation
- `PHASE2_TASK_SPECIFIC_PLAN.md` - P300/spectral features implementation

### `/data` - Dataset Cache
```
data/
└── raw/
    └── eegdash_cache/  ← Cached EEG datasets (R1-R5)
```

**Storage:** ~50GB cached data, don't delete!

### `/tests` - Unit Tests
```
tests/
└── ...test files
```

### `/notebooks` - Jupyter Notebooks
```
notebooks/
└── ...exploratory analysis
```

---

## �� Configuration Directories

### `.github` - GitHub Actions
CI/CD workflows and issue templates

### `.vscode` - VS Code Settings
Editor configuration

### `config/` - Application Configs
Competition and model configurations

---

## 🎯 Critical Files for Submission

**Required Files:**
1. ✅ `submission.py` (root)
2. ✅ `weights/weights_challenge_1_multi_release.pt`
3. ✅ `weights/weights_challenge_2_multi_release.pt`
4. ⏳ `METHODS_DOCUMENT.pdf` (to be generated)

**Create Submission Package:**
```bash
cd /home/kevin/Projects/eeg2025

# Generate PDF
pandoc METHODS_DOCUMENT.md -o METHODS_DOCUMENT.pdf

# Create submission zip
zip submission_multi_release_final.zip \
    submission.py \
    weights/weights_challenge_1_multi_release.pt \
    weights/weights_challenge_2_multi_release.pt \
    METHODS_DOCUMENT.pdf
```

---

## 📊 Current Training Status

**Location:** `/home/kevin/Projects/eeg2025`

**Active Processes:**
- Challenge 1: `logs/challenge1_training_v7_R4val.log`
- Challenge 2: `logs/challenge2_training_v8_R4val.log`

**Check Status:**
```bash
# Quick check
ps aux | grep "[p]ython3 scripts/train" | wc -l

# View latest NRMSE
tail -100 logs/challenge1_training_v7_R4val.log | grep "NRMSE"
tail -100 logs/challenge2_training_v8_R4val.log | grep "NRMSE"
```

---

## 🗂️ File Organization Rules

### Keep in Root:
- Active task lists (TODO.md)
- Main documentation (README.md)
- Submission files (submission.py, METHODS_DOCUMENT.*)
- Configuration files

### Keep in `/scripts`:
- Current training scripts (*_multi_release.py)
- Can archive old single-release scripts

### Keep in `/weights`:
- Current multi-release weights
- Archive old weights in `/weights/archive`

### Keep in `/docs`:
- Current status documents
- Active plans in `/docs/plans`
- Archive old docs in `/docs/archive`

### Keep in `/logs`:
- Current training logs (v7, v8)
- Can delete very old logs (v1-v3) if needed

---

## 🧹 Cleanup Commands

**Safe Cleanup (archive, don't delete):**
```bash
# Archive old logs
mkdir -p logs/archive
mv logs/*_v[1-5].log logs/archive/

# Archive old scripts
mkdir -p scripts/archive
mv scripts/train_challenge*_improved.py scripts/archive/
```

**Check Disk Usage:**
```bash
du -h --max-depth=1 | sort -hr | head -10
```

---

## 📞 Quick Reference

**Project Root:** `/home/kevin/Projects/eeg2025`  
**Virtual Env:** `source venv/bin/activate`  
**Competition:** https://www.codabench.org/competitions/4287/  
**Repository:** https://github.com/hkevin01/eeg2025

**Need Help?**
- Check `TODO.md` for current tasks
- Check `docs/FINAL_STATUS_FIXED.md` for latest status
- Check `docs/plans/PHASE2_TASK_SPECIFIC_PLAN.md` for future work

