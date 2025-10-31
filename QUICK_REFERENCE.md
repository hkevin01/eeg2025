# Quick Reference Guide

## 📍 Where to Find Things

### Active Work
- **Source Code**: `src/`
- **Training Scripts**: `scripts/training/`
- **Monitoring Scripts**: `scripts/monitoring/`
- **Tests**: `tests/`
- **Configs**: `configs/`

### Models & Results
- **Checkpoints**: `checkpoints/`
  - C1 V8: `c1_v8_best.pt`
  - C2 Phase 1: `c2_phase1_best.pt`
- **Submissions**: `submissions/phase1_v9/`
- **Current Package**: `phase1_v9_submission.zip` (root)

### Documentation
- **Project Docs**: `docs/`
- **Strategy**: `docs/strategy/`
- **Structure**: `DIRECTORY_INDEX.md`
- **Organization**: `PROJECT_ORGANIZATION_COMPLETE.md`

### Historical
- **Old Logs**: `logs/archive/`
- **Status Reports**: `archive/status_reports/`
- **Submission History**: `archive/submissions/`

## 🔧 Common Commands

### Training
```bash
# Train C1 model
python scripts/training/train_c1_*.py

# Train C2 model
python scripts/training/train_c2_*.py

# Monitor training
./scripts/monitoring/watch_training.sh
```

### Testing
```bash
# Run simple validation
python tests/simple_validation.py

# Run full test suite
pytest tests/
```

### Submission
```bash
# Create submission
cd submissions/phase1_v9
python submission.py

# Verify submission
./scripts/monitoring/verify_*.sh
```

### Cleanup
```bash
# Run cleanup script
python scripts/cleanup_project.py
```

## 📊 Current State

### Models
| Model | Architecture | Val Loss | Score | Status |
|-------|-------------|----------|-------|--------|
| C1 V8 | CompactResponseTimeCNN | 0.079314 | 1.0002 | ✅ Optimal |
| C2 Phase 1 | EEGNeX | 0.252475 | 1.0055-1.0075* | ✅ Ready |

*Expected

### Submissions
- **Current**: V9 (`phase1_v9_submission.zip`)
- **Previous**: V8 (score: 1.0044)
- **Expected**: 1.0028-1.0038

## 🚀 Quick Actions

### Check Status
```bash
# Check training logs
tail -f logs/c2_phase1_*.log

# Check checkpoint
ls -lh checkpoints/
```

### Submit to Competition
```bash
# 1. Verify submission
cd submissions/phase1_v9
python submission.py

# 2. Upload phase1_v9_submission.zip
```

### Clean Up
```bash
# Remove cache
find . -type d -name "__pycache__" -exec rm -rf {} +

# Organize files
python scripts/cleanup_project.py
```

## 📝 File Naming Conventions

### Scripts
- Training: `train_<challenge>_<variant>.py`
- Submission: `submission_<variant>.py`
- Monitoring: `<action>_<target>.sh`

### Logs
- Training: `training_<model>_<timestamp>.log`
- C2: `c2_phase1_<device>_<timestamp>.log`

### Checkpoints
- Format: `<challenge>_<variant>_best.pt`
- C1: `c1_v8_best.pt`
- C2: `c2_phase1_best.pt`

### Documentation
- Status: `<TOPIC>_STATUS.md`
- Complete: `<TASK>_COMPLETE.md`
- Summary: `<EVENT>_SUMMARY.md`

## 🎯 Next Steps

1. ✅ Project organized
2. ⏳ Review structure
3. ⏳ Commit changes
4. ⏳ Submit V9
5. ⏳ Monitor results

---

**Last Updated**: October 31, 2025  
**Organization Script**: `scripts/cleanup_project.py`  
**Status**: Ready for production
