# Overnight Training Plan - October 18, 2025

## Problem Identified
- Hybrid model with on-the-fly feature extraction is TOO SLOW
- Feature extraction happens on CPU during training (bottleneck)
- Would take ~10-15 hours to complete one epoch
- Need faster approach for overnight training

## Solution: Multi-Stage Approach

### Stage 1: Pre-compute Neuroscience Features (30 min)
**Why**: Extract features once, save to HDF5, reuse during training
**How**: Create preprocessing script that adds features to HDF5 files

### Stage 2: Train Baseline CNN Only (2-3 hours)
**Why**: Establish baseline performance, ensure training works
**How**: Use HybridNeuroModel with use_neuro_features=False

### Stage 3: Train Hybrid Model with Pre-computed Features (3-4 hours)
**Why**: Test if neuroscience features improve performance
**How**: Load pre-computed features from enhanced HDF5 files

### Stage 4: Compare & Select Best Model (1 hour)
**Why**: Choose best model for submission
**How**: Compare NRMSE scores, update submission.py if improved

## Implementation Plan

### ✅ Step 1: Create Feature Preprocessing Script
```bash
python scripts/preprocessing/add_neuro_features_to_hdf5.py
```
- Opens existing HDF5 files
- Extracts neuroscience features for all windows
- Adds 'neuro_features' dataset to HDF5
- Progress: 41,071 windows (~30 minutes)

### ✅ Step 2: Train Baseline CNN (No Neuro Features)
```bash
nohup python scripts/training/challenge1/train_baseline_fast.py > logs/baseline_training.log 2>&1 &
```
- Use HybridNeuroModel(use_neuro_features=False)
- Same architecture as previous best (0.26 NRMSE)
- Should take 2-3 hours for 50 epochs with early stopping
- Target: Match or beat 0.26 NRMSE

### ✅ Step 3: Train Hybrid Model (With Neuro Features)
```bash
nohup python scripts/training/challenge1/train_hybrid_fast.py > logs/hybrid_training.log 2>&1 &
```
- Use HybridNeuroModel(use_neuro_features=True)
- Load pre-computed features from HDF5
- Should take 3-4 hours
- Target: Beat baseline by 5-10%

### ✅ Step 4: Compare & Update Submission
```bash
python scripts/compare_models.py
```
- Load both models
- Compare validation NRMSE
- Update submission.py if hybrid is better
- Save best checkpoint

## Anti-Overfitting Measures (Applied to All)
- ✅ Dropout 0.4
- ✅ Weight decay 1e-4  
- ✅ Early stopping patience 10
- ✅ Learning rate 1e-4
- ✅ Train/val monitoring
- ✅ Gradient clipping

## Expected Timeline

| Stage | Task | Duration | Start | End |
|-------|------|----------|-------|-----|
| 1 | Pre-compute features | 30 min | 21:30 | 22:00 |
| 2 | Train baseline CNN | 2-3 hrs | 22:00 | 01:00 |
| 3 | Train hybrid model | 3-4 hrs | 01:00 | 05:00 |
| 4 | Compare & select | 1 hr | 05:00 | 06:00 |
| **Total** | **Complete pipeline** | **~7-8 hours** | **21:30** | **06:00** |

## Expected Results

### Conservative Estimate:
- Baseline CNN: 0.26-0.28 NRMSE (match previous)
- Hybrid Model: 0.24-0.26 NRMSE (5-8% improvement)

### Optimistic Estimate:
- Baseline CNN: 0.24-0.26 NRMSE
- Hybrid Model: 0.22-0.24 NRMSE (8-15% improvement)

## Monitoring Commands

Check training status:
```bash
watch -n 5 'ps aux | grep python | grep train'
```

Check GPU usage:
```bash
watch -n 5 rocm-smi
```

Check logs:
```bash
tail -f logs/baseline_training.log
tail -f logs/hybrid_training.log
```

Check memory:
```bash
free -h
```

## Backup Plan

If anything fails:
1. Use baseline CNN model (proven 0.26 NRMSE)
2. Keep current checkpoint
3. Don't update submission.py
4. Analyze what went wrong in morning

## Success Criteria

✅ **Minimum Success**: Baseline CNN trains successfully, matches 0.26 NRMSE
✅ **Good Success**: Hybrid model trains, shows improvement
✅ **Excellent Success**: Hybrid model beats baseline by >5%

## Files Created Tonight

1. `scripts/preprocessing/add_neuro_features_to_hdf5.py` - Feature preprocessing
2. `scripts/training/challenge1/train_baseline_fast.py` - Baseline training
3. `scripts/training/challenge1/train_hybrid_fast.py` - Hybrid training  
4. `scripts/compare_models.py` - Model comparison
5. `checkpoints/baseline_best.pth` - Baseline checkpoint
6. `checkpoints/hybrid_best.pth` - Hybrid checkpoint

## Safety Measures

- ✅ Keep existing checkpoints (don't overwrite)
- ✅ Save separate baseline and hybrid models
- ✅ Only update submission.py if confident improvement
- ✅ Log everything for morning analysis
- ✅ Memory-efficient (HDF5 keeps RAM < 4GB)
- ✅ Early stopping prevents wasted compute

---

**Status**: Ready to execute
**Start Time**: 21:30 (October 18, 2025)
**Expected Completion**: 06:00 (October 19, 2025)
**Primary Goal**: Train thoroughly, improve if possible, don't break what works

