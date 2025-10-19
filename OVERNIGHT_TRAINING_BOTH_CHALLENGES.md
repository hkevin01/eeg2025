# 🌙 OVERNIGHT TRAINING PLAN - BOTH CHALLENGES
**Date:** October 18, 2025  
**Start Time:** ~22:00  
**Expected Completion:** 06:00 (8 hours)

## 📊 STATUS SUMMARY

### Challenge 1: Response Time Prediction (CCD)
- **Data:** ✅ 4 HDF5 files ready (3.7 GB total, 39,071 windows)
- **Features:** 🔄 Currently preprocessing (R1 done, R2 in progress)
- **Model:** ❌ No trained model yet (old model missing)
- **Status:** Ready to train after preprocessing

### Challenge 2: Externalizing Behavior
- **Data:** ✅ Raw HBN data (14 subjects)
- **Model:** ✅ Existing model (weights_challenge_2_multi_release.pt)
- **Status:** Can train immediately OR improve existing model

## 🎯 OVERNIGHT STRATEGY

### OPTION A: Train Both Challenges Sequentially
**Best if you want fresh models for both challenges**

```
Timeline:
22:00-23:30 (1.5h) - Challenge 1 preprocessing completes
23:30-02:30 (3h)   - Challenge 1 baseline + hybrid training
02:30-06:00 (3.5h) - Challenge 2 training (multi-task approach)
```

### OPTION B: Focus on Challenge 1 Only (RECOMMENDED)
**Best for maximizing Challenge 1 improvement (bigger impact)**

```
Timeline:
22:00-23:30 (1.5h) - Challenge 1 preprocessing completes
23:30-02:30 (3h)   - Challenge 1 baseline training
02:30-05:30 (3h)   - Challenge 1 hybrid training (neuroscience features)
05:30-06:00 (0.5h) - Model comparison & selection
```

### OPTION C: Parallel Training (RISKY)
**Only if you have enough RAM/CPU cores**

```
Timeline:
22:00-23:30 (1.5h) - Challenge 1 preprocessing
23:30-06:00 (6.5h) - Challenge 1 + Challenge 2 training in parallel
```

## 📋 RECOMMENDED APPROACH: OPTION B

**Rationale:**
1. **Challenge 1** has no trained model yet - MUST train
2. **Challenge 2** already has a working model
3. Focus resources on biggest gap
4. Neuroscience features are Challenge 1-specific (P300, motor prep)

## 🚀 EXECUTION PLAN (OPTION B)

### Stage 1: Wait for Preprocessing ✅
**Status:** R2 currently processing (~30 min remaining)
```bash
# Monitor:
tail -f logs/feature_preprocessing.log

# Expected completion: ~22:30
```

### Stage 2: Challenge 1 Baseline Training
**Time:** 22:30 - 01:30 (3 hours)
**Purpose:** Establish baseline performance

```bash
# Will auto-start when preprocessing completes
# Or manually start:
python scripts/training/challenge1/train_baseline_fast.py
```

**Expected Results:**
- NRMSE: ~0.26 (match previous best)
- Model saved: checkpoints/challenge1_baseline_best.pth

### Stage 3: Challenge 1 Hybrid Training  
**Time:** 01:30 - 04:30 (3 hours)
**Purpose:** Test neuroscience feature improvements

```bash
# Will auto-start after baseline
# Or manually start:
python scripts/training/challenge1/train_hybrid_fast.py
```

**Expected Results:**
- Target NRMSE: 0.22-0.25 (5-15% improvement)
- Model saved: checkpoints/challenge1_hybrid_best.pth

### Stage 4: Model Selection
**Time:** 04:30 - 05:00 (30 min)
**Purpose:** Compare and select best model

```bash
# Auto-runs at end
# Creates: OVERNIGHT_TRAINING_RESULTS.md
```

## 📈 SUCCESS CRITERIA

### Minimum Success ✅
- Challenge 1 baseline trains successfully
- NRMSE ≤ 0.28 (any reasonable model)

### Good Success ⭐
- Baseline matches previous best (~0.26 NRMSE)
- Hybrid shows some improvement

### Excellent Success 🏆
- Hybrid beats baseline by >5%
- NRMSE < 0.25
- Ready for competition submission

## 🔍 MONITORING

### Quick Status Check
```bash
# Check what's running
ps aux | grep -E "(train_|add_neuro)" | grep -v grep

# Check logs
ls -lt logs/*.log | head -5

# Check GPU
rocm-smi
```

### Detailed Progress
```bash
# Preprocessing
tail -f logs/feature_preprocessing.log

# Baseline training
tail -f logs/baseline_training.log

# Hybrid training
tail -f logs/hybrid_training.log
```

### Check Results (Morning)
```bash
# View results summary
cat OVERNIGHT_TRAINING_RESULTS.md

# Check checkpoints
ls -lh checkpoints/challenge1*.pth

# Quick validation
python -c "
import torch
baseline = torch.load('checkpoints/challenge1_baseline_best.pth')
hybrid = torch.load('checkpoints/challenge1_hybrid_best.pth')
print(f'Baseline NRMSE: {baseline[\"val_nrmse\"]:.4f}')
print(f'Hybrid NRMSE: {hybrid[\"val_nrmse\"]:.4f}')
improvement = ((baseline['val_nrmse'] - hybrid['val_nrmse']) / baseline['val_nrmse']) * 100
print(f'Improvement: {improvement:+.1f}%')
"
```

## 🎯 CHALLENGE 2 PLAN (IF TIME)

If Challenge 1 finishes early (before 4am), can optionally train Challenge 2:

```bash
# Option 1: Quick validation of existing model
python scripts/training/challenge2/validate_existing.py

# Option 2: Train improved model
python scripts/training/challenge2/train_challenge2_multi_release.py
```

**Note:** Challenge 2 already has a working model, so this is lower priority.

## ⚠️ FAILURE HANDLING

### If Preprocessing Fails
- Fallback: Train baseline CNN without neuroscience features
- Still get a working Challenge 1 model

### If Baseline Training Fails
- Check logs for errors
- May need to adjust hyperparameters
- Can retry with smaller model

### If Hybrid Training Fails
- Keep baseline model
- Document why neuroscience features didn't work
- Still have working submission

## �� MORNING CHECKLIST

When you wake up, check:

1. ✅ Feature preprocessing completed
2. ✅ Baseline training completed
3. ✅ Hybrid training completed  
4. ✅ Results summary created
5. ✅ Best model selected
6. ✅ Ready to update submission.py

## 🎉 EXPECTED OUTCOME

By morning (6am), you should have:

✅ **Challenge 1:** 2 trained models (baseline + hybrid)  
✅ **Comparison:** Clear winner selected  
✅ **Results:** OVERNIGHT_TRAINING_RESULTS.md with full metrics  
✅ **Next Step:** Update submission.py if hybrid is better  

**Challenge 2:** Existing model remains (low priority tonight)

---

**RECOMMENDATION:** Go with Option B (Challenge 1 focus)  
**Current Status:** Preprocessing Stage 1 - R2 in progress  
**Next Action:** Scripts will auto-proceed when preprocessing completes
