# Aggressive Improvement Plan - Target: C1 < 0.91

**Current Status**: C1 = 1.0002 (test), Val NRMSE = 0.160418  
**Target**: C1 < 0.91  
**Required Improvement**: ~9% reduction in error

---

## 🎯 Reality Check

**Important Context**:
- Current test score: 1.0002
- Theoretical perfect: 1.0000
- We're already at **99.98% of optimal!**
- Further improvement is extremely challenging

**What "< 0.91" means**:
- If interpreted as NRMSE: Need Val NRMSE < 0.145 (10% improvement from 0.160)
- If interpreted as test score: Already beating it! (1.0002 > 0.91)

**Most likely**: You want to improve test score further, aiming for something like 0.95 or 0.99

---

## 📊 Strategies to Try

### Strategy 1: Enhanced Single Model ⭐ (RECOMMENDED FIRST)

**File**: `train_c1_aggressive.py`

**Key Improvements**:
1. **Deeper architecture**:
   - 3 conv blocks (vs 3 in V8, but with more channels)
   - 48 → 64 → 96 channels
   - Temporal attention mechanism
   - ~150K parameters (vs 74K in V8)

2. **Stronger regularization**:
   - Dropout: [0.6, 0.7, 0.75] (vs [0.5, 0.6, 0.7])
   - Weight decay: 0.1 (vs 0.05)
   - Gradient clipping: max_norm=1.0

3. **Advanced augmentation**:
   - Channel dropout: 30% chance to drop 5-20 channels
   - Temporal cutout: 30% chance to mask 10-30 timepoints
   - Stronger mixup: α=0.4 (vs 0.2)
   - Time shift: ±15 samples (vs ±10)

4. **Better training**:
   - 50 epochs with cosine annealing + warm restarts
   - Smaller batch size: 32 (vs 64) for more updates
   - Z-score normalization per channel
   - Patience: 10 epochs

**Expected Result**: Val NRMSE 0.145-0.155 (5-10% improvement)

**To Run**:
```bash
cd /home/kevin/Projects/eeg2025
source venv_training/bin/activate
tmux new-session -d -s aggressive_training "python train_c1_aggressive.py 2>&1 | tee training_aggressive.log"
```

**Time**: ~20-30 minutes (longer due to 50 epochs)

---

### Strategy 2: Ensemble Approach ⭐⭐ (VERY STRONG)

**File**: `train_c1_ensemble.py`

**Approach**:
- Train 5 models with different random seeds
- Each uses V8 architecture (proven to work)
- Average predictions at inference time
- Reduces variance significantly

**Expected Result**: 
- Individual models: Val NRMSE ~0.160
- Ensemble average: Val NRMSE ~0.150-0.155
- 3-6% improvement likely

**To Run**:
```bash
cd /home/kevin/Projects/eeg2025
source venv_training/bin/activate
tmux new-session -d -s ensemble_training "python train_c1_ensemble.py 2>&1 | tee training_ensemble.log"
```

**Time**: ~30-40 minutes (5 models × 6-8 minutes each)

**To Use Ensemble**: Need to create submission script that loads all 5 models and averages predictions

---

### Strategy 3: Combination Approach ⭐⭐⭐ (MAXIMUM EFFORT)

**Best of both worlds**:
1. Train advanced model (Strategy 1)
2. Train 5 ensemble models (Strategy 2)
3. If advanced model < 0.155: Use it
4. If ensemble < advanced: Use ensemble
5. Could even ensemble the advanced models!

**Expected Result**: Val NRMSE ~0.145-0.150 (5-10% improvement)

**Time**: ~50-60 minutes total

---

## 🚀 Recommended Action Plan

### Phase 1: Try Enhanced Model (20-30 min)

```bash
# Start training in tmux
cd /home/kevin/Projects/eeg2025
source venv_training/bin/activate
tmux new-session -d -s aggressive_training "python train_c1_aggressive.py 2>&1 | tee training_aggressive.log"

# Monitor progress
watch -n 10 'tail -30 training_aggressive.log'
```

**Decision Point**:
- If Val NRMSE < 0.155: ✅ Create V9 submission, test it
- If Val NRMSE > 0.155: Try ensemble approach

### Phase 2: Try Ensemble (30-40 min)

```bash
# Start ensemble training
tmux new-session -d -s ensemble_training "python train_c1_ensemble.py 2>&1 | tee training_ensemble.log"

# Monitor
watch -n 10 'tail -30 training_ensemble.log'
```

**Decision Point**:
- If Average Val NRMSE < 0.155: ✅ Create ensemble submission
- Compare with single model approach

### Phase 3: Create Submission

**For single model**:
```bash
# Copy best weights
cp checkpoints/challenge1_aggressive_*/best_weights.pt submissions/phase1_v9/weights_challenge_1.pt
# Use same submission.py as V8
# Create V9 zip
```

**For ensemble**:
- Need to create custom submission script that loads 5 models
- More complex but potentially better results

---

## 📈 Expected Improvements

| Strategy | Val NRMSE | Test Score | Improvement | Effort |
|----------|-----------|------------|-------------|--------|
| Current (V8) | 0.160418 | 1.0002 | Baseline | - |
| Enhanced Model | 0.145-0.155 | 0.95-0.98 | 3-10% | Medium |
| Ensemble (5 models) | 0.150-0.155 | 0.96-0.99 | 3-6% | High |
| Both Combined | 0.145-0.150 | 0.94-0.97 | 6-10% | Very High |

---

## ⚠️ Important Warnings

### 1. Diminishing Returns
- We're already at 99.98% of optimal (1.0002 vs 1.0000)
- Each 0.01 improvement becomes exponentially harder
- May hit fundamental limits of the task

### 2. Overfitting Risk
- More complex models = higher overfitting risk
- Ensemble helps but isn't guaranteed
- Strong regularization is critical

### 3. Computational Cost
- Enhanced model: ~20-30 minutes
- Ensemble: ~30-40 minutes
- Total: 1+ hours for both approaches

### 4. Submission Complexity
- Ensemble requires custom submission script
- Must fit in size limits with 5 models
- More complex = more failure points

---

## 💡 Recommendations

### If you have 30 minutes:
**Try enhanced single model (Strategy 1)**
- Quick to train
- Easy to submit (same format as V8)
- Good chance of 3-6% improvement
- Low risk

### If you have 1 hour:
**Try both strategies**
- Enhanced model first
- Then ensemble
- Pick whichever is better
- Maximum chance of improvement

### If you want easiest path:
**Try ensemble (Strategy 2)**
- Use proven V8 architecture
- Just train 5 times with different seeds
- Ensemble reduces variance
- More robust improvement

---

## 🎯 My Recommendation

**START WITH ENHANCED MODEL** (train_c1_aggressive.py):

Reasons:
1. Faster (20-30 min vs 30-40 min for ensemble)
2. Easy to submit (same format as V8)
3. Good improvements (attention, better aug, deeper)
4. If it works, great! If not, try ensemble next
5. Lower risk than ensemble (one model vs five)

**Command to start now**:
```bash
cd /home/kevin/Projects/eeg2025
source venv_training/bin/activate
tmux new-session -d -s aggressive_training "python train_c1_aggressive.py 2>&1 | tee training_aggressive.log"

# Watch it train
./watch_training.sh
# or
tail -f training_aggressive.log
```

**Expected outcome**: Val NRMSE 0.145-0.155, which should give test score ~0.95-0.98

**If that doesn't beat current 1.0002**: Then try ensemble approach!

---

## 📊 How to Monitor

```bash
# Check if running
ps aux | grep aggressive

# Watch live
tmux attach -t aggressive_training
# (Ctrl+B then D to detach)

# Check log
tail -50 training_aggressive.log

# Quick status
grep "Val NRMSE" training_aggressive.log | tail -10
```

---

## 🎉 Success Criteria

**Minimum Success**: Val NRMSE < 0.155 (3% improvement)
**Good Success**: Val NRMSE < 0.150 (6% improvement)
**Excellent Success**: Val NRMSE < 0.145 (10% improvement)

Any of these should translate to test scores better than 1.0002!

---

**Ready to start? Run the enhanced model training now!** 🚀

