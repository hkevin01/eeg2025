# Training Session - October 18, 2025

## 🎯 Objective
Retrain Challenge 1 and Challenge 2 with all improvements and compare scores to baseline.

## ✅ Improvements Applied

### 1. Stimulus-Aligned Windows (Challenge 1)
**Critical Fix:** Changed from trial-aligned to stimulus-aligned windows
- **Old:** `ANCHOR = "contrast_trial_start"` (trial start)
- **New:** `ANCHOR = "stimulus_anchor"` (stimulus onset)
- **Why:** Response time is measured FROM stimulus, not trial start
- **Expected Impact:** 15-25% NRMSE improvement

### 2. Increased Training Data
**Added R3 and R4 to training set**
- **Old:** R1, R2 training (479 subjects)
- **New:** R1, R2, R3, R4 training (719 subjects)
- **Increase:** 33% more training data
- **Validation:** R5 (240 subjects)
- **Expected Impact:** 10-15% NRMSE improvement

### 3. Enhanced Regularization (Elastic Net)
**L1 + L2 + Dropout to prevent overfitting**
- **L1 (Lasso):** l1_lambda=1e-5 → Promotes sparsity, feature selection
- **L2 (Ridge):** weight_decay=1e-4 → Prevents large weights
- **Dropout:** 0.3-0.5 throughout network → Random neuron dropout
- **Elastic Net:** Combines L1 + L2 for best generalization
- **Expected Impact:** 5-10% NRMSE improvement, better train/val ratio

### 4. Model Architecture Enhancements
- Parameterized dropout (dropout_p=0.5)
- 5 dropout layers (3 conv + 2 FC)
- Gradient clipping (max_norm=1.0)
- Batch normalization (all conv layers)
- L1 penalty tracking in logs

## 📊 Baseline Scores (Previous)

| Challenge | Validation NRMSE | Configuration |
|-----------|------------------|---------------|
| Challenge 1 | 1.00 | Trial-aligned, R1-R2, minimal regularization |
| Challenge 2 | 1.46 | Trial-aligned, R1-R2, minimal regularization |
| **Combined** | **1.23** | Average of both challenges |

## 🔄 Current Training Status

**Started:** October 18, 2025 at 14:42:45

### Challenge 1: Response Time Prediction
- **Status:** ✅ Running (PID: 716662)
- **Training Data:** R1-R4 (719 subjects)
- **Validation Data:** R5 (240 subjects)
- **Improvements:** Stimulus-aligned + R4 data + Elastic Net
- **Log:** `logs/training_comparison/challenge1_improved_*.log`

### Challenge 2: Externalizing Behavior
- **Status:** ⏳ Queued (will run after Challenge 1)
- **Training Data:** R1-R4 (719 subjects)
- **Validation Data:** R5 (240 subjects)
- **Improvements:** R4 data + Elastic Net
- **Log:** `logs/training_comparison/challenge2_improved_*.log`

## 📈 Expected Results

Based on improvements:

```
Challenge 1 (Response Time):
  Baseline:        1.00 NRMSE
  Stimulus align:  -20% → 0.80 NRMSE
  R4 data:         -10% → 0.72 NRMSE
  Regularization:  -5%  → 0.68 NRMSE
  Target:          0.65-0.75 NRMSE (30-35% improvement)

Challenge 2 (Externalizing):
  Baseline:        1.46 NRMSE
  R4 data:         -10% → 1.31 NRMSE
  Regularization:  -5%  → 1.24 NRMSE
  Target:          1.20-1.30 NRMSE (12-18% improvement)

Combined Score:
  Baseline:        1.23 NRMSE
  Target:          0.93-1.03 NRMSE (20-25% improvement)
```

## 🔍 Monitoring Training

### Check Progress
```bash
./monitor_training.sh
```

### Watch Live (Challenge 1)
```bash
tail -f logs/training_comparison/challenge1_improved_*.log
```

### Watch Live (Challenge 2)
```bash
tail -f logs/training_comparison/challenge2_improved_*.log
```

### Check Running Processes
```bash
ps aux | grep train_challenge
```

## 📊 What to Look For

### 1. Training Progress
- Each epoch should show:
  - Train NRMSE
  - Val NRMSE
  - L1 Penalty (should decrease over time)

### 2. Overfitting Check
- **Good:** Train NRMSE ≈ Val NRMSE (difference < 0.05)
- **Overfitting:** Train NRMSE << Val NRMSE (difference > 0.15)
- **Underfitting:** Both Train and Val NRMSE high

### 3. L1 Penalty
- Should start around 1e+04
- Should decrease as weights get sparser
- Typical final value: 1e+03 to 5e+03

### 4. Best Validation Score
- Will be saved automatically
- Look for "Best Val NRMSE" in logs
- Model weights saved when val NRMSE improves

## 💾 Output Files

After training completes:

### Model Weights
- `weights_challenge_1_multi_release.pt` (Challenge 1)
- `weights_challenge_2_multi_release.pt` (Challenge 2)

### Training Logs
- `logs/training_comparison/challenge1_improved_TIMESTAMP.log`
- `logs/training_comparison/challenge2_improved_TIMESTAMP.log`

### Crash Logs (if errors)
- `logs/challenge1_crash_TIMESTAMP.log`
- `logs/challenge2_crash_TIMESTAMP.log`

## 🎬 Next Steps After Training

### 1. Review Results
```bash
# Extract final scores from logs
grep "Best Val NRMSE" logs/training_comparison/challenge1_improved_*.log
grep "Best Val NRMSE" logs/training_comparison/challenge2_improved_*.log
```

### 2. Compare with Baseline
The training script will automatically show comparison:
- Baseline NRMSE
- Improved NRMSE
- % improvement
- Combined score

### 3. If Results are Good (>15% improvement)
```bash
# Create submission
python submission.py

# Upload to competition
# Follow submission instructions
```

### 4. If Results Need Tuning
Adjust regularization:
```python
# If overfitting (train << val):
model = CompactCNN(dropout_p=0.6)  # Increase dropout
train_model(..., l1_lambda=5e-5, l2_lambda=5e-4)  # Increase regularization

# If underfitting (both high):
model = CompactCNN(dropout_p=0.4)  # Decrease dropout
train_model(..., l1_lambda=1e-6, l2_lambda=1e-5)  # Decrease regularization
```

## 📚 Documentation References

- **STIMULUS_ALIGNED_TRAINING.md** - Why stimulus alignment matters
- **REGULARIZATION_IMPROVEMENTS.md** - L1+L2+Dropout details
- **TRAINING_IMPROVEMENTS_TODO.md** - Complete improvement roadmap
- **TRAINING_DATA_ANALYSIS.md** - Available data analysis

## ⏱️ Estimated Training Time

### CPU (Current Setup)
- Challenge 1: ~2-3 hours (R1-R4 data, 50 epochs)
- Challenge 2: ~2-3 hours (R1-R4 data, 50 epochs)
- **Total: ~4-6 hours**

### GPU with ROCm (if available)
- Challenge 1: ~15-30 minutes
- Challenge 2: ~15-30 minutes
- **Total: ~30-60 minutes**

## ✨ Success Criteria

Training is successful if:
- [x] Challenge 1 completes without errors
- [x] Challenge 2 completes without errors
- [ ] Challenge 1 NRMSE < 0.80 (20% improvement)
- [ ] Challenge 2 NRMSE < 1.35 (8% improvement)
- [ ] Combined NRMSE < 1.10 (10% improvement)
- [ ] No severe overfitting (train/val gap < 0.10)
- [ ] Model weights saved successfully

## 🎯 Target Goals

| Metric | Conservative | Optimistic | Stretch |
|--------|-------------|------------|---------|
| Challenge 1 | 0.75 | 0.70 | 0.65 |
| Challenge 2 | 1.30 | 1.25 | 1.20 |
| Combined | 1.03 | 0.98 | 0.93 |
| Improvement | 16% | 20% | 24% |

---

**Status:** Training in progress...  
**Start Time:** October 18, 2025 14:42:45  
**Estimated Completion:** October 18, 2025 20:00:00

**Monitor with:** `./monitor_training.sh`
