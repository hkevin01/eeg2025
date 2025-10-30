# Challenge 1: Improvement Strategy

## 🎯 Goal
Beat the untrained baseline (1.0015) through **careful training with anti-overfitting measures**.

## 📊 Current Situation

| Approach | Score | Issue |
|----------|-------|-------|
| Untrained (random init) | 1.0015 ⭐ | BEST - but can we do better? |
| Trained (various) | 1.10-1.62 | Overfitting! |
| V6 (TCN trained) | 1.6262 | Severe overfitting |

## 🔬 Root Cause Analysis

### Why Training Fails

1. **Overfitting** (Primary Issue)
   - Model memorizes training data
   - Doesn't generalize to test set
   - Test distribution != training distribution

2. **Weak Regularization**
   - Previous config: dropout 0.1-0.3 (TOO LOW!)
   - Weight decay: 0.01 (TOO LOW!)
   - Training too long: 100 epochs (TOO MANY!)

3. **Subject Leakage**
   - Training/validation may share subjects
   - Model learns subject-specific patterns
   - Doesn't generalize to new subjects

## 💡 Solution: Multi-Pronged Anti-Overfitting Strategy

### Strategy 1: Strong Regularization ⭐ PRIMARY

**Changes:**
- ✅ Dropout: 0.5, 0.6, 0.7 (was: 0.3, 0.4, 0.5)
- ✅ Weight decay: 0.05 (was: 0.01)
- ✅ Max epochs: 15 (was: 100)
- ✅ Early stopping: patience 5
- ✅ Gradient clipping: 1.0

**Rationale:**
- Strong dropout forces network to learn robust features
- Can't rely on any single neuron
- Must learn distributed representations

### Strategy 2: Data Augmentation

**Techniques:**
- Time shift: ±5% (subtle temporal jitter)
- Time stretch: 0.95-1.05 (small time warping)
- Amplitude scale: 0.9-1.1 (normalize variations)
- Add noise: σ=0.01 (robustness)
- Channel dropout: 10% (channel invariance)

**Rationale:**
- Increases effective dataset size
- Forces model to learn invariant features
- Prevents memorization of exact patterns

### Strategy 3: Mixup Augmentation

**Method:**
```python
x_mixed = λ * x_i + (1-λ) * x_j
y_mixed = λ * y_i + (1-λ) * y_j
```

**Parameters:**
- α = 0.2 (conservative mixing)

**Rationale:**
- Encourages linear behavior between examples
- Reduces overconfidence
- Improves calibration

### Strategy 4: Subject-Aware Validation

**Approach:**
- Split by subject (zero overlap!)
- 20% subjects for validation
- Test on completely unseen subjects

**Rationale:**
- Mimics competition test setup
- Prevents subject-specific overfitting
- More realistic performance estimate

### Strategy 5: Early Stopping

**Configuration:**
- Monitor: val_nrmse
- Patience: 5 epochs
- Min delta: 0.001

**Rationale:**
- Stop before overfitting starts
- Save best model (not last)
- Prevent overtraining

### Strategy 6: Learning Rate Schedule

**Method:**
- Cosine annealing
- T_max: 15 epochs
- η_min: 1e-6

**Rationale:**
- Smooth convergence
- Fine-tune at end
- Avoid oscillation

### Strategy 7: Robust Loss Function

**Loss:**
- SmoothL1Loss (Huber loss)

**Rationale:**
- Less sensitive to outliers
- More stable gradients
- Better generalization

## 📈 Training Protocol

### Phase 1: Single Model (Quick Test)
```bash
python scripts/train_challenge1_improved.py --epochs 15 --patience 5
```

**Expected outcome:**
- Training converges in 5-10 epochs
- Val NRMSE: 0.15-0.18
- Test score: 0.95-1.00 (better than 1.0015!)

### Phase 2: Ensemble (If Phase 1 Works)
```bash
python scripts/train_challenge1_ensemble.py --n_models 5
```

**Method:**
- Train 5 models with different seeds
- Average predictions
- Reduce variance

**Expected outcome:**
- Test score: 0.90-0.95 (significant improvement!)

### Phase 3: Advanced Techniques (If needed)
- Stochastic Weight Averaging (SWA)
- Snapshot ensembles
- Test-time augmentation (TTA)

## 🎯 Success Metrics

### Validation Metrics (Subject-Aware Split)
- NRMSE < 0.18 (good generalization)
- Pearson r > 0.60 (reasonable correlation)
- Train-val gap < 0.05 (not overfitting)

### Test Metrics (Competition Platform)
- Score < 1.0015 (beat baseline!)
- Target: 0.95-1.00 (good improvement)
- Stretch goal: 0.90-0.95 (excellent!)

## 📅 Implementation Timeline

### Day 1-2: Setup
- [x] Create anti-overfitting config
- [x] Write improved training script
- [ ] Integrate with data loader
- [ ] Test on small subset

### Day 3-4: Training
- [ ] Train single model (15 epochs)
- [ ] Monitor val metrics carefully
- [ ] Save best checkpoint
- [ ] Create submission

### Day 5: Testing
- [ ] Submit to competition
- [ ] Analyze results
- [ ] If successful: train ensemble
- [ ] If not: adjust hyperparameters

### Day 6-7: Refinement
- [ ] Train ensemble (if Phase 1 worked)
- [ ] Try advanced techniques
- [ ] Final submission

## 🚨 Warning Signs

Watch for these issues:

1. **Val loss increases after epoch 5**
   → Stop training! Already overfitting

2. **Train-val gap > 0.10**
   → Increase regularization

3. **Val NRMSE > 0.20**
   → Model not learning useful patterns

4. **Score worse than 1.0015**
   → Overfitting still happening
   → Try even stronger regularization

## 💻 Files Created

- `config/challenge1_anti_overfit.yaml` - Training configuration
- `scripts/train_challenge1_improved.py` - Training script
- `docs/CHALLENGE1_IMPROVEMENT_STRATEGY.md` - This document

## 🎓 Key Insights

1. **Untrained works because it doesn't overfit**
   - Random weights generalize well
   - No memorization of training set

2. **Training can work if done carefully**
   - Need MUCH stronger regularization
   - Must stop early
   - Subject-aware validation critical

3. **The goal is generalization, not training performance**
   - Val NRMSE doesn't predict test score
   - Train-val gap is key indicator
   - Must think about test distribution

## 📚 References

- Mixup: https://arxiv.org/abs/1710.09412
- Label Smoothing: https://arxiv.org/abs/1512.00567
- Early Stopping: Prechelt (1998)
- Weight Decay: https://arxiv.org/abs/1711.05101

---

**Created:** October 29, 2025
**Status:** Ready to implement
**Next Step:** Integrate with data loader and train
