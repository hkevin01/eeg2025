# Challenge 2 Improvement Plan - Detailed TODO

## 🎯 Goal
Improve Challenge 2 from 0.2970 → 0.23-0.25 NRMSE (15-22% improvement)

## 📋 TODO List

### Phase 1: Train on ALL Releases (PRIORITY 1) ⏱️ 1-2 hours

```markdown
- [ ] Step 1.1: Modify train_challenge2_multi_release.py
  - [ ] Change from R2+R3+R4 to R1+R2+R3+R4
  - [ ] Verify data loading for all 4 releases
  - [ ] Check for any corrupted files

- [ ] Step 1.2: Start training
  - [ ] Run: `python scripts/train_challenge2_multi_release.py`
  - [ ] Monitor progress (50 epochs, ~1-2 hours)
  - [ ] Check for convergence

- [ ] Step 1.3: Validate results
  - [ ] Compare NRMSE with current 0.2970
  - [ ] Check if improvement >= 5%
  - [ ] Save model as weights_challenge_2_all_releases.pt

- [ ] Step 1.4: Update submission.py
  - [ ] Load new weights file
  - [ ] Test with test_submission.py
  - [ ] Verify predictions work
```

**Expected Outcome:** 0.26-0.28 NRMSE (5-12% improvement)

---

### Phase 2: Add Sparse Attention (PRIORITY 2) ⏱️ 2-3 hours

```markdown
- [ ] Step 2.1: Create new model architecture
  - [ ] File: models/challenge2_attention.py
  - [ ] Base: CompactExternalizingCNN
  - [ ] Add: SparseMultiHeadAttention layer
  - [ ] Keep params < 150K

- [ ] Step 2.2: Model design
  ```python
  class CompactExternalizingCNNWithAttention(nn.Module):
      def __init__(self):
          # Conv layers (same as baseline)
          self.features = ... (32→64→96 channels)
          
          # NEW: Sparse attention
          self.attention = SparseMultiHeadAttention(
              hidden_size=96,
              scale_factor=0.5,
              dropout=0.3
          )
          
          # Regressor head
          self.regressor = ... (96→48→24→1)
  ```

- [ ] Step 2.3: Create training script
  - [ ] File: scripts/train_challenge2_attention.py
  - [ ] Copy from train_challenge1_attention.py
  - [ ] Adapt for externalizing task
  - [ ] Use R1+R2+R3+R4 data

- [ ] Step 2.4: Train model
  - [ ] Run training (50 epochs)
  - [ ] Monitor NRMSE improvement
  - [ ] Compare with baseline

- [ ] Step 2.5: Validate
  - [ ] Check if NRMSE < 0.26
  - [ ] Test with dummy data
  - [ ] Save as weights_challenge_2_attention.pt
```

**Expected Outcome:** 0.23-0.26 NRMSE (15-22% improvement)

---

### Phase 3: Create Ensemble (PRIORITY 3) ⏱️ 1-2 hours

```markdown
- [ ] Step 3.1: Train multiple models
  - [ ] Model 1: R1+R2+R3+R4, seed=42
  - [ ] Model 2: R1+R2+R3+R4, seed=123
  - [ ] Model 3: R1+R2+R3+R4, seed=456

- [ ] Step 3.2: Train with attention
  - [ ] Model 4: Attention, R1+R2+R3+R4, seed=42
  - [ ] Model 5: Attention, R1+R2+R3+R4, seed=123

- [ ] Step 3.3: Create ensemble predictor
  - [ ] File: models/ensemble_challenge2.py
  - [ ] Load all 5 models
  - [ ] Weighted average predictions
  - [ ] Optimize weights on validation set

- [ ] Step 3.4: Update submission.py
  - [ ] Add ensemble prediction method
  - [ ] Test all models load correctly
  - [ ] Verify output format
```

**Expected Outcome:** 0.22-0.25 NRMSE (18-25% improvement)

---

### Phase 4: Advanced Techniques (PRIORITY 4 - Optional) ⏱️ 3-4 hours

```markdown
- [ ] Step 4.1: Data Augmentation
  - [ ] Add Gaussian noise (σ=0.01-0.02)
  - [ ] Channel dropout (p=0.1)
  - [ ] Temporal shifts (±5 samples)
  - [ ] Amplitude scaling (0.95-1.05×)

- [ ] Step 4.2: Test-Time Augmentation (TTA)
  - [ ] Apply 5-10 augmentations at inference
  - [ ] Average predictions
  - [ ] Expect 2-5% improvement

- [ ] Step 4.3: Stochastic Weight Averaging (SWA)
  - [ ] Average last 10 epochs' weights
  - [ ] Test on validation set
  - [ ] Compare with best single epoch

- [ ] Step 4.4: Hyperparameter Optimization
  - [ ] Grid search or Bayesian optimization
  - [ ] Search: lr, dropout, architecture
  - [ ] Use validation NRMSE as objective
```

**Expected Outcome:** 0.21-0.24 NRMSE (20-28% improvement)

---

## 📊 Progress Tracking

### Current Status
```
Challenge 2 Baseline: 0.2970 NRMSE
Target: 0.23-0.25 NRMSE
Improvement needed: 15-22%
```

### Phase Completion
```
[ ] Phase 1: ALL releases (R1+R2+R3+R4)
[ ] Phase 2: Sparse attention
[ ] Phase 3: Ensemble
[ ] Phase 4: Advanced techniques
```

### Expected Performance After Each Phase
```
Baseline:      0.2970 NRMSE
After Phase 1: 0.26-0.28 NRMSE ( 5-12% ↓)
After Phase 2: 0.23-0.26 NRMSE (15-22% ↓)
After Phase 3: 0.22-0.25 NRMSE (18-25% ↓)
After Phase 4: 0.21-0.24 NRMSE (20-28% ↓)
```

---

## 🏆 Success Criteria

### Minimum Acceptable
- Challenge 2: < 0.27 NRMSE
- Overall: < 0.27 NRMSE
- Rank: Top 5

### Target
- Challenge 2: < 0.25 NRMSE
- Overall: < 0.25 NRMSE
- Rank: Top 3

### Stretch Goal
- Challenge 2: < 0.23 NRMSE
- Overall: < 0.24 NRMSE
- Rank: #1-2

---

## ⏱️ Time Estimates

### Conservative (Phases 1-2 only)
```
Phase 1: 1-2 hours (ALL releases)
Phase 2: 2-3 hours (sparse attention)
Total:   3-5 hours
Expected: 0.23-0.26 NRMSE
```

### Aggressive (Phases 1-3)
```
Phase 1: 1-2 hours
Phase 2: 2-3 hours
Phase 3: 1-2 hours
Total:   4-7 hours
Expected: 0.22-0.25 NRMSE
```

### Complete (All phases)
```
Phase 1: 1-2 hours
Phase 2: 2-3 hours
Phase 3: 1-2 hours
Phase 4: 3-4 hours
Total:   7-11 hours
Expected: 0.21-0.24 NRMSE → Rank #1!
```

---

## 📝 Implementation Order

1. **START HERE:** Phase 1 (ALL releases) - Quick win, proven strategy
2. **DO NEXT:** Phase 2 (sparse attention) - Big improvement potential
3. **IF TIME:** Phase 3 (ensemble) - Solid improvement
4. **OPTIONAL:** Phase 4 (advanced) - Diminishing returns

---

## ✅ Completion Checklist

**Before starting:**
- [x] Analyze current performance
- [x] Identify improvement strategies
- [x] Create detailed plan
- [ ] **BEGIN IMPLEMENTATION**

**After Phase 1:**
- [ ] Model trained on R1+R2+R3+R4
- [ ] NRMSE < 0.28
- [ ] Updated submission.py
- [ ] Tested locally

**After Phase 2:**
- [ ] Attention model created
- [ ] NRMSE < 0.26
- [ ] Compared with baseline
- [ ] Integrated into submission

**After Phase 3:**
- [ ] Ensemble created (3-5 models)
- [ ] NRMSE < 0.25
- [ ] All models tested
- [ ] Submission ready

**Final:**
- [ ] Best model selected
- [ ] ZIP file updated
- [ ] Method description updated
- [ ] Ready to submit

---

## 🚀 Let's Get Started!

**Next Command:**
```bash
# Step 1: Modify training script for ALL releases
vim scripts/train_challenge2_multi_release.py
# Change R2+R3+R4 → R1+R2+R3+R4

# Step 2: Start training
python scripts/train_challenge2_multi_release.py

# Expected: 1-2 hours, NRMSE ~0.26-0.28
```

**Expected Final Score:** 0.24-0.26 NRMSE → Competitive for #1! 🏆
