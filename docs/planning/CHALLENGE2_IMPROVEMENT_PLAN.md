# Challenge 2: Improvement Plan - Actionable Tasks

## 🎯 Goal
Improve Challenge 2 from 0.2970 → 0.23-0.26 NRMSE (13-23% improvement)

---

## Phase 1: Quick Wins (Est: 2-3 hours, +6-9% improvement)

### Task 1.1: Complete R2+R3+R4 Training ⏱️ 30 min
**Status:** ⭕ Not Started  
**Priority:** 🔴 Critical  
**Expected Gain:** -6-9% NRMSE

**Actions:**
```bash
# Fix and rerun the multi-release training
python scripts/train_challenge2_multi_release.py --releases R2 R3 R4
```

**Success Criteria:**
- Training completes without crashes
- Validation NRMSE < 0.28
- Model saved to checkpoints/

**Files to check:**
- `scripts/train_challenge2_multi_release.py`
- `logs/challenge2_expanded_*.log`

---

### Task 1.2: Add Data Augmentation ⏱️ 45 min
**Status:** ⭕ Not Started  
**Priority:** 🟠 High  
**Expected Gain:** -3-5% NRMSE

**Actions:**
1. Add augmentation to Challenge 2 training:
   - Gaussian noise (σ=0.01-0.02)
   - Channel dropout (p=0.05-0.1)
   - Temporal shifts (±3-5 samples)
   - Amplitude scaling (0.95-1.05×)

2. Modify training script:
```python
# Add to train_challenge2_multi_release.py
def augment_eeg(x):
    # Gaussian noise
    if random.random() < 0.5:
        x = x + torch.randn_like(x) * 0.02
    # Channel dropout
    if random.random() < 0.3:
        mask = torch.rand(x.shape[1]) > 0.1
        x[:, ~mask, :] = 0
    # Amplitude scaling
    if random.random() < 0.5:
        x = x * random.uniform(0.95, 1.05)
    return x
```

**Success Criteria:**
- Augmentation applied during training
- Training time increases by <20%
- Validation NRMSE improves

---

### Task 1.3: Implement 3-Fold Cross-Validation ⏱️ 1 hour
**Status:** ⭕ Not Started  
**Priority:** 🟠 High  
**Expected Gain:** -3-4% NRMSE

**Actions:**
1. Create CV training script for Challenge 2
2. Split data into 3 folds
3. Train on each fold
4. Ensemble predictions

**Files to create:**
- `scripts/train_challenge2_cv.py` (adapt from train_challenge1_attention.py)

**Success Criteria:**
- 3 models trained successfully
- Each fold NRMSE < 0.30
- Mean CV NRMSE < 0.27

---

## Phase 2: Architecture Enhancements (Est: 3-4 hours, +8-12% improvement)

### Task 2.1: Add Sparse Attention to Challenge 2 ⏱️ 1.5 hours
**Status:** ⭕ Not Started  
**Priority:** 🟠 High  
**Expected Gain:** -4-6% NRMSE

**Actions:**
1. Create `models/challenge2_attention.py`
2. Adapt LightweightResponseTimeCNNWithAttention for externalizing
3. Keep model compact (<200K params)
4. Train with same strategy as Challenge 1

**Architecture:**
```python
class CompactExternalizingCNNWithAttention(nn.Module):
    def __init__(self):
        # CNN backbone (same as current)
        # + Channel attention
        # + Sparse multi-head attention
        # + Regression head
```

**Success Criteria:**
- Model parameters < 200K
- Training time < 20 min
- Validation NRMSE < 0.25

---

### Task 2.2: Try Larger Model ⏱️ 45 min
**Status:** ⭕ Not Started  
**Priority:** 🟡 Medium  
**Expected Gain:** -2-4% NRMSE

**Actions:**
1. Increase model capacity:
   - 129 → 64 → 128 → 192 (vs current 32 → 64 → 96)
   - Add one more conv layer
   - Larger regression head

2. Train with strong regularization to prevent overfitting

**Success Criteria:**
- Model parameters 100-200K
- No overfitting (train/val gap < 15%)
- Validation NRMSE < 0.26

---

### Task 2.3: Hyperparameter Tuning ⏱️ 2 hours
**Status:** ⭕ Not Started  
**Priority:** 🟡 Medium  
**Expected Gain:** -2-3% NRMSE

**Parameters to tune:**
- Learning rate: [0.0005, 0.001, 0.002]
- Weight decay: [0.01, 0.02, 0.05]
- Dropout: [0.3, 0.4, 0.5, 0.6]
- Batch size: [32, 64, 128]

**Actions:**
```bash
# Grid search or random search
python scripts/hyperparam_search_challenge2.py
```

**Success Criteria:**
- Best hyperparameters identified
- Validation NRMSE improvement > 2%

---

## Phase 3: Advanced Techniques (Est: 4-6 hours, +5-10% improvement)

### Task 3.1: Ensemble Multiple Models ⏱️ 1 hour
**Status:** ⭕ Not Started  
**Priority:** 🟠 High  
**Expected Gain:** -4-5% NRMSE

**Actions:**
1. Train 3-5 diverse models:
   - Different architectures
   - Different random seeds
   - Different data splits

2. Weighted ensemble:
```python
predictions = (
    0.4 * model1_pred +
    0.3 * model2_pred +
    0.3 * model3_pred
)
```

**Success Criteria:**
- 3+ models trained
- Ensemble NRMSE < individual models
- Improvement > 3%

---

### Task 3.2: Release-Aware Training ⏱️ 2 hours
**Status:** ⭕ Not Started  
**Priority:** 🟡 Medium  
**Expected Gain:** -2-4% NRMSE

**Actions:**
1. Add release embeddings to model
2. Learn release-specific adjustments
3. Condition predictions on release

**Architecture:**
```python
class ReleaseAwareModel(nn.Module):
    def __init__(self):
        self.cnn = CompactExternalizingCNN()
        self.release_embedding = nn.Embedding(5, 32)  # 5 releases
        self.fusion = nn.Linear(output_dim + 32, 1)
```

**Success Criteria:**
- Model learns release patterns
- Better generalization across releases
- Validation NRMSE < 0.25

---

### Task 3.3: Transfer Learning from Challenge 1 ⏱️ 2 hours
**Status:** ⭕ Not Started  
**Priority:** 🟢 Low  
**Expected Gain:** -2-3% NRMSE

**Actions:**
1. Use Challenge 1 CNN backbone (pretrained)
2. Fine-tune on Challenge 2 data
3. Share attention mechanisms

**Success Criteria:**
- Faster convergence
- Better feature extraction
- Validation NRMSE < 0.26

---

## Implementation Priority

### Must Do (Before Next Submission)
1. ✅ Task 1.1: R2+R3+R4 training
2. ✅ Task 1.2: Data augmentation
3. ✅ Task 1.3: Cross-validation

### Should Do (For Significant Improvement)
4. ✅ Task 2.1: Sparse attention
5. ✅ Task 3.1: Ensemble

### Nice to Have (If Time Permits)
6. ⭕ Task 2.2: Larger model
7. ⭕ Task 2.3: Hyperparameter tuning
8. ⭕ Task 3.2: Release-aware training

---

## Expected Timeline

**Week 1 (October 17-24):**
- Day 1: Tasks 1.1, 1.2 (R2+R3+R4 + augmentation)
- Day 2: Task 1.3 (cross-validation)
- Day 3: Task 2.1 (sparse attention)
- Day 4: Task 3.1 (ensemble)
- Day 5: Testing and validation
- Day 6-7: Submission preparation

**Target NRMSE by end of Week 1:** 0.24-0.26

---

## Success Metrics

```
Phase 1 Complete:  0.27-0.28 NRMSE (✓ submittable)
Phase 2 Complete:  0.24-0.26 NRMSE (✓✓ competitive)
Phase 3 Complete:  0.23-0.24 NRMSE (✓✓✓ excellent)

Current submission: 0.2970
Target for next:    0.24-0.26 (13-19% improvement)
Stretch goal:       0.23 (23% improvement)
```

---

**Last Updated:** October 17, 2025  
**Status:** Ready to implement  
**Next Action:** Start with Task 1.1 (R2+R3+R4 training)
