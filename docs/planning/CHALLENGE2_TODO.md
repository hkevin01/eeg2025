# Challenge 2 Improvement - TODO Checklist

## 🎯 Current Status
- Challenge 2 NRMSE: 0.2970
- Target: 0.23-0.26 (13-23% improvement)
- Priority: Improve before next submission

---

## ✅ Phase 1: Quick Wins (2-3 hours)

```markdown
- [x] Task 1.1: Complete R2+R3+R4 Training (30 min) 🔴 CRITICAL
  - [x] Fix crash issue in train_challenge2_multi_release.py
  - [x] Run training with R2+R3+R4 releases
  - [ ] Verify training completes successfully (IN PROGRESS)
  - [ ] Check validation NRMSE < 0.28
  - [ ] Save model to checkpoints/

- [ ] Task 1.2: Add Data Augmentation (45 min) 🟠 HIGH
  - [ ] Add Gaussian noise (σ=0.01-0.02)
  - [ ] Add channel dropout (p=0.05-0.1)
  - [ ] Add temporal shifts (±3-5 samples)
  - [ ] Add amplitude scaling (0.95-1.05×)
  - [ ] Test augmentation doesn't break training
  - [ ] Verify validation NRMSE improves

- [ ] Task 1.3: Implement Cross-Validation (1 hour) 🟠 HIGH
  - [ ] Create train_challenge2_cv.py
  - [ ] Implement 3-fold CV split
  - [ ] Train model on each fold
  - [ ] Save all fold models
  - [ ] Calculate mean CV NRMSE
  - [ ] Verify mean NRMSE < 0.27
```

**Expected Result After Phase 1:** NRMSE = 0.27-0.28

---

## ✅ Phase 2: Architecture Enhancements (3-4 hours)

```markdown
- [ ] Task 2.1: Add Sparse Attention (1.5 hours) 🟠 HIGH
  - [ ] Create models/challenge2_attention.py
  - [ ] Implement CompactExternalizingCNNWithAttention
  - [ ] Keep parameters < 200K
  - [ ] Add channel attention
  - [ ] Add sparse multi-head attention
  - [ ] Train and validate
  - [ ] Verify NRMSE < 0.25

- [ ] Task 2.2: Try Larger Model (45 min) 🟡 MEDIUM
  - [ ] Increase conv channels (64→128→192)
  - [ ] Add extra conv layer
  - [ ] Enlarge regression head
  - [ ] Train with strong regularization
  - [ ] Check for overfitting
  - [ ] Verify NRMSE < 0.26

- [ ] Task 2.3: Hyperparameter Tuning (2 hours) 🟡 MEDIUM
  - [ ] Create hyperparam_search_challenge2.py
  - [ ] Define search space (lr, wd, dropout, bs)
  - [ ] Run grid/random search
  - [ ] Identify best parameters
  - [ ] Retrain with best params
  - [ ] Verify improvement > 2%
```

**Expected Result After Phase 2:** NRMSE = 0.24-0.26

---

## ✅ Phase 3: Advanced Techniques (4-6 hours)

```markdown
- [ ] Task 3.1: Ensemble Models (1 hour) 🟠 HIGH
  - [ ] Train 3-5 diverse models
  - [ ] Different architectures/seeds/splits
  - [ ] Implement weighted ensemble
  - [ ] Tune ensemble weights
  - [ ] Verify ensemble > individual models
  - [ ] Check improvement > 3%

- [ ] Task 3.2: Release-Aware Training (2 hours) 🟡 MEDIUM
  - [ ] Add release embeddings
  - [ ] Implement release-conditional model
  - [ ] Train on multi-release data
  - [ ] Test generalization
  - [ ] Verify NRMSE < 0.25

- [ ] Task 3.3: Transfer Learning (2 hours) 🟢 LOW
  - [ ] Load Challenge 1 pretrained backbone
  - [ ] Fine-tune on Challenge 2 data
  - [ ] Test shared representations
  - [ ] Verify NRMSE < 0.26
```

**Expected Result After Phase 3:** NRMSE = 0.23-0.24

---

## 📋 Implementation Order

### Day 1 (Today - Oct 17)
1. ⬜ Task 1.1: R2+R3+R4 training (30 min)
2. ⬜ Task 1.2: Data augmentation (45 min)

### Day 2 (Oct 18)
3. ⬜ Task 1.3: Cross-validation (1 hour)
4. ⬜ Task 2.1: Sparse attention (1.5 hours)

### Day 3 (Oct 19)
5. ⬜ Task 3.1: Ensemble (1 hour)
6. ⬜ Testing & validation (2 hours)

### Day 4-5 (Oct 20-21)
7. ⬜ Optional: Tasks 2.2, 2.3, 3.2
8. ⬜ Final submission preparation

---

## 🎯 Success Criteria

```
Phase 1 ✓:  NRMSE 0.27-0.28  →  Ready to submit
Phase 2 ✓:  NRMSE 0.24-0.26  →  Competitive
Phase 3 ✓:  NRMSE 0.23-0.24  →  Excellent

Minimum viable: Complete Phase 1
Target: Complete Phase 1 + Phase 2
Stretch: Complete all phases
```

---

## 🚀 Next Action

**START HERE:**
```bash
# 1. Check current training script
cat scripts/train_challenge2_multi_release.py | grep -A 5 "releases"

# 2. Run R2+R3+R4 training
python scripts/train_challenge2_multi_release.py

# 3. Monitor logs
tail -f logs/challenge2_*.log
```

---

**Last Updated:** October 17, 2025, 14:30  
**Current Focus:** Task 1.1 - R2+R3+R4 Training  
**Status:** Ready to start implementation
