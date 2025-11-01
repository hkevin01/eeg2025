# C1 Improvement TODO - Target: < 0.8 NRMSE

**Current Status:** C1 = 1.00019 (Rank #72)
**Target:** C1 < 0.8 (Top-tier performance)
**Gap:** 20%+ improvement needed

---

## 📋 Phase 1: Quick Wins (Target: C1 < 0.95)
**Timeline:** 1-2 days  
**Status:** 🟡 In Progress

```markdown
- [x] Research multi-scale temporal architectures
- [x] Design MultiScaleCNN with 3 parallel scales
- [x] Implement comprehensive EEG augmentation suite
  - [x] Time warping (non-linear time stretching)
  - [x] Channel dropout (missing electrode simulation)
  - [x] Gaussian noise injection
  - [x] Amplitude scaling
  - [x] Time shifting
  - [x] Temporal cutout masking
- [x] Create training script (train_c1_multiscale.py)
- [ ] Train 10 models with different random seeds
- [ ] Evaluate ensemble performance on validation set
- [ ] Create submission script for Phase 1 models
- [ ] Test locally to verify NRMSE < 0.95
- [ ] Submit to competition if target achieved
```

**Expected Outcome:** C1 = 0.94-0.96 (4-6% improvement)

---

## 📋 Phase 2: Advanced Methods (Target: C1 < 0.92)
**Timeline:** 2-3 days  
**Status:** ⭕ Not Started

### 2.1 Temporal Convolutional Network (TCN)
```markdown
- [ ] Implement dilated causal convolutions
- [ ] Design TCN with exponentially increasing receptive field
  - [ ] Dilation rates: [1, 2, 4, 8, 16] → RF = 63 timesteps
- [ ] Add residual connections between blocks
- [ ] Train 5 models with different seeds
- [ ] Evaluate performance (expected: 0.92-0.94)
```

### 2.2 Time-Frequency Dual-Branch Model
```markdown
- [ ] Implement frequency encoder (STFT-based)
- [ ] Design dual-branch architecture:
  - [ ] Time-domain branch (1D CNN)
  - [ ] Frequency-domain branch (2D CNN)
- [ ] Add fusion layer for combined features
- [ ] Train 5 models with different seeds
- [ ] Evaluate performance (expected: 0.91-0.93)
```

### 2.3 Ensemble of Phase 2 Models
```markdown
- [ ] Combine TCN + Time-Frequency models
- [ ] Test different ensemble strategies:
  - [ ] Simple averaging
  - [ ] Weighted averaging (based on validation loss)
  - [ ] Stacking with meta-learner
- [ ] Create submission script
- [ ] Test locally
- [ ] Submit to competition if C1 < 0.92
```

**Expected Outcome:** C1 = 0.91-0.93 (7-9% improvement)

---

## 📋 Phase 3: State-of-Art Methods (Target: C1 < 0.8)
**Timeline:** 3-5 days  
**Status:** ⭕ Not Started

### 3.1 Transformer Architecture
```markdown
- [ ] Design temporal attention mechanism
- [ ] Implement multi-head attention over timepoints
- [ ] Add positional encoding
- [ ] Create 4-layer transformer model
- [ ] Train with large batch size (128+)
- [ ] Evaluate performance (expected: 0.85-0.90)
```

### 3.2 Contrastive Pre-training
```markdown
- [ ] Design self-supervised contrastive loss
- [ ] Implement augmentation pairs:
  - [ ] Time shifts
  - [ ] Amplitude scaling
  - [ ] Temporal masking
- [ ] Pre-train encoder on unlabeled data
- [ ] Fine-tune on labeled Challenge 1 data
- [ ] Evaluate performance (expected: 0.80-0.85)
```

### 3.3 Mega-Ensemble
```markdown
- [ ] Combine all architectures:
  - [ ] 10x MultiScale CNN (Phase 1)
  - [ ] 5x TCN (Phase 2)
  - [ ] 5x Time-Frequency (Phase 2)
  - [ ] 3x Transformer (Phase 3)
  - [ ] 3x Contrastive Pre-trained (Phase 3)
- [ ] Optimize ensemble weights
- [ ] Test with different TTA strategies
- [ ] Create final submission
- [ ] Test locally (target: < 0.8)
- [ ] Submit to competition
```

**Expected Outcome:** C1 = 0.78-0.82 (18-22% improvement)

---

## 🔬 Research & Analysis Tasks

```markdown
- [x] Analyze current architecture limitations
- [x] Create comprehensive improvement plan
- [ ] Study braindecode EEG-specific models
  - [ ] Check EEGConformer implementation
  - [ ] Review TCNet architecture
  - [ ] Analyze ShallowFBCSPNet design
- [ ] Research EEG response time prediction papers
  - [ ] Identify relevant frequency bands (alpha, beta)
  - [ ] Understand readiness potentials
  - [ ] Review motor preparation signals
- [ ] Analyze competition data characteristics
  - [ ] Time window properties (-0.5 to 0s before RT)
  - [ ] Channel importance (motor vs frontal)
  - [ ] Subject variability patterns
```

---

## 📊 Validation & Testing

```markdown
- [ ] Create automated validation pipeline
- [ ] Implement k-fold cross-validation
- [ ] Test for subject leakage (critical!)
- [ ] Verify TTA strategies work correctly
- [ ] Profile model inference speed
- [ ] Check memory usage on competition platform
```

---

## 🎯 Success Metrics

### Minimum Viable (Phase 1)
- **Target:** C1 < 0.95
- **Rank:** #40-50 (from #72)
- **Status:** 🟡 In Progress

### Intermediate (Phase 2)
- **Target:** C1 < 0.92
- **Rank:** #20-30
- **Status:** ⭕ Not Started

### Stretch Goal (Phase 3)
- **Target:** C1 < 0.8
- **Rank:** Top 10
- **Status:** ⭕ Not Started

---

## 📈 Progress Tracking

| Date | Phase | Best C1 | Rank | Notes |
|------|-------|---------|------|-------|
| Nov 1 | Baseline | 1.00019 | #72 | EnhancedCompactCNN (V10) |
| Nov 1 | Phase 1 | ? | ? | Multi-Scale CNN training started |
| | Phase 2 | ? | ? | TCN + Time-Freq |
| | Phase 3 | ? | ? | Transformer + Contrastive |

---

## ⚠️ Risks & Mitigation

```markdown
- [ ] **Risk:** Platform size limits (failed at 6.1 MB)
  - **Mitigation:** Keep individual models < 1 MB, ensemble < 5 MB
  
- [ ] **Risk:** Overfitting to validation set
  - **Mitigation:** Use proper train/val/test splits, heavy augmentation
  
- [ ] **Risk:** Subject leakage in folds
  - **Mitigation:** Verify subject-aware splitting
  
- [ ] **Risk:** Time budget (10 days for 20% improvement)
  - **Mitigation:** Parallelize training, focus on high-impact methods first
  
- [ ] **Risk:** Hardware limits (GPU memory for transformers)
  - **Mitigation:** Use gradient checkpointing, smaller batch sizes
```

---

## 💻 Commands to Run

### Phase 1: Train Multi-Scale CNN
```bash
cd /home/kevin/Projects/eeg2025
python scripts/training/train_c1_multiscale.py
```

### Check Results
```bash
ls -lh checkpoints/phase1_multiscale/
python scripts/eval_c1_ensemble.py
```

### Create Submission
```bash
python scripts/create_submission_v14.py
```

---

## 📝 Notes

- **Priority 1:** Get Phase 1 working ASAP (quick wins)
- **Priority 2:** Implement TCN (proven for EEG)
- **Priority 3:** Try transformer if time permits
- **Key Insight:** Top performer (brain&ai) only at 0.91222 - target of 0.8 is VERY ambitious
- **Alternative:** Aim for 0.90-0.92 first (realistic), then push toward 0.8

---

**Last Updated:** November 1, 2024 - 6:15 PM
**Status:** Phase 1 implementation complete, ready to train
**Next Action:** Run `train_c1_multiscale.py` to train 10 models
