# 🎯 COMPREHENSIVE TODO LIST - EEG2025 Project
**Generated:** October 15, 2025
**Status:** Post-Initial Training Phase

---

## 📊 CURRENT STATUS SUMMARY

### ✅ COMPLETED (100%)
- [x] VS Code optimization (88% CPU reduction, zero crashes)
- [x] Foundation model trained (minimal_best.pth, 50.8% val acc)
- [x] Challenge 1 trained (challenge1_best.pth, Pearson r=0.0593)
- [x] Submission generated (challenge1_predictions.csv, 400 rows)
- [x] Comprehensive documentation (8 guides, ~2000 lines)

### ⚠️ NEEDS IMPROVEMENT (Key Issues)
- [ ] Challenge 1 performance LOW (0.0593 r < 0.3 target) - using random ages
- [ ] Challenge 2 NOT STARTED
- [ ] Only 13% of data used (5K/38K samples)
- [ ] Foundation model undertrained (small model, limited data)
- [ ] No testing infrastructure
- [ ] No artifact detection/removal

---

## 🔥 PRIORITY 1: CRITICAL FIXES (Est: 1-2 hours) ⭐⭐⭐

**Goal:** Get Challenge 1 to meet minimum target (Pearson r > 0.3)

### Task 1.1: Get Real Age Labels (15 min)
```bash
- [ ] Check if participants.tsv exists
      cd /home/kevin/Projects/eeg2025
      ls -lh data/raw/hbn/participants.tsv

- [ ] If exists: Verify format
      head data/raw/hbn/participants.tsv
      # Expected columns: participant_id, age, sex

- [ ] If missing: Extract from BIDS metadata
      # See GUIDE_IMPROVE_CHALLENGE1.md for extraction script

- [ ] Create age mapping for dataset
      # participant_id → age mapping
```

**Deliverable:** Real age labels for all 38,506 windows

### Task 1.2: Re-train Challenge 1 with Real Labels (30 min)
```bash
- [ ] Update train_challenge1_simple.py to use real ages
      # Replace random age generation with real labels
      
- [ ] Use more samples (5K instead of 2K)
      # Better sampling of age distribution
      
- [ ] Train for more epochs (5-10 instead of 3)
      python3 scripts/train_challenge1_simple.py

- [ ] Verify improved performance
      # Target: Pearson r > 0.3 ✅
```

**Expected Result:** Pearson r: 0.3-0.4 (meets target!)

### Task 1.3: Implement Challenge 2 (30 min)
```bash
- [ ] Copy Challenge 1 script
      cp scripts/train_challenge1_simple.py scripts/train_challenge2.py

- [ ] Modify for binary classification
      # Change: regression → classification
      # Loss: MSE → BCE
      # Metric: Pearson r → AUROC
      # Labels: age → sex (M=1, F=0)

- [ ] Train model
      python3 scripts/train_challenge2.py

- [ ] Generate submission
      # submissions/challenge2_predictions.csv
```

**Expected Result:** AUROC > 0.7 ✅

**Completion Criteria:**
- [ ] Challenge 1: Pearson r > 0.3
- [ ] Challenge 2: AUROC > 0.7
- [ ] Both submissions ready for competition

---

## 🚀 PRIORITY 2: PERFORMANCE IMPROVEMENTS (Est: 4-6 hours) ⭐⭐

**Goal:** Maximize model performance for competition ranking

### Task 2.1: Train on Full Dataset (3-5 hours)
```bash
- [ ] Copy minimal script
      cp scripts/train_minimal.py scripts/train_full.py

- [ ] Modify configuration
      # max_samples: 5000 → None (use all 38,506)
      # hidden_dim: 64 → 128
      # n_heads: 4 → 8
      # n_layers: 2 → 4
      # epochs: 5 → 10

- [ ] Run in background with tmux
      tmux new -s fulltraining
      python3 scripts/train_full.py
      # Ctrl+B then D to detach

- [ ] Monitor progress
      tail -f logs/full_*.log

- [ ] Verify completion
      ls -lh checkpoints/full_best.pth
      # Should be ~5-10MB (larger model)
```

**Expected Result:** Val accuracy 60-70% (vs 50.8%)

### Task 2.2: Progressive Unfreezing for Challenges (1 hour)
```bash
- [ ] Implement 3-phase training
      # Phase 1: Train head only (3 epochs)
      # Phase 2: Unfreeze last layer (3 epochs)
      # Phase 3: Fine-tune all (4 epochs)

- [ ] Update Challenge 1 script
      # See GUIDE_IMPROVE_CHALLENGE1.md for code

- [ ] Re-train Challenge 1
      python3 scripts/train_challenge1_improved.py

- [ ] Re-train Challenge 2
      python3 scripts/train_challenge2_improved.py
```

**Expected Result:** +0.05-0.10 improvement in metrics

### Task 2.3: Hyperparameter Optimization (2 hours)
```bash
- [ ] Define search space
      # learning_rate: [1e-4, 1e-3, 5e-3]
      # batch_size: [32, 64, 128]
      # dropout: [0.1, 0.2, 0.3]

- [ ] Run grid search or random search
      # Try 10-20 combinations

- [ ] Select best config

- [ ] Retrain with optimal hyperparameters
```

**Expected Result:** +0.02-0.05 improvement

---

## 🔬 PRIORITY 3: ROBUSTNESS & QUALITY (Est: 4-6 hours) ⭐

**Goal:** Improve data quality and model reliability

### Task 3.1: Artifact Detection (2 hours)
```bash
- [ ] Implement bad channel detection
      # Detect flat/noisy channels
      # Remove or interpolate

- [ ] Implement amplitude-based rejection
      # Detect extreme values
      # Flag artifact windows

- [ ] Add ICA-based artifact removal
      # Use MNE-Python ICA
      # Remove eye blinks, muscle artifacts

- [ ] Create before/after visualizations

- [ ] Document artifact statistics
```

### Task 3.2: Testing Infrastructure (2-3 hours)
```bash
- [ ] Create tests/test_data_loading.py
      # Test dataset loads correctly
      # Test all subjects readable
      # Test window extraction

- [ ] Create tests/test_models.py
      # Test model forward pass
      # Test output shapes
      # Test gradient flow

- [ ] Create tests/test_metrics.py
      # Test Pearson r calculation
      # Test AUROC calculation
      # Test balanced accuracy

- [ ] Create tests/test_training.py
      # Test training loop
      # Test checkpoint saving
      # Test early stopping

- [ ] Setup pytest and coverage
      pip install pytest pytest-cov
      pytest --cov=scripts tests/

- [ ] Target: >30% code coverage
```

### Task 3.3: Cross-Site Validation (2 hours)
```bash
- [ ] Identify sites in HBN dataset
      # List unique sites/scanners

- [ ] Implement leave-one-site-out CV
      # Train on N-1 sites
      # Test on held-out site

- [ ] Run for each site

- [ ] Compare performance across sites
      # Identify problematic sites
      # Document cross-site generalization

- [ ] Apply domain adaptation if needed
```

---

## 📚 PRIORITY 4: DOCUMENTATION & ORGANIZATION (Est: 2-3 hours) ⭐

**Goal:** Clean documentation and reproducible pipeline

### Task 4.1: Update Documentation (1 hour)
```bash
- [ ] Update README.md with latest results
      # Add performance metrics
      # Add competition status
      # Update installation instructions

- [ ] Create RESULTS.md
      # Detailed results for all experiments
      # Tables, plots, analysis

- [ ] Create REPRODUCIBILITY.md
      # Step-by-step instructions
      # Environment setup
      # Data preparation
      # Training commands
      # Submission generation

- [ ] Archive old TODO files
      mkdir -p docs/archive
      mv TODO_*.md docs/archive/
```

### Task 4.2: Code Organization (1 hour)
```bash
- [ ] Consolidate training scripts
      # Merge similar scripts
      # Create unified train.py with flags

- [ ] Create requirements-lock.txt
      pip freeze > requirements-lock.txt

- [ ] Add type hints
      # Main functions and classes

- [ ] Add docstrings
      # All public functions

- [ ] Format code consistently
      # Consider using black/ruff
```

### Task 4.3: Create Submission Package (30 min)
```bash
- [ ] Create submission README
      # Method description
      # Architecture details
      # Training procedure

- [ ] Prepare code for sharing
      # Clean up notebooks
      # Remove debug code
      # Add comments

- [ ] Create submission.zip
      zip -r submission.zip scripts/ configs/ README.md
```

---

## 🎯 PRIORITY 5: COMPETITION SUBMISSION (Est: 30 min) ⭐⭐⭐

**Goal:** Submit to competition and get ranked

### Task 5.1: Validate Submissions (10 min)
```bash
- [ ] Run validation script
      python3 << 'PY'
      import pandas as pd
      
      # Challenge 1
      df1 = pd.read_csv('submissions/challenge1_predictions.csv')
      assert list(df1.columns) == ['participant_id', 'age_prediction']
      assert len(df1) == 400
      assert df1['age_prediction'].between(5, 25).all()
      assert df1.isnull().sum().sum() == 0
      print("✅ Challenge 1 valid")
      
      # Challenge 2
      df2 = pd.read_csv('submissions/challenge2_predictions.csv')
      assert list(df2.columns) == ['participant_id', 'sex_prediction']
      assert len(df2) == 400
      assert df2['sex_prediction'].between(0, 1).all()
      assert df2.isnull().sum().sum() == 0
      print("✅ Challenge 2 valid")
      PY

- [ ] Check file sizes
      ls -lh submissions/*.csv
      # Should be < 100KB each
```

### Task 5.2: Submit to Platform (15 min)
```bash
- [ ] Install Kaggle CLI (if using Kaggle)
      pip install kaggle
      mkdir -p ~/.kaggle
      # Download kaggle.json from account settings

- [ ] Submit Challenge 1
      kaggle competitions submit -c <competition>         -f submissions/challenge1_predictions.csv         -m "Transfer learning with [describe approach]"

- [ ] Submit Challenge 2
      kaggle competitions submit -c <competition>         -f submissions/challenge2_predictions.csv         -m "Binary classification with [describe approach]"

- [ ] Check leaderboard
      kaggle competitions leaderboard <competition>
```

### Task 5.3: Track Results (5 min)
```bash
- [ ] Record submission details
      # Date, time, method, score, rank

- [ ] Update submission log
      # See GUIDE_COMPETITION_SUBMISSION.md

- [ ] Plan next iteration based on results
```

---

## 📈 PRIORITY 6: ADVANCED IMPROVEMENTS (Optional) ⭐

**Goal:** Push performance to top tier

### Task 6.1: Ensemble Methods (2 hours)
```bash
- [ ] Train multiple models with different seeds
- [ ] Train with different architectures
- [ ] Average predictions
- [ ] Test ensemble performance
```

### Task 6.2: Advanced Architectures (3-4 hours)
```bash
- [ ] Implement CNN-Transformer hybrid
- [ ] Try attention mechanisms
- [ ] Experiment with different positional encodings
- [ ] Test multi-scale features
```

### Task 6.3: Data Augmentation (2 hours)
```bash
- [ ] Time-domain augmentation
      # Time shift, scale, jitter

- [ ] Frequency-domain augmentation
      # Frequency masking, noise injection

- [ ] Test augmentation impact
```

---

## 📋 COMPLETION CHECKLIST

### Must Have (Before Competition Deadline)
- [ ] Challenge 1: Pearson r > 0.3 ✅
- [ ] Challenge 2: AUROC > 0.7 ✅
- [ ] Submissions validated and uploaded
- [ ] Code documented and reproducible
- [ ] Results documented

### Should Have (For Strong Performance)
- [ ] Full dataset training complete
- [ ] Progressive unfreezing implemented
- [ ] Artifact detection working
- [ ] Testing coverage > 30%
- [ ] Cross-site validation done

### Nice to Have (For Top Performance)
- [ ] Ensemble models trained
- [ ] Advanced architectures tested
- [ ] Hyperparameter optimization complete
- [ ] Data augmentation implemented

---

## ⏱️ TIME ESTIMATES

| Priority | Tasks | Est. Time | Importance |
|----------|-------|-----------|------------|
| P1: Critical Fixes | 3 tasks | 1-2 hours | ⭐⭐⭐ Must Do |
| P2: Performance | 3 tasks | 4-6 hours | ⭐⭐ Should Do |
| P3: Robustness | 3 tasks | 4-6 hours | ⭐ Good to Do |
| P4: Documentation | 3 tasks | 2-3 hours | ⭐ Good to Do |
| P5: Submission | 3 tasks | 30 min | ⭐⭐⭐ Must Do |
| P6: Advanced | 3 tasks | 7-10 hours | ⭐ Optional |

**Minimum Path:** P1 + P5 = 2 hours → Competition ready ✅  
**Recommended Path:** P1 + P2 + P5 = 6-8 hours → Competitive performance ✅✅  
**Complete Path:** P1-P6 = 20-30 hours → Top tier ✅✅✅

---

## �� NEXT IMMEDIATE ACTIONS

**If you have 2 hours:**
```bash
1. Get real age labels (15 min)
2. Re-train Challenge 1 (30 min)
3. Train Challenge 2 (30 min)
4. Submit both (15 min)
→ Result: Competition ready!
```

**If you have 6-8 hours:**
```bash
1. Do above (2 hours)
2. Train full dataset (3-5 hours)
3. Re-train challenges with full model (1 hour)
4. Submit improved versions (15 min)
→ Result: Strong competition performance!
```

**If you have time later:**
```bash
1. Add testing (2-3 hours)
2. Implement artifact detection (2 hours)
3. Cross-site validation (2 hours)
4. Advanced improvements (as needed)
→ Result: Publication-quality work!
```

---

## 📊 SUCCESS METRICS

**Minimum (Pass):**
- ✅ Challenge 1: Pearson r > 0.3
- ✅ Challenge 2: AUROC > 0.7
- ✅ Submissions uploaded

**Target (Competitive):**
- ✅ Challenge 1: Pearson r > 0.5
- ✅ Challenge 2: AUROC > 0.8
- ✅ Full dataset training complete

**Stretch (Top 10%):**
- 🏆 Challenge 1: Pearson r > 0.7
- 🏆 Challenge 2: AUROC > 0.9
- 🏆 Ensemble methods working

---

**Last Updated:** October 15, 2025  
**Status:** Ready for Priority 1 execution

🚀 **START WITH:** Task 1.1 - Get real age labels (15 minutes)
