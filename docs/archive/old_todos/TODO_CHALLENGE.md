# EEG Foundation Challenge 2025 - Action Plan

**Date**: October 13, 2025  
**Status**: Infrastructure Complete ✅ | Data Acquisition Needed 🔴  
**Goal**: Top 25% Performance (Stretch: Top 10%)

---

## 🔴 CRITICAL - Week 1 (Do First!)

### Data Acquisition & Validation

- [ ] **Install AWS CLI**
  ```bash
  pip install awscli
  ```

- [ ] **Download HBN Dataset** (Choose one)
  - [ ] Option 1: Quick Test (10 subjects, ~5GB) - **Recommended to start**
    ```bash
    ./scripts/download_hbn_data.sh
    # Select option 1
    ```
  - [ ] Option 2: Medium Set (100 subjects, ~50GB) - For development
  - [ ] Option 3: Full Dataset (1500+ subjects, ~500GB) - For final submission

- [ ] **Run Data Validation Tests**
  ```bash
  export HBN_DATA_PATH="/path/to/your/data"
  python tests/test_data_acquisition.py
  ```
  - [ ] Verify all 8 tests pass
  - [ ] Check data quality metrics
  - [ ] Confirm official splits loaded

- [ ] **Update Project Configuration**
  ```bash
  # Update configs/enhanced.yaml with your data path
  sed -i "s|/path/to/hbn/data|$HBN_DATA_PATH|g" configs/enhanced.yaml
  ```

- [ ] **Test Data Loading Pipeline**
  ```bash
  python scripts/dry_run.py --data_path $HBN_DATA_PATH
  ```

---

## 🟡 HIGH PRIORITY - Week 2 (Critical for Competition)

### Baseline Model Training

- [ ] **Challenge 1: Cross-Task Transfer (SuS → CCD)**
  ```bash
  python src/training/train_cross_task.py \
      --config configs/enhanced.yaml \
      --gpus 1 \
      --epochs 50
  ```
  - [ ] Record baseline metrics (RT correlation, Success AUROC)
  - [ ] Save best checkpoint
  - [ ] Log training curves to wandb

- [ ] **Challenge 2: Psychopathology Prediction**
  ```bash
  python src/training/train_psych.py \
      --config configs/enhanced.yaml \
      --gpus 1 \
      --epochs 50
  ```
  - [ ] Record baseline metrics (P-factor, Internalizing, Externalizing, Attention)
  - [ ] Save best checkpoint
  - [ ] Monitor multi-task loss curves

### Artifact Detection Implementation

- [ ] **Create Artifact Removal Module**
  - [ ] Eye blink detection (EOG-based)
  - [ ] Muscle artifact detection (>70Hz)
  - [ ] Bad channel detection (correlation)
  - [ ] Amplitude-based rejection (>100μV)

- [ ] **Integrate into Preprocessing Pipeline**
  ```bash
  # Update src/dataio/preprocessing.py
  # Add ArtifactRemover class
  ```

- [ ] **Test Artifact Removal**
  ```bash
  python tests/test_artifact_removal.py
  ```

### Inference Speed Validation

- [ ] **Create Inference Speed Test**
  ```python
  # tests/test_inference_speed.py
  # Measure latency: MUST be <50ms
  ```

- [ ] **Optimize Model for Speed**
  - [ ] Model quantization (INT8)
  - [ ] Pruning (30-50% reduction)
  - [ ] ONNX export
  - [ ] Batch processing optimization

- [ ] **Verify <50ms Requirement**
  ```bash
  python tests/test_inference_speed.py
  # Target: <40ms average (10ms buffer)
  ```

---

## 🟢 MEDIUM PRIORITY - Week 3 (Performance Optimization)

### Cross-Site Validation

- [ ] **Implement Leave-One-Site-Out CV**
  ```python
  # Test domain adaptation on held-out sites
  sites = ['RU', 'CBIC', 'CUNY', 'SI']
  for held_out_site in sites:
      # Train on others, validate on held_out
      pass
  ```

- [ ] **Validate Domain Adaptation**
  - [ ] Check performance drop on new sites
  - [ ] Tune DANN lambda parameter
  - [ ] Test multi-adversary effectiveness

### Hyperparameter Optimization

- [ ] **Set Up HPO Framework**
  ```bash
  pip install optuna
  ```

- [ ] **Challenge 1 HPO**
  ```bash
  python scripts/hyperparameter_search.py \
      --challenge challenge1 \
      --n_trials 50 \
      --search_space config/hpo_challenge1.yaml
  ```
  - [ ] Optimize learning rate
  - [ ] Optimize model architecture
  - [ ] Optimize data augmentation

- [ ] **Challenge 2 HPO**
  ```bash
  python scripts/hyperparameter_search.py \
      --challenge challenge2 \
      --n_trials 50 \
      --search_space config/hpo_challenge2.yaml
  ```
  - [ ] Optimize multi-task weights
  - [ ] Optimize uncertainty weighting
  - [ ] Optimize age normalization

### Model Ensemble Strategy

- [ ] **Train Multiple Models**
  - [ ] Model 1: Transformer (current)
  - [ ] Model 2: EEGNet
  - [ ] Model 3: TSCeption
  - [ ] Model 4: Hybrid CNN-Transformer
  - [ ] Model 5: Different random seeds

- [ ] **Implement Ensemble**
  ```python
  # Weighted averaging of predictions
  ensemble_pred = sum(w_i * pred_i for i, (w_i, pred_i) in enumerate(models))
  ```

- [ ] **Optimize Ensemble Weights**
  - [ ] Use validation set
  - [ ] Test different weighting strategies

---

## �� NICE TO HAVE - Week 4 (Polish & Submission)

### Feature Engineering

- [ ] **Challenge 1 Features**
  - [ ] Frontal theta power (decision-making)
  - [ ] P300 amplitude (cognitive control)
  - [ ] Motor beta (response preparation)
  - [ ] Alpha suppression (attention)

- [ ] **Challenge 2 Features**
  - [ ] Frontal alpha asymmetry (depression)
  - [ ] Theta/beta ratio (ADHD)
  - [ ] Alpha peak frequency (cognition)
  - [ ] Sample entropy (complexity)

### Explainability & Debugging

- [ ] **Attention Visualization**
  ```python
  # Visualize which time points matter
  visualize_attention_weights(model, sample)
  ```

- [ ] **Feature Importance**
  ```python
  # Which channels/frequencies are important
  compute_feature_importance(model, dataset)
  ```

- [ ] **Error Analysis**
  - [ ] Analyze failure cases
  - [ ] Identify systematic errors
  - [ ] Fix specific weaknesses

### Final Submission Preparation

- [ ] **Generate Submission Files**
  ```bash
  python src/evaluation/submission.py \
      --model_path checkpoints/best_model.pt \
      --output_dir submissions/
  ```

- [ ] **Validate Submission Format**
  ```bash
  python src/evaluation/submission.py \
      --validate \
      --submission_file submissions/predictions.csv
  ```

- [ ] **Test on Public Leaderboard**
  - [ ] Submit baseline model
  - [ ] Analyze leaderboard feedback
  - [ ] Iterate based on results

- [ ] **Final Ensemble Submission**
  - [ ] Combine best models
  - [ ] Generate final predictions
  - [ ] Submit before deadline

---

## 📊 Success Metrics & Targets

### Minimum Viable Performance (MVP)
- [ ] Challenge 1: Combined score >0.50
- [ ] Challenge 2: Average correlation >0.25
- [ ] Inference: <50ms per window
- [ ] Robustness: Works on all sites

### Competitive Performance (Target)
- [ ] Challenge 1: Combined score >0.60
- [ ] Challenge 2: Average correlation >0.35
- [ ] Top 25% on public leaderboard

### Winning Performance (Stretch Goal)
- [ ] Challenge 1: Combined score >0.70
- [ ] Challenge 2: Average correlation >0.40
- [ ] Top 3 on final leaderboard

---

## 📝 Daily Checklist Template

Copy this for each day:

```markdown
### Day X - [Date]

**Goal**: [What you want to accomplish today]

Morning:
- [ ] Task 1
- [ ] Task 2

Afternoon:
- [ ] Task 3
- [ ] Task 4

Evening:
- [ ] Review progress
- [ ] Update this checklist
- [ ] Plan tomorrow

**Blockers**: [Any issues encountered]
**Learnings**: [Key insights from today]
**Tomorrow**: [Top priority for next day]
```

---

## 🚨 Risk Management

### High-Risk Items (Monitor Closely)
- [ ] **Data Download Time**: Full dataset is 500GB
  - *Mitigation*: Start with test subset
  
- [ ] **Inference Speed**: <50ms is challenging
  - *Mitigation*: Early optimization + testing
  
- [ ] **Cross-Site Generalization**: May fail on new sites
  - *Mitigation*: Leave-one-site-out CV
  
- [ ] **Weak EEG-Psych Correlation**: Clinical prediction is hard
  - *Mitigation*: Feature engineering + ensemble

### Medium-Risk Items (Watch For)
- [ ] Overfitting to public leaderboard
- [ ] Hyperparameter tuning taking too long
- [ ] Model ensemble complexity
- [ ] Submission format errors

---

## 🎯 Competition Timeline

### Week 1 (Now - Day 7)
- **Focus**: Data acquisition & validation
- **Deliverable**: Working data pipeline
- **Success**: All validation tests pass

### Week 2 (Day 8-14)
- **Focus**: Baseline training & optimization
- **Deliverable**: First competition submissions
- **Success**: Public leaderboard position established

### Week 3 (Day 15-21)
- **Focus**: Advanced optimization & ensemble
- **Deliverable**: Improved models & submissions
- **Success**: Top 25% on leaderboard

### Week 4 (Day 22-28)
- **Focus**: Final polish & submission
- **Deliverable**: Best ensemble model
- **Success**: Final submission complete

---

## 📞 Quick Reference Commands

```bash
# Data acquisition
./scripts/download_hbn_data.sh

# Validation
export HBN_DATA_PATH="/path/to/data"
python tests/test_data_acquisition.py

# Training Challenge 1
python src/training/train_cross_task.py --config configs/enhanced.yaml

# Training Challenge 2
python src/training/train_psych.py --config configs/enhanced.yaml

# Inference speed test
python tests/test_inference_speed.py

# Generate submission
python src/evaluation/submission.py --model_path checkpoints/best.pt

# Validate submission
python src/evaluation/submission.py --validate --submission_file predictions.csv
```

---

## ✅ Completion Tracking

**Overall Progress**: [ ] 0% → [ ] 25% → [ ] 50% → [ ] 75% → [ ] 100%

- Critical Tasks: 0/5 ⭕
- High Priority: 0/15 ⭕  
- Medium Priority: 0/10 ⭕
- Nice to Have: 0/8 ⭕

**Last Updated**: [Date]  
**Current Blocker**: [What's blocking progress]  
**Next Milestone**: [What's the next big goal]

---

Good luck with the challenge! 🚀
Remember: **Start with data acquisition - nothing else matters without real data!**
