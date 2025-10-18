# Quick Start TODO Checklist

**Last Updated**: October 14, 2025  
**Status**: Ready to Execute

---

## Week 1: Critical Foundation (Days 1-7)

### Day 1-2: Data Acquisition 🔴 BLOCKING

```markdown
- [ ] **Register for HBN dataset access**
  - Visit: https://www.nitrc.org/account/register.php
  - Submit institutional review
  - Wait for approval (24-48 hours)
  
- [ ] **Set up data directory structure**
  ```bash
  mkdir -p data/raw/hbn
  mkdir -p data/processed
  mkdir -p data/reports
  ```

- [ ] **Start sample data download** (don't wait for full approval)
  ```bash
  python scripts/download_hbn_data.py \
      --output_dir data/raw/hbn_sample \
      --max_subjects 10 \
      --tasks sus
  ```

- [ ] **Verify sample data integrity**
  ```bash
  python scripts/verify_data_integrity.py \
      --data_dir data/raw/hbn_sample
  ```
```

### Day 3-4: Testing Infrastructure 🔴 CRITICAL

```markdown
- [ ] **Create critical test files**
  - [ ] tests/test_data_loading.py
  - [ ] tests/test_model_forward.py
  - [ ] tests/test_official_metrics.py
  - [ ] tests/test_inference_speed.py

- [ ] **Run initial tests on sample data**
  ```bash
  pytest tests/test_data_loading.py -v
  pytest tests/test_model_forward.py -v
  ```

- [ ] **Set up continuous testing**
  ```bash
  # Add to .github/workflows/ci.yml
  - name: Run pytest
    run: pytest tests/ -v --cov=src
  ```
```

### Day 5-7: Baseline Model Training 🟡 HIGH

```markdown
- [ ] **Train first baseline model (Challenge 1)**
  ```bash
  python src/training/train_cross_task.py \
      --data_dir data/raw/hbn_sample \
      --epochs 10 \
      --batch_size 16 \
      --output_dir outputs/baseline_challenge1
  ```

- [ ] **Evaluate baseline performance**
  - Expected RT correlation: 0.1-0.3 (random is ~0)
  - Expected Success AUROC: 0.55-0.65 (random is 0.5)

- [ ] **Train baseline model (Challenge 2)**
  ```bash
  python src/training/train_psych.py \
      --data_dir data/raw/hbn_sample \
      --epochs 10 \
      --batch_size 16 \
      --output_dir outputs/baseline_challenge2
  ```

- [ ] **Document baseline results**
  - Save metrics to outputs/baseline_metrics.json
  - Create visualization of training curves
```

---

## Week 2: Optimization & Validation (Days 8-14)

### Day 8-9: Artifact Detection 🟡 HIGH

```markdown
- [ ] **Implement artifact detection module**
  - [ ] Eye blink detection
  - [ ] Muscle artifact detection  
  - [ ] Bad channel detection
  - [ ] Add to preprocessing pipeline

- [ ] **Test artifact detection**
  ```bash
  pytest tests/test_artifact_detection.py
  ```

- [ ] **Retrain with clean data**
  - Compare performance with/without artifact removal
  - Document improvement in metrics
```

### Day 10-11: Inference Optimization 🔴 CRITICAL

```markdown
- [ ] **Measure current inference latency**
  ```bash
  pytest tests/test_inference_speed.py -v
  ```

- [ ] **If >50ms, apply optimizations:**
  - [ ] Model quantization (INT8)
  - [ ] Pruning (30% sparsity)
  - [ ] ONNX export
  - [ ] TensorRT compilation

- [ ] **Verify latency meets requirement**
  - Target: <50ms average
  - Target: <75ms p95
```

### Day 12-14: Cross-Site Validation 🟡 HIGH

```markdown
- [ ] **Implement leave-one-site-out CV**
  ```python
  sites = ['RU', 'CBIC', 'SI', 'CUNY']
  for held_out_site in sites:
      train_sites = [s for s in sites if s != held_out_site]
      # Train and evaluate
  ```

- [ ] **Check domain adaptation effectiveness**
  - Performance should be consistent across sites
  - If variance >20%, investigate DANN effectiveness

- [ ] **Document cross-site results**
  - Create site-wise performance table
  - Identify problematic sites
```

---

## Week 3: Advanced Optimization (Days 15-21)

### Day 15-17: Hyperparameter Optimization

```markdown
- [ ] **Set up HPO framework**
  ```bash
  pip install optuna
  ```

- [ ] **Run HPO for Challenge 1**
  ```bash
  python scripts/hyperparameter_search.py \
      --challenge challenge1 \
      --n_trials 100 \
      --output hpo_results/challenge1
  ```

- [ ] **Run HPO for Challenge 2**
  ```bash
  python scripts/hyperparameter_search.py \
      --challenge challenge2 \
      --n_trials 100 \
      --output hpo_results/challenge2
  ```

- [ ] **Train with best hyperparameters**
  - Retrain from scratch with optimal config
  - Validate improvement over baseline
```

### Day 18-20: Model Ensemble

```markdown
- [ ] **Train ensemble models**
  - [ ] Train 5 models with different seeds
  - [ ] Train 3 different architectures (Transformer, CNN, Hybrid)
  - [ ] Train on different data splits

- [ ] **Implement ensemble prediction**
  ```python
  predictions = []
  for model in ensemble:
      pred = model(x)
      predictions.append(pred)
  
  final_pred = weighted_average(predictions)
  ```

- [ ] **Validate ensemble improvement**
  - Should see 2-5% improvement over single model
```

### Day 21: Full Dataset Training

```markdown
- [ ] **Download full HBN dataset** (if not already done)
  ```bash
  python scripts/download_hbn_data.py \
      --output_dir data/raw/hbn \
      --tasks sus ccd rest
  ```

- [ ] **Train final models on full data**
  - Challenge 1: SuS task data
  - Challenge 2: Resting state data

- [ ] **Validate on held-out test set**
  - Do NOT use test set until this point
  - Compare to validation set performance
```

---

## Week 4: Submission Preparation (Days 22-28)

### Day 22-24: Submission Generation

```markdown
- [ ] **Generate Challenge 1 predictions**
  ```bash
  python scripts/generate_submission.py \
      --challenge 1 \
      --model_path outputs/challenge1/best_model.pt \
      --data_dir data/raw/hbn \
      --output submissions/challenge1_predictions.csv
  ```

- [ ] **Generate Challenge 2 predictions**
  ```bash
  python scripts/generate_submission.py \
      --challenge 2 \
      --model_path outputs/challenge2/best_model.pt \
      --data_dir data/raw/hbn \
      --output submissions/challenge2_predictions.csv
  ```

- [ ] **Validate submission format**
  ```bash
  python scripts/validate_submission.py \
      --submission submissions/challenge1_predictions.csv \
      --challenge 1
  ```
```

### Day 25-26: Final Validation

```markdown
- [ ] **Run all tests one final time**
  ```bash
  pytest tests/ -v --cov=src
  ```

- [ ] **Check inference latency**
  - Must be <50ms
  - Run on production hardware if available

- [ ] **Validate submission files**
  - Correct format (CSV with required columns)
  - No missing values
  - Predictions in valid range

- [ ] **Create submission documentation**
  - Model description
  - Training procedure
  - Hyperparameters used
  - Expected performance
```

### Day 27-28: Submit & Monitor

```markdown
- [ ] **Submit to competition platform**
  - Upload submission files
  - Fill out metadata form
  - Note submission timestamp

- [ ] **Monitor leaderboard**
  - Check public leaderboard score
  - Compare to validation performance
  - Identify overfitting/underfitting

- [ ] **Iterate if needed**
  - If score is much worse than expected, investigate
  - Consider retraining or ensemble adjustment
  - Submit updated version if time permits
```

---

## Emergency Fallback Plan

If behind schedule:

```markdown
**Priority 1 (Must Have)**
- [ ] Basic data loading working
- [ ] Model trains without crashing
- [ ] Can generate submission files
- [ ] Submission format is valid

**Priority 2 (Should Have)**
- [ ] Inference <50ms
- [ ] Basic cross-validation
- [ ] Artifact detection working

**Priority 3 (Nice to Have)**
- [ ] Hyperparameter optimization
- [ ] Model ensemble
- [ ] Advanced domain adaptation
```

---

## Daily Checklist Template

Copy this for each day:

```markdown
**Date**: ___________
**Focus**: ___________

Morning (3-4 hours):
- [ ] Task 1
- [ ] Task 2

Afternoon (3-4 hours):
- [ ] Task 3
- [ ] Task 4

End of Day:
- [ ] Commit code changes
- [ ] Update progress log
- [ ] Note blockers/issues
- [ ] Plan next day

Blockers:
- 

Notes:
- 
```

---

## Success Criteria

By end of Week 4, you should have:

✅ Full HBN dataset downloaded and validated  
✅ All critical tests passing (>80% coverage)  
✅ Inference latency <50ms  
✅ Cross-site validation completed  
✅ Final models trained on full dataset  
✅ Submission files generated and validated  
✅ Competition submission uploaded  
✅ Initial leaderboard score received

---

## Resources & Contacts

**Documentation**:
- Part 1: docs/01_DATA_ACQUISITION_GUIDE.md
- Part 2: docs/02_VALIDATION_TESTING_GUIDE.md
- Part 3: This file (03_QUICK_START_TODO.md)

**Support**:
- Competition forum: [TBD]
- GitHub issues: github.com/hkevin01/eeg2025/issues
- HBN support: hbn@childmind.org

**Tracking**:
- Daily progress: docs/daily_log.md
- Issues: github.com/hkevin01/eeg2025/issues
- Metrics: outputs/metrics_tracking.json

