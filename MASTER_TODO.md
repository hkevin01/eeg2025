# EEG2025 Master TODO

**Last Updated:** October 15, 2025  
**Project Status:** Phase 2 - Advanced Development  
**GPU Status:** CPU-Only Mode (AMD RX 5600 XT Unstable)

---

## 📊 Project Overview

**Competition:** EEG Foundation Challenge 2025  
**Challenges:**
- Challenge 1: Age Prediction (Regression)
- Challenge 2: Sex Classification (Binary)

**Current Approach:** CPU-based training with advanced neural architectures

---

## 🎯 Current Phase Status

### ✅ Phase 1: Infrastructure & GPU Analysis (COMPLETE)

- [x] GPU environment setup and testing
- [x] AMD RX 5600 XT stability analysis
- [x] CPU-only fallback implementation
- [x] Safe training scripts created
- [x] Comprehensive documentation

**Outcome:** GPU unstable, CPU-only mode adopted

### 🔄 Phase 2: Advanced Pipeline Development (IN PROGRESS)

- [x] Multi-scale CNN architecture
- [x] Data augmentation pipeline
- [x] Advanced preprocessing
- [x] Test-time augmentation
- [x] Improved inference pipeline
- [ ] Train baseline models ← **CURRENT**
- [ ] Hyperparameter optimization
- [ ] Model ensemble creation

### ⭕ Phase 3: Competition Preparation (NOT STARTED)

- [ ] Full dataset training
- [ ] Cross-validation implementation
- [ ] Final model selection
- [ ] Submission preparation
- [ ] Documentation finalization

---

## 🚀 Immediate Next Steps (Priority Order)

### 1. ⚡ Train Baseline Models (HIGH PRIORITY)

**Goal:** Establish performance baselines for both challenges

#### Challenge 1: Age Prediction
```bash
# Train improved model
cd /home/kevin/Projects/eeg2025
python scripts/train_improved_cpu.py

# Monitor training
tail -f logs/training.log
cat checkpoints/training_history.json
```

**Targets:**
- [ ] Train on 1000 samples (quick baseline)
- [ ] Achieve correlation > 0.40
- [ ] Save best checkpoint
- [ ] Document performance

#### Challenge 2: Sex Classification
```bash
# Modify config in train_improved_cpu.py
# Set challenge = 2
python scripts/train_improved_cpu.py
```

**Targets:**
- [ ] Train on 1000 samples
- [ ] Achieve accuracy > 75%
- [ ] Save best checkpoint
- [ ] Document performance

**Estimated Time:** 2-3 hours per challenge

---

### 2. 🔧 Preprocessing Pipeline Integration

**Goal:** Apply advanced preprocessing to improve data quality

**Tasks:**
- [ ] Run preprocessing on HBN dataset
  ```bash
  python scripts/preprocess_advanced.py
  ```
- [ ] Save preprocessed data to disk
- [ ] Measure preprocessing impact on performance
- [ ] Document preprocessing parameters

**Files:**
- `scripts/preprocess_advanced.py` - Preprocessing implementation
- `data/processed/` - Output location

**Estimated Time:** 1-2 hours

---

### 3. 📈 Hyperparameter Optimization

**Goal:** Find optimal training parameters

**Parameters to Optimize:**
- Learning rate: [1e-5, 1e-3]
- Batch size: [8, 16, 32]
- Dropout: [0.1, 0.3, 0.5]
- Model depth: [4, 6, 8 layers]

**Tasks:**
- [ ] Create hyperparameter search script
- [ ] Run grid search or Bayesian optimization
- [ ] Log all experiments
- [ ] Select best configuration

**Estimated Time:** 4-6 hours (CPU-intensive)

---

### 4. �� Model Ensemble Creation

**Goal:** Combine multiple models for better predictions

**Tasks:**
- [ ] Train 3-5 models with different:
  - Random seeds
  - Architectures
  - Augmentation strategies
- [ ] Implement ensemble inference
- [ ] Test ensemble on validation set
- [ ] Compare with single model

**Code:**
```python
# Use inference_improved.py ensemble support
from scripts.inference_improved import ModelEnsemble

ensemble = ModelEnsemble([
    'checkpoints/model1.pth',
    'checkpoints/model2.pth',
    'checkpoints/model3.pth'
])
```

**Estimated Time:** 6-8 hours

---

### 5. 📊 Validation & Evaluation

**Goal:** Comprehensive model evaluation

**Tasks:**
- [ ] Implement k-fold cross-validation
- [ ] Calculate confidence intervals
- [ ] Generate performance plots
- [ ] Create evaluation report

**Metrics to Track:**
- Challenge 1: Pearson correlation, MAE, RMSE
- Challenge 2: Accuracy, F1, ROC-AUC

**Estimated Time:** 2-3 hours

---

## 📁 Project Organization

### Root Folder Structure

```
eeg2025/
├── README.md              # Project overview
├── MASTER_TODO.md         # This file
├── QUICK_START_SAFE.md    # Quick start guide
├── CHANGELOG.md           # Version history
├── LICENSE                # License file
├── Makefile               # Build automation
├── requirements.txt       # Python dependencies
├── setup.py               # Package setup
│
├── archive/               # Archived files
│   ├── status_reports/    # Old status files
│   ├── old_todos/         # Deprecated TODOs
│   └── old_configs/       # Old configurations
│
├── checkpoints/           # Model checkpoints
│   ├── best.pth          # Best model
│   ├── latest.pth        # Latest checkpoint
│   └── training_history.json
│
├── data/                  # Dataset storage
│   ├── raw/              # Raw EEG data
│   ├── processed/        # Preprocessed data
│   └── features/         # Extracted features
│
├── docs/                  # Documentation
│   ├── PIPELINE_IMPROVEMENTS.md
│   ├── GPU_OPTIMIZATION_SUMMARY.md
│   └── api/              # API documentation
│
├── notebooks/             # Jupyter notebooks
│   └── exploratory/      # Data exploration
│
├── outputs/               # Training outputs
│   ├── figures/          # Generated plots
│   └── results/          # Prediction results
│
├── scripts/               # Executable scripts
│   ├── train_improved_cpu.py      # Main training
│   ├── inference_improved.py      # Inference with TTA
│   ├── preprocess_advanced.py     # Preprocessing
│   ├── test_cpu_minimal.py        # Quick test
│   │
│   ├── monitoring/        # Monitoring scripts
│   ├── testing/           # Validation scripts
│   └── gpu_tests/         # GPU test scripts (unsafe)
│
├── src/                   # Source code
│   ├── gpu/              # GPU optimization modules
│   ├── models/           # Model architectures
│   └── utils/            # Utility functions
│
└── tests/                 # Unit tests
    └── test_*.py         # Test files
```

---

## 🔧 Development Workflow

### Daily Development Cycle

1. **Morning: Training**
   ```bash
   # Start training session
   python scripts/train_improved_cpu.py
   
   # Monitor progress
   tail -f logs/training.log
   ```

2. **Afternoon: Evaluation**
   ```bash
   # Run inference
   python scripts/inference_improved.py
   
   # Check results
   cat outputs/results/predictions.csv
   ```

3. **Evening: Analysis**
   ```bash
   # Review training history
   python -c "
   import json
   with open('checkpoints/training_history.json') as f:
       history = json.load(f)
   # Analyze metrics
   "
   ```

---

## 📚 Key Files & Their Purpose

### Training Scripts
| File | Purpose | Status |
|------|---------|--------|
| `scripts/train_improved_cpu.py` | Advanced training with augmentation | ✅ Ready |
| `scripts/train_cpu_only_safe.py` | Basic safe training | ✅ Ready |
| `scripts/test_cpu_minimal.py` | Quick validation test | ✅ Ready |

### Inference Scripts
| File | Purpose | Status |
|------|---------|--------|
| `scripts/inference_improved.py` | TTA & ensemble inference | ✅ Ready |

### Preprocessing Scripts
| File | Purpose | Status |
|------|---------|--------|
| `scripts/preprocess_advanced.py` | Advanced preprocessing | ✅ Ready |

### Documentation
| File | Purpose | Status |
|------|---------|--------|
| `docs/PIPELINE_IMPROVEMENTS.md` | Pipeline documentation | ✅ Complete |
| `docs/GPU_OPTIMIZATION_SUMMARY.md` | GPU issues & solutions | ✅ Complete |
| `QUICK_START_SAFE.md` | Quick start guide | ✅ Complete |

---

## ⚠️ Important Notes

### GPU Usage - DO NOT USE

**Your AMD Radeon RX 5600 XT is UNSTABLE for deep learning!**

❌ **Never run these:**
- `scripts/gpu_tests/*` - Any GPU test scripts
- `scripts/train_*_gpu.py` - GPU training scripts
- `scripts/train_amd_*.py` - AMD-specific scripts

✅ **Always use these:**
- `scripts/train_improved_cpu.py` - Safe CPU training
- `scripts/train_cpu_only_safe.py` - Basic CPU training
- `scripts/test_cpu_minimal.py` - CPU validation

**Symptoms of GPU issues:**
- Checkerboard/RGB visual artifacts
- System freezes
- Display corruption
- Desktop crashes

**If you see these → Ctrl+C immediately!**

---

## 🎯 Success Metrics

### Challenge 1: Age Prediction

| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| Correlation | 0.35-0.45 | 0.50-0.60 | 0.65+ |
| MAE | 8-10 years | 6-8 years | <6 years |
| RMSE | 10-12 years | 8-10 years | <8 years |

### Challenge 2: Sex Classification

| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| Accuracy | 70-75% | 78-83% | 85%+ |
| F1 Score | 0.65-0.70 | 0.75-0.80 | 0.85+ |
| ROC-AUC | 0.75-0.80 | 0.82-0.87 | 0.90+ |

---

## 🔄 Weekly Goals

### Week 1 (Current): Baseline Training
- [x] Setup infrastructure
- [x] Create advanced pipeline
- [ ] Train baseline models
- [ ] Initial evaluation

### Week 2: Optimization
- [ ] Hyperparameter tuning
- [ ] Model ensemble
- [ ] Preprocessing integration
- [ ] Performance analysis

### Week 3: Scale-Up
- [ ] Full dataset training
- [ ] Cross-validation
- [ ] Model selection
- [ ] Documentation

### Week 4: Competition Prep
- [ ] Final training
- [ ] Submission preparation
- [ ] Code cleanup
- [ ] Final documentation

---

## 💡 Tips & Best Practices

### Training Tips
1. **Start Small:** Train on 500-1000 samples first
2. **Monitor RAM:** Keep usage under 80%
3. **Save Frequently:** Checkpoint every epoch
4. **Use Augmentation:** Always enable for training
5. **Early Stopping:** Be patient (5-10 epochs)

### Debugging Tips
1. **Check Data:** Verify labels are correct
2. **Watch Gradients:** Ensure they're not exploding
3. **Monitor Metrics:** Train and val should be close
4. **Log Everything:** You can't debug what you can't see
5. **Test Incrementally:** Small changes, frequent tests

### Performance Tips
1. **CPU Cores:** Use `num_workers=2-4` for DataLoader
2. **Batch Size:** 16-32 for 16GB RAM, 32-64 for 32GB RAM
3. **Precision:** FP32 only (CPU doesn't benefit from FP16)
4. **Caching:** Preprocess once, reuse many times
5. **Parallel:** Train multiple models simultaneously if RAM allows

---

## 🆘 Troubleshooting

### Common Issues

**Issue:** Training is slow
- **Solution:** Reduce batch size, decrease model complexity, use fewer workers

**Issue:** Out of memory
- **Solution:** Reduce batch size to 8, limit max_samples to 500

**Issue:** Model not learning
- **Solution:** Check learning rate (try 1e-3 to 1e-5), verify data labels, check loss function

**Issue:** Overfitting
- **Solution:** Increase dropout, add more augmentation, reduce model size

**Issue:** Validation worse than training
- **Solution:** Enable dropout during inference, check for data leakage, use early stopping

---

## 📞 Quick Commands Reference

```bash
# Training
python scripts/train_improved_cpu.py                    # Full training
python scripts/train_cpu_only_safe.py                   # Simple training
python scripts/test_cpu_minimal.py                      # Quick test

# Inference
python scripts/inference_improved.py                    # Run inference

# Preprocessing
python scripts/preprocess_advanced.py                   # Preprocess data

# Monitoring
tail -f logs/training.log                              # Watch training
cat checkpoints/training_history.json | python -m json.tool  # View history
htop                                                    # Monitor CPU/RAM
```

---

## 🎓 Learning Resources

### EEG Processing
- MNE-Python documentation
- EEG signal processing tutorials
- Brain oscillations papers

### Deep Learning
- PyTorch documentation
- Attention mechanisms papers
- Data augmentation strategies

### Competition
- EEG Foundation Challenge website
- HBN dataset documentation
- Previous competition solutions

---

## 📝 Notes

- All training is CPU-only for stability
- Expected training time: 15-20 min/epoch
- Use ensemble of 3-5 models for best results
- Test-time augmentation adds ~2-3% performance
- Preprocessing can improve results by 5-10%

---

## ✅ Checklist for Competition Submission

- [ ] Train final models (3-5 models)
- [ ] Run cross-validation
- [ ] Generate predictions with TTA
- [ ] Create ensemble predictions
- [ ] Verify submission format
- [ ] Write model documentation
- [ ] Prepare code repository
- [ ] Create README for submission
- [ ] Test submission files
- [ ] Submit to competition platform

---

**Last Action:** Organized root folder, moved files to appropriate subfolders  
**Next Action:** Train baseline models for both challenges  
**Estimated Completion:** 2-3 hours

