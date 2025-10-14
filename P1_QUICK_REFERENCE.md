# 🚀 P1 Tasks - Quick Reference Guide

## 📋 One-Line Commands

### Train Baseline Models
```bash
# Challenge 1 (Classification)
python scripts/train_baseline.py --challenge 1 --model random_forest

# Challenge 2 (Regression)
python scripts/train_baseline.py --challenge 2 --model random_forest --target p_factor
```

### Artifact Detection
```bash
# Single subject
python scripts/artifact_detection.py --subject NDARAC904DMU --task RestingState

# All subjects
python scripts/artifact_detection.py --all-subjects
```

### Cross-Site Validation
```bash
# Leave-one-site-out
python scripts/cross_site_validation.py --challenge 2 --strategy leave_one_site_out

# Grouped K-fold
python scripts/cross_site_validation.py --challenge 2 --strategy grouped_k_fold --cv-folds 5
```

### Hyperparameter Optimization
```bash
# Quick search (10 trials)
python scripts/hyperparameter_optimization.py --challenge 2 --model random_forest --n-trials 10

# Full search (100 trials)
python scripts/hyperparameter_optimization.py --challenge 2 --model random_forest --n-trials 100
```

---

## 🧪 Run All Tests
```bash
pytest tests/test_p1_scripts.py -v
```

---

## 📂 File Locations

| Script | Path |
|--------|------|
| Baseline Training | `scripts/train_baseline.py` |
| Artifact Detection | `scripts/artifact_detection.py` |
| Cross-Site Validation | `scripts/cross_site_validation.py` |
| Hyperparameter Optimization | `scripts/hyperparameter_optimization.py` |
| Tests | `tests/test_p1_scripts.py` |

---

## 📊 Output Directories

| Task | Output Directory |
|------|------------------|
| Baseline Models | `outputs/baselines/` |
| Artifact Detection | `data/processed/` |
| Cross-Site Validation | `outputs/cross_site/` |
| Hyperparameter Optimization | `outputs/hyperopt/` |

---

## 🎯 Model Choices

| Task | Models Available |
|------|------------------|
| Classification | `logistic`, `random_forest`, `svm`, `mlp` |
| Regression | `linear`, `ridge`, `random_forest`, `svm`, `mlp` |

---

## 📈 Metrics

| Challenge | Metrics |
|-----------|---------|
| Challenge 1 (Classification) | Accuracy, AUROC |
| Challenge 2 (Regression) | MSE, RMSE, R², Pearson r |

---

## ⚡ Quick Examples

### Complete Workflow
```bash
# 1. Clean data
python scripts/artifact_detection.py --all-subjects

# 2. Find best hyperparameters
python scripts/hyperparameter_optimization.py --challenge 2 --model random_forest --n-trials 50

# 3. Validate cross-site performance
python scripts/cross_site_validation.py --challenge 2 --strategy leave_one_site_out

# 4. Train final model
python scripts/train_baseline.py --challenge 2 --model random_forest --target p_factor
```

---

## 🔧 Common Options

### All Scripts
- `--help` - Show help message
- `--bids-root` - BIDS dataset path (default: `data/raw/hbn`)
- `--output-dir` - Output directory

### Model Training
- `--challenge {1,2}` - Challenge number
- `--model {logistic,random_forest,...}` - Model type
- `--target {p_factor,attention,...}` - Target for Challenge 2
- `--feature-method {psd,stats,raw_mean}` - Feature extraction

### Cross-Site Validation
- `--strategy {leave_one_site_out,grouped_k_fold}` - CV strategy
- `--cv-folds N` - Number of folds (for grouped k-fold)

### Hyperparameter Optimization
- `--n-trials N` - Number of optimization trials
- `--metric {r2,pearson_r,neg_mse}` - Metric to optimize

---

## 🐛 Troubleshooting

### Import Errors
```bash
# Install missing dependencies
python -m pip install --break-system-packages scipy optuna mne mne-bids
```

### Memory Issues (ICA)
```bash
# Reduce number of ICA components
# Edit artifact_detection.py: n_components=20 → n_components=10
```

### Slow Optimization
```bash
# Reduce number of trials
python scripts/hyperparameter_optimization.py ... --n-trials 10
```

---

## 📞 Getting Help

```bash
# View script help
python scripts/train_baseline.py --help
python scripts/artifact_detection.py --help
python scripts/cross_site_validation.py --help
python scripts/hyperparameter_optimization.py --help
```

---

**Quick Start**: Run `pytest tests/test_p1_scripts.py -v` to verify everything works!
