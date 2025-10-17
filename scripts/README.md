# 📂 Scripts Directory - Navigation Guide

This directory contains all executable scripts for the EEG2025 Foundation Model project, organized by function.

## 🗂️ Directory Structure

```
scripts/
├── README.md                    ← You are here
├── REORGANIZATION_PLAN.md       ← Organization plan and rationale
├── analysis/                    ← Data analysis and results visualization
├── data/                        ← Data processing and extraction
├── deprecated/                  ← Old scripts kept for reference (8 scripts)
├── features/                    ← Feature engineering (3 scripts)
├── gpu_tests/                   ← GPU testing and benchmarking
├── inference/                   ← Model inference and predictions (6 scripts)
├── launchers/                   ← Shell scripts to start training/jobs (6 scripts)
├── models/                      ← Model definitions and architecture tests (3 scripts)
├── monitoring/                  ← Training monitoring and logging (14 scripts)
├── testing/                     ← Unit tests and integration tests (14 scripts)
├── training/                    ← Training scripts organized by challenge
│   ├── challenge1/              ← Challenge 1 (Response Time) training
│   ├── challenge2/              ← Challenge 2 (Externalizing) training
│   └── common/                  ← Shared training utilities
├── utilities/                   ← Helper scripts and status checks (5 scripts)
└── validation/                  ← Validation and evaluation scripts (7 scripts)
```

---

## 🚀 Quick Start

### Training

#### Challenge 1 (Response Time Prediction)
```bash
# Train with sparse attention architecture (BEST)
python training/challenge1/train_challenge1_attention.py

# Or use the launcher
bash launchers/launch_training.sh challenge1
```

#### Challenge 2 (Externalizing Factor Prediction)
```bash
# Train with multi-release strategy
python training/challenge2/train_challenge2_multi_release.py

# Monitor progress
bash monitoring/monitor_training.sh challenge2
```

### Monitoring

```bash
# Quick status check
bash utilities/status.sh

# Watch training in real-time
bash monitoring/monitor_training.sh

# Monitor specific challenge
bash monitoring/monitor_training_enhanced.sh challenge2
```

### Validation

```bash
# Validate all components
python validation/validate_enhancements.py

# Simple validation
python validation/simple_validation.py

# Cross-task metrics
python testing/test_cross_task_simple.py
```

### Inference

```bash
# Run inference on test set
python inference/run_inference.py

# Create submission package
python inference/create_submission.py
```

---

## 📁 Detailed Directory Descriptions

### 📊 `analysis/` - Data Analysis
Currently empty - reserved for future analysis scripts.

**Intended Purpose:**
- Results visualization
- Performance analysis
- Data exploration notebooks
- Statistical analysis scripts

### 💾 `data/` - Data Processing
Scripts for loading, processing, and managing EEG datasets.

**Key Scripts:**
- Dataset loading utilities
- Data preprocessing pipelines
- Metadata extraction
- P300 event detection

### 🗑️ `deprecated/` - Old Scripts (8 files)
Scripts that are no longer actively used but kept for reference.

**Contents:**
- Old training scripts
- Superseded architectures
- Initial experiments

**Note:** Don't use these unless you know what you're doing!

### 🧬 `features/` - Feature Engineering (3 files)
Feature extraction and engineering scripts.

**Key Features:**
- Frequency band power extraction
- Event-related potentials (ERPs)
- Temporal features
- Spatial features

### 🎮 `gpu_tests/` - GPU Testing
GPU benchmarking and testing scripts.

**Key Tests:**
- ROCm compatibility checks
- Performance benchmarks
- Memory usage tests
- Triton kernel tests

### 🔮 `inference/` - Model Inference (6 files)
Scripts for running trained models on new data.

**Key Scripts:**
- `run_inference.py` - Run model on test set
- `create_submission.py` - Package for Codabench
- Batch inference utilities
- Prediction aggregation

### 🚀 `launchers/` - Job Launchers (6 files)
Shell scripts to start training jobs and long-running processes.

**Available Launchers:**
- `launch_training.sh` - Main training launcher
- `restart_training.sh` - Restart interrupted training
- `restart_training_cpu.sh` - CPU-only restart
- `restart_training_hybrid.sh` - Hybrid CPU/GPU restart
- `run_overnight_training.sh` - Overnight training setup
- `start_independent_training.sh` - Independent training session

**Usage Example:**
```bash
cd /home/kevin/Projects/eeg2025
bash scripts/launchers/launch_training.sh
```

### 🏗️ `models/` - Model Definitions (3 files)
Model architecture definitions and tests.

**Key Files:**
- Architecture prototypes
- Model component tests
- Custom layer implementations

### 📈 `monitoring/` - Training Monitoring (14 files)
Scripts for monitoring training progress and logging.

**Key Monitors:**
- `monitor_training.sh` - General training monitor
- `monitor_training_enhanced.sh` - Enhanced with metrics
- `monitor_p300_extraction.sh` - P300 extraction monitor
- `monitor_debug.sh` - Debug-level monitoring
- `monitor_simple.sh` - Simple progress check
- `watch_p300.sh` - Continuous P300 watching

**Usage Example:**
```bash
# Monitor in real-time
tail -f logs/challenge1_training.log

# Or use our monitor
bash scripts/monitoring/monitor_training_enhanced.sh
```

### 🧪 `testing/` - Tests (14 files)
Unit tests, integration tests, and validation scripts.

**Test Categories:**
1. **Unit Tests** - Individual component testing
2. **Integration Tests** - System-level testing
3. **Demo Tests** - End-to-end demo validation
4. **Cross-Task Tests** - Multi-challenge validation

**Run All Tests:**
```bash
# Quick validation
python scripts/testing/simple_validation.py

# Comprehensive testing
python scripts/testing/test_cross_task_simple.py
```

### 🎓 `training/` - Training Scripts
Organized by challenge and common utilities.

#### `training/challenge1/` - Challenge 1 Training
**Response Time Prediction (CCD Task)**

**Key Script:**
- `train_challenge1_attention.py` ⭐ **BEST MODEL**
  - Sparse attention architecture
  - 5-fold cross-validation
  - NRMSE: 0.2632 ± 0.0368

**Architecture:**
- SparseAttentionResponseTimeCNN
- 2.5M parameters
- Multi-head attention (8 heads)
- Channel attention mechanism

#### `training/challenge2/` - Challenge 2 Training
**Externalizing Factor Prediction**

**Key Script:**
- `train_challenge2_multi_release.py`
  - Multi-release training (R2+R3+R4)
  - Fixed-length window extraction
  - Target NRMSE: < 0.35

**Architecture:**
- ExternalizingCNN
- 240K parameters
- 4-layer CNN architecture

#### `training/common/` - Shared Utilities
Common training utilities used across challenges:
- Loss functions
- Optimizers
- Learning rate schedulers
- Data augmentation
- Training loops

### 🔧 `utilities/` - Helper Scripts (5 files)
General-purpose utility scripts.

**Available Utilities:**
- `status.sh` - Quick project status
- `quick_status.sh` - Even quicker status
- `check_training_status.sh` - Training progress check
- `organize_project.py` - Project organization
- `organize_files.sh` - File organization

**Usage Example:**
```bash
bash scripts/utilities/status.sh
```

### ✅ `validation/` - Validation Scripts (7 files)
Model validation and evaluation scripts.

**Key Validators:**
- Component validation
- Enhancement validation
- Simple validation
- Cross-validation utilities

---

## 🎯 Common Workflows

### Workflow 1: Train Challenge 1 from Scratch
```bash
# 1. Navigate to project root
cd /home/kevin/Projects/eeg2025

# 2. Train with sparse attention (BEST)
python scripts/training/challenge1/train_challenge1_attention.py

# 3. Monitor progress (in another terminal)
bash scripts/monitoring/monitor_training_enhanced.sh

# 4. Validate results
python scripts/validation/validate_enhancements.py

# Expected time: ~10 minutes (5 folds × ~2 min each)
# Expected NRMSE: 0.26-0.27
```

### Workflow 2: Train Challenge 2 Multi-Release
```bash
# 1. Navigate to project root
cd /home/kevin/Projects/eeg2025

# 2. Start training
python scripts/training/challenge2/train_challenge2_multi_release.py

# 3. Monitor in real-time
tail -f logs/challenge2_r234_final.log

# Expected time: ~60-90 minutes
# Expected NRMSE: 0.30-0.35
```

### Workflow 3: Create Submission Package
```bash
# 1. Ensure both models are trained
ls checkpoints/response_time_attention.pth
ls checkpoints/weights_challenge_2_multi_release.pt

# 2. Run inference
python scripts/inference/run_inference.py

# 3. Create submission
python scripts/inference/create_submission.py

# 4. Verify package
unzip -l submission.zip
```

### Workflow 4: Quick Health Check
```bash
# 1. Quick status
bash scripts/utilities/status.sh

# 2. Check for errors
python scripts/validation/simple_validation.py

# 3. Verify training logs
tail -100 logs/*.log | grep -E "(Error|Exception)"
```

---

## 📊 Directory Statistics

```
Total Scripts: 66 files
├── Launchers:     6 files (Shell scripts for starting jobs)
├── Monitoring:   14 files (Training progress tracking)
├── Testing:      14 files (Unit and integration tests)
├── Deprecated:    8 files (Old scripts, keep for reference)
├── Validation:    7 files (Model validation)
├── Inference:     6 files (Model inference and submission)
├── Utilities:     5 files (Helper scripts)
├── Features:      3 files (Feature engineering)
└── Models:        3 files (Architecture definitions)
```

---

## 🔗 Related Directories

### Project Root
```
/home/kevin/Projects/eeg2025/
├── scripts/              ← You are here
├── src/                  ← Source code (models, data loaders, etc.)
├── tests/                ← Pytest tests
├── config/               ← Configuration files
├── data/                 ← Dataset storage
├── checkpoints/          ← Trained model weights
├── logs/                 ← Training logs
├── results/              ← Evaluation results
├── submission_history/   ← Past submissions
└── docs/                 ← Documentation
```

### Source Code (`src/`)
For actual model implementations and core code, see:
- `src/models/` - Model architectures
- `src/dataio/` - Data loaders
- `src/training/` - Training loops and trainers
- `src/gpu/` - GPU acceleration (Triton kernels)

---

## 💡 Tips & Best Practices

### Running Scripts

1. **Always run from project root:**
   ```bash
   cd /home/kevin/Projects/eeg2025
   python scripts/training/challenge1/train_challenge1_attention.py
   ```

2. **Check logs after running:**
   ```bash
   tail -100 logs/challenge1_attention_YYYYMMDD_HHMMSS.log
   ```

3. **Monitor GPU usage (if available):**
   ```bash
   watch -n 1 rocm-smi
   ```

### Organizing New Scripts

When adding new scripts:
1. Identify purpose (training, testing, inference, etc.)
2. Place in appropriate subdirectory
3. Add description to this README
4. Update directory statistics
5. Consider creating symlink in root if frequently used

### Symlinks

Some scripts have symlinks for convenience:
```bash
# From project root
ln -s scripts/monitoring/monitor_training.sh monitor_training.sh
ln -s scripts/training/challenge1/train_challenge1_attention.py train.py
```

---

## 🆘 Troubleshooting

### "Script not found"
```bash
# Check if script exists
ls scripts/path/to/script.py

# If missing, search for it
find scripts/ -name "script.py"
```

### "Permission denied"
```bash
# Make script executable
chmod +x scripts/path/to/script.sh
```

### "Module not found"
```bash
# Ensure you're in project root
cd /home/kevin/Projects/eeg2025

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### "Training not starting"
```bash
# Check if already running
ps aux | grep train_challenge

# Check GPU availability
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

# Check disk space
df -h
```

---

## 📚 Additional Resources

- **Project Documentation:** `docs/`
- **Competition Rules:** `archive/COMPETITION_RULES.md`
- **Submission History:** `archive/SUBMISSION_HISTORY.md`
- **Project Analysis:** `docs/analysis/PROJECT_ANALYSIS_OCT17.md`
- **Method Description:** `METHODS_DOCUMENT.pdf`

---

## 🔄 Recent Changes

**October 17, 2025:**
- ✅ Reorganized scripts into logical subdirectories
- ✅ Created launchers/ for shell scripts
- ✅ Moved train_challenge2_multi_release.py to training/challenge2/
- ✅ Consolidated monitoring scripts
- ✅ Created this README for navigation

---

## 📞 Quick Reference

```bash
# Status check
bash scripts/utilities/status.sh

# Train Challenge 1
python scripts/training/challenge1/train_challenge1_attention.py

# Train Challenge 2
python scripts/training/challenge2/train_challenge2_multi_release.py

# Monitor training
bash scripts/monitoring/monitor_training_enhanced.sh

# Validate
python scripts/validation/simple_validation.py

# Create submission
python scripts/inference/create_submission.py
```

---

**Last Updated:** October 17, 2025  
**Maintained By:** EEG2025 Competition Team  
**Competition Deadline:** November 2, 2025 (16 days remaining)
