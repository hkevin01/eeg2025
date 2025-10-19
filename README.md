# 🧠 EEG 2025 NeurIPS Competition Solution

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://eeg2025.github.io/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Deep learning solution for EEG-based brain decoding challenges: response time prediction and behavioral factor assessment.**

**Competition:** [NeurIPS 2025 EEG Foundation Challenge](https://eeg2025.github.io/)  
**Deadline:** November 2, 2025  
**Status:** Challenge 1 Ready ✅ | Challenge 2 Training 🔄

---

## 📋 Table of Contents

- [Competition Overview](#-competition-overview)
- [Current Status](#-current-status)
- [Project Structure](#-project-structure)
- [Models](#-models)
- [Training](#-training)
- [Monitoring](#-monitoring)
- [Installation](#-installation)
- [Usage](#-usage)
- [Documentation](#-documentation)

---

## 🎯 Competition Overview

### Challenges

**Challenge 1: Cross-Task Transfer Learning**
- **Goal:** Predict response time from EEG during contrastChangeDetection task
- **Input:** 129-channel EEG, 2-second stimulus-locked windows
- **Target:** Response time (RT) in seconds
- **Metric:** NRMSE (Normalized Root Mean Square Error)
- **Weight:** 30% of final score

**Challenge 2: Subject-Invariant Representation**
- **Goal:** Predict externalizing factor (p_factor) from EEG
- **Input:** 129-channel EEG, 4-second windows with 2-second random crops
- **Target:** Clinical externalizing measure from CBCL
- **Metric:** L1 loss (Mean Absolute Error)
- **Weight:** 70% of final score
- **Emphasis:** Cross-subject generalization, avoid overfitting

### Dataset

**Healthy Brain Network (HBN) EEG Dataset**
- **Participants:** 3,000+ children and adolescents
- **Channels:** 129 EEG channels
- **Sampling Rate:** 100 Hz
- **Preprocessing:** 0.5-50 Hz bandpass filter, Cz reference
- **Format:** BIDS-compliant, MNE Raw objects
- **Tasks:** 6 cognitive tasks (focus on contrastChangeDetection)
- **Releases:** R1-R5 for training/validation, hidden test set for evaluation

---

## 📊 Current Status

### Challenge 1: Response Time Prediction ✅

**Model:** TCN (Temporal Convolutional Network)
- **Architecture:** 5 TemporalBlocks with BatchNorm
- **Parameters:** 196,225
- **Checkpoint:** `checkpoints/challenge1_tcn_competition_best.pth`
- **Submission Weights:** `weights_challenge_1.pt`
- **Validation Loss:** 0.010170 (epoch 2)
- **Status:** ✅ **READY FOR SUBMISSION**

**Training Details:**
- Task: contrastChangeDetection
- Windows: 2-second stimulus-locked, start +0.5s after stimulus
- Architecture: 129 input channels → 48 filters, kernel 7
- Dilation: [1, 2, 4, 8, 16] for multi-scale temporal features
- Dropout: 0.3 for regularization
- Trained: October 17, 2025

### Challenge 2: Externalizing Factor Prediction 🔄

**Model:** EEGNeX (generalization-focused)
- **Architecture:** Lightweight CNN from braindecode
- **Training:** Started October 19, 2025 at 13:52
- **Status:** 🔄 **TRAINING IN PROGRESS** (Epoch 1/20)
- **Current Progress:** Batch 740/5214 (~14% of epoch 1)
- **Monitoring:** Active watchdog system monitoring for crashes/freezes

**Training Configuration:**
- Task: contrastChangeDetection (same as Challenge 1!)
- Target: p_factor (externalizing factor from CBCL)
- Data: R1-R4 (training), R5 (validation)
- Windows: 4-second with 2-second random crops (data augmentation)
- Loss: L1 (MAE) - robust to outliers
- Optimizer: Adamax (lr=0.002)
- Max Epochs: 20 with early stopping (patience=5)
- Output: `weights_challenge_2_correct.pt`

**Key Design Choices:**
- Small model size to avoid overfitting
- Random cropping for data augmentation
- L1 loss for robustness to outliers
- Focus on cross-subject generalization over training accuracy

---

## 📁 Project Structure

```
eeg2025/
├── README.md                          # This file
├── submission.py                      # Competition submission file
├── test_submission_verbose.py         # Submission validator
│
├── 📄 Active Documentation
│   ├── CHALLENGE2_TRAINING_STATUS.md  # Training configuration & status
│   ├── WATCHDOG_QUICK_REFERENCE.md    # Monitoring system guide
│   └── ORGANIZATION_COMPLETE.md       # Project organization summary
│
├── 🔧 Active Scripts
│   ├── train_challenge2_correct.py    # Challenge 2 training (running)
│   ├── watchdog_challenge2.sh         # Crash/freeze monitor (running)
│   ├── manage_watchdog.sh             # Watchdog control interface
│   ├── monitor_challenge2.sh          # Full training monitor
│   └── quick_training_status.sh       # Quick progress check
│
├── 📁 scripts/                        # Organized scripts
│   ├── README.md                      # Scripts documentation
│   ├── monitoring/                    # Monitoring scripts
│   │   ├── watchdog_challenge2.sh
│   │   ├── manage_watchdog.sh
│   │   ├── monitor_challenge2.sh
│   │   └── quick_training_status.sh
│   └── training/                      # Training scripts
│       └── train_challenge2_correct.py
│
├── 📁 src/                            # Source code
│   ├── models/                        # Model architectures
│   ├── data/                          # Data loading utilities
│   └── utils/                         # Helper functions
│
├── 📁 checkpoints/                    # Model checkpoints
│   ├── challenge1_tcn_competition_best.pth  # Challenge 1 best model
│   └── challenge2_*.pth               # Challenge 2 checkpoints
│
├── 📁 weights_*.pt                    # Submission weights
│   ├── weights_challenge_1.pt         # Challenge 1 (ready)
│   └── weights_challenge_2.pt         # Challenge 2 (will update)
│
├── 📁 logs/                           # Training logs
│   ├── challenge2_correct_training.log
│   ├── watchdog.log
│   └── watchdog_output.log
│
├── 📁 archive/                        # Historical files
│   ├── README.md                      # Archive documentation
│   ├── scripts/                       # Old scripts
│   │   ├── monitoring/
│   │   ├── training/
│   │   └── testing/
│   └── docs/                          # Old documentation
│       ├── status_reports/
│       ├── sessions/
│       ├── overnight/
│       └── submission/
│
├── 📁 .github/instructions/           # Competition knowledge
│   └── memory.instruction.md          # Comprehensive competition reference
│
├── 📁 config/                         # Configuration files
├── 📁 data/                           # Dataset (not in repo)
├── 📁 starter_kit_integration/        # Official starter kit code
└── 📁 notebooks/                      # Jupyter notebooks
```

---

## 🤖 Models

### Challenge 1: TCN (Temporal Convolutional Network)

**Architecture:**
```python
TCN_EEG(
  input_channels=129,
  output_size=1,
  num_channels=[48, 48, 48, 48, 48],  # 5 layers
  kernel_size=7,
  dropout=0.3
)

# 5 TemporalBlocks, each with:
#   - Conv1d (kernel=7, dilation=[1,2,4,8,16])
#   - BatchNorm1d
#   - ReLU
#   - Dropout(0.3)
#   - Residual connection

Total Parameters: 196,225
```

**Key Features:**
- Multi-scale temporal feature extraction via dilated convolutions
- Batch normalization for stable training
- Residual connections for gradient flow
- Dropout for regularization

### Challenge 2: EEGNeX

**Architecture:**
```python
EEGNeX(
  n_chans=129,
  n_outputs=1,
  n_times=200  # 2 seconds at 100 Hz
)

# Lightweight CNN from braindecode library
# Optimized for generalization, not memorization
```

**Key Features:**
- Designed for out-of-distribution robustness
- Small parameter count to prevent overfitting
- Efficient processing of EEG data
- Focus on cross-subject generalization

---

## 🚀 Training

### Challenge 1 (Completed)

**Already trained and ready for submission.**

Training was completed on October 17, 2025. The model achieved excellent validation performance (loss: 0.010170) and is ready for competition submission.

### Challenge 2 (In Progress)

**Current Training Status:**
```bash
# Check quick status
./quick_training_status.sh

# Full monitoring (auto-refresh)
./monitor_challenge2.sh

# Watchdog status
./manage_watchdog.sh status
```

**Training Script:**
```bash
# Already running in background
# Started: October 19, 2025 at 13:52
# PID: 548497
# Log: logs/challenge2_correct_training.log

# To check progress:
tail -f logs/challenge2_correct_training.log
```

**After Training Completes:**
```bash
# 1. Copy trained weights
cp weights_challenge_2_correct.pt weights_challenge_2.pt

# 2. Test submission
python test_submission_verbose.py

# 3. Create submission package
zip -j submission.zip submission.py weights_challenge_1.pt weights_challenge_2.pt

# 4. Submit to competition
# Upload submission.zip to competition platform
```

---

## 📊 Monitoring

### Watchdog System 🐕

An automated monitoring system that watches the training process 24/7 and alerts on issues.

**Features:**
- ❌ **Crash Detection:** Alerts if training process dies
- ❄️ **Freeze Detection:** Alerts if no log updates for 5+ minutes
- ⚠️ **Error Scanning:** Checks logs for errors and exceptions
- 💾 **Memory Monitoring:** Warns if memory usage exceeds 90%
- ✅ **Completion Detection:** Notifies when training finishes

**Quick Commands:**
```bash
# Check status
./manage_watchdog.sh status

# View logs
./manage_watchdog.sh logs

# Follow live output
./manage_watchdog.sh follow

# Stop watchdog
./manage_watchdog.sh stop
```

**Monitoring Scripts:**
```bash
# Quick status snapshot
./quick_training_status.sh

# Full monitor with auto-refresh
./monitor_challenge2.sh

# Live log tail
tail -f logs/challenge2_correct_training.log
```

---

## 💻 Installation

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

### Setup

```bash
# Clone repository
git clone https://github.com/hkevin01/eeg2025.git
cd eeg2025

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### Data Setup

Download the HBN-EEG dataset from the competition website and place in `data/` directory.

The dataset should follow BIDS format with releases R1-R5 available.

---

## 🎮 Usage

### Testing Submission

```bash
# Verbose testing (recommended)
python test_submission_verbose.py

# Basic testing
python submission.py
```

### Creating Submission Package

```bash
# Create submission.zip with all required files
zip -j submission.zip submission.py weights_challenge_1.pt weights_challenge_2.pt

# Verify contents
unzip -l submission.zip

# Expected output:
# submission.py
# weights_challenge_1.pt
# weights_challenge_2.pt
```

### Training New Models

See `CHALLENGE2_TRAINING_STATUS.md` for detailed training configuration and `scripts/README.md` for available training scripts.

---

## 📚 Documentation

### Core Documentation

- **README.md** (this file) - Project overview and quick start
- **CHALLENGE2_TRAINING_STATUS.md** - Training configuration and status
- **WATCHDOG_QUICK_REFERENCE.md** - Monitoring system guide
- **ORGANIZATION_COMPLETE.md** - Project organization summary

### Reference Documentation

- **.github/instructions/memory.instruction.md** - Comprehensive competition knowledge base
  - Challenge requirements and specifications
  - Key differences between challenges
  - Common mistakes to avoid
  - Dataset details and preprocessing
  - Submission format

### Scripts Documentation

- **scripts/README.md** - Active scripts documentation
- **archive/README.md** - Historical files documentation

### Competition Resources

- **Competition Website:** https://eeg2025.github.io/
- **Starter Kit:** `starter_kit_integration/`
- **Challenge 1 Starter:** `starter_kit_integration/challenge_1.py`
- **Challenge 2 Starter:** `starter_kit_integration/challenge_2.py`

---

## 🎯 Key Insights

### Challenge 1 (Response Time Prediction)

**Task Focus:**
- Predict trial-by-trial response time during active cognitive task
- Stimulus-locked windows (+0.5s after stimulus onset, 2s duration)
- ERP components and SSVEP signals important
- Temporal dynamics are key

**Model Strategy:**
- TCN with dilated convolutions for multi-scale temporal features
- Fixed time-locking to stimulus ensures consistent temporal structure
- Larger model acceptable (196K params) - less overfitting risk

### Challenge 2 (Externalizing Factor Prediction)

**Task Focus:**
- Predict subject-level clinical measure from EEG
- Cross-subject generalization is paramount
- Random cropping provides data augmentation
- Must work on unseen subjects/sites

**Model Strategy:**
- Smaller model to prevent overfitting
- L1 loss for robustness to outliers in clinical data
- Random cropping = data augmentation
- Focus on generalization, not training accuracy

### Common Mistakes to Avoid

❌ **Don't use resting task for Challenge 2** - Use contrastChangeDetection!  
❌ **Don't overfit on training data** - Challenge 2 penalizes overfitting heavily  
❌ **Don't ignore random cropping** - It's data augmentation, not just preprocessing  
❌ **Don't use MSE for Challenge 2** - L1 loss is more robust for clinical targets

---

## 📅 Timeline

- **October 17, 2025:** Challenge 1 training completed ✅
- **October 19, 2025:** Challenge 2 training started 🔄
- **October 19, 2025:** Project organization completed ✅
- **October 19, 2025:** Monitoring system activated ✅
- **November 2, 2025:** Competition deadline 🎯

---

## 🤝 Contributing

This is a competition submission repository. Contributions are not currently accepted.

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **NeurIPS 2025 EEG Foundation Challenge** organizers
- **Healthy Brain Network** for the dataset
- **Braindecode** library for baseline models
- **PyTorch** and **MNE-Python** communities

---

## 📧 Contact

For questions about this implementation, please open an issue on GitHub.

For competition-related questions, refer to the [official competition website](https://eeg2025.github.io/).

---

**Last Updated:** October 19, 2025  
**Status:** Challenge 1 Ready | Challenge 2 Training | Repository Organized
