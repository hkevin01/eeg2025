# EEG 2025 Challenge - Project Description

## Project Overview

**Competition:** EEG 2025 Challenge on Codabench  
**URL:** https://www.codabench.org/competitions/4287/  
**Goal:** Predict behavioral and clinical outcomes from EEG data  
**Current Status:** Training Challenge 2 TCN model (Submission v6 in progress)

## Competition Structure

### Challenge 1: Response Time Prediction
- **Task:** Predict reaction time from visual stimulus EEG data
- **Input:** 129-channel EEG, 200 time points (2 seconds at 100Hz)
- **Output:** Single value (response time in seconds)
- **Metric:** Normalized Root Mean Square Error (NRMSE)
- **Baseline NRMSE:** 0.2832
- **Our Best:** Val loss 0.010170 (NRMSE ~0.10, 65% improvement)

### Challenge 2: Externalizing Behavior Prediction
- **Task:** Predict externalizing scores from resting state EEG
- **Input:** 129-channel EEG, 200 time points (2 seconds at 100Hz)
- **Output:** Single value (externalizing score)
- **Metric:** Normalized Root Mean Square Error (NRMSE)
- **Baseline NRMSE:** 0.2917
- **Current Status:** Training TCN model (in progress)

## Dataset

### Source
- **Name:** Healthy Brain Network (HBN) Dataset
- **Preprocessing:** Pre-applied by competition organizers
  - Downsampled: 500Hz → 100Hz
  - Bandpass filtered: 0.5-50Hz
  - Re-referenced: Cz channel
- **Access:** Via `EEGChallengeDataset` from eegdash package

### Data Splits
- **R1:** Release 1 (train/validation)
- **R2:** Release 2 (train/validation)
- **R3:** Release 3 (train/validation)
- **R4:** Release 4 (train/validation)
- **R5:** Release 5 (test - competition evaluation)

### Current Training Setup
- **Challenge 1:** R1-R3 train (11,502 samples), R4 validate (3,189 samples)
- **Challenge 2:** R1-R3 train (99,063 samples), R4 validate (63,163 samples)

## Technical Stack

### Core Framework
- **Python:** 3.12
- **PyTorch:** 2.x (CUDA/ROCm support)
- **MNE-Python:** EEG data handling
- **braindecode:** EEG-specific preprocessing
- **eegdash:** Competition data access

### Key Libraries
- **NumPy/SciPy:** Numerical computing
- **scikit-learn:** Machine learning utilities
- **pandas:** Data manipulation
- **matplotlib/seaborn:** Visualization

### Development Tools
- **tmux:** Independent training sessions (survives crashes)
- **VS Code:** Primary IDE
- **Git:** Version control
- **Codabench:** Competition submission platform

## Model Architecture

### Current Approach: Temporal Convolutional Network (TCN)

**Architecture Details:**
```python
TCN_EEG(
    num_channels=129,        # EEG channels
    num_outputs=1,           # Single regression output
    num_filters=48,          # Filters per temporal block
    kernel_size=7,           # Kernel size for convolutions
    dropout=0.3,             # Dropout rate
    num_levels=5             # Number of temporal blocks
)
```

**Key Features:**
- Dilated causal convolutions (dilation: 1, 2, 4, 8, 16)
- Batch normalization in each temporal block
- Residual connections
- ReLU activations
- Dropout for regularization
- Parameters: 196,225 (77% smaller than previous models)

**Why TCN?**
- Captures long-range temporal dependencies (receptive field: 127 time steps)
- Causal structure respects EEG time series nature
- Efficient: 5x fewer parameters than attention-based models
- Proven: 65% improvement on Challenge 1

## Project Goals

### Primary Goal
🏆 **Achieve Top 3 ranking on competition leaderboard**

### Secondary Goals
1. Develop robust, generalizable EEG models
2. Beat baseline performance on both challenges
3. Create reproducible training pipeline
4. Document methodology for publication/sharing

### Success Metrics
- **Challenge 1:** NRMSE < 0.15 (50% better than baseline)
- **Challenge 2:** NRMSE < 0.25 (15% better than baseline)
- **Overall:** Combined score in top 3 teams
- **Reproducibility:** Training runs independently, survives crashes

## Current Architecture

### Directory Structure
```
eeg2025/
├── memory-bank/              # Project memory (you are here)
├── src/                      # Source code
│   ├── models/              # Model architectures
│   ├── dataio/              # Data loading
│   ├── training/            # Training loops
│   └── gpu/                 # GPU optimizations
├── scripts/                  # Training scripts
│   ├── train_challenge1_tcn.py
│   ├── train_challenge2_tcn.py
│   └── monitoring/          # Training monitors
├── checkpoints/             # Model weights
│   ├── challenge1_tcn_competition_best.pth (2.4MB)
│   └── challenge2_tcn_competition_best.pth (2.4MB, training)
├── logs/                    # Training logs
├── data/                    # Dataset cache
├── submission.py            # Competition submission file
└── improvements/            # Model development archive
```

## Target Users

### Primary User: Competition Participants
- Need to quickly resume training after interruptions
- Require clear documentation of model choices
- Want reproducible results

### Secondary User: Future Developers
- Inheriting this codebase
- Need to understand architecture decisions
- Want to extend or modify models

### Tertiary User: Research Community
- Interested in EEG analysis methods
- Want to learn from our approach
- May cite or build upon this work

## Key Innovations

1. **Independent Training System**
   - Uses tmux for crash-resistant training
   - Survives VS Code disconnects, SSH drops, terminal closes
   - Training logs captured continuously

2. **TCN Architecture for EEG**
   - First successful application in this competition
   - 65% improvement over baseline (Challenge 1)
   - 77% parameter reduction vs attention models

3. **Efficient Data Loading**
   - Fixed window indexing bugs
   - Handles 99K+ samples efficiently
   - Proper dtype handling (Float32)

4. **Comprehensive Monitoring**
   - Real-time training progress tracking
   - Automatic best model saving
   - Early stopping with patience

## Current Phase: Submission v6

**Status:** 🔄 In Progress

**Components:**
- ✅ Challenge 1 TCN trained and integrated
- 🔄 Challenge 2 TCN training (epoch 4/100)
- ⏳ Final integration and testing
- ⏳ Package and upload to Codabench

**Expected Completion:** Tonight or tomorrow morning

**Next Steps:** See `memory-bank/implementation-plans/submission-v6.md`

