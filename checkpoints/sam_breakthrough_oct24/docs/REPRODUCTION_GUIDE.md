# Reproduction Guide - SAM Breakthrough

This guide provides step-by-step instructions to reproduce the SAM breakthrough results.

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 22.04+ recommended)
- **CPU**: Multi-core x86-64 processor
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB free space
- **GPU**: AMD RX 5600 XT or better (for C2)

### Software Requirements
- Python 3.11
- Git
- ROCm SDK (for GPU training)

## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/hkevin01/eeg2025.git
cd eeg2025
```

### 2. Install Dependencies
```bash
# Standard dependencies (for C1)
pip install -r requirements.txt

# For C2 GPU training, use ROCm SDK
source /opt/rocm_sdk_612/bin/activate_sdk.sh
```

### 3. Download Competition Data
```bash
# Download from Codabench
# Place in data/ directory:
# - data/challenge1_train.h5
# - data/challenge2_train.h5
```

## Reproducing C1 Results (Val NRMSE: 0.3008)

### Step 1: Verify Data
```bash
python -c "
import h5py
with h5py.File('data/challenge1_train.h5', 'r') as f:
    print('C1 subjects:', len(f['data'].keys()))
    print('C1 channels:', f['data']['001']['eeg'].shape[0])
"
# Expected: 72 subjects, 72 channels
```

### Step 2: Run C1 Training
```bash
# Copy training script from checkpoint
cp checkpoints/sam_breakthrough_oct24/configs/train_c1_sam_simple.py ./

# Run training (takes ~4 hours on CPU)
python train_c1_sam_simple.py
```

### Step 3: Monitor Training
```bash
# In another terminal
tail -f training_sam_c1_cpu.log

# Look for:
# - "Best val NRMSE: 0.3008" (should appear around epoch 21)
# - Training completes after 30 epochs
```

### Step 4: Verify Results
```bash
# Check final validation NRMSE
grep "Best val NRMSE" training_sam_c1_cpu.log

# Expected output:
# Best val NRMSE: 0.3008 (Epoch 21)
```

### Step 5: Compare Weights
```bash
# Your trained weights
ls -lh weights_challenge_1_correct.pt

# Checkpoint reference
ls -lh checkpoints/sam_breakthrough_oct24/c1/sam_c1_best_model.pt

# Should both be ~259K
```

## Reproducing C2 Results (Target: Val NRMSE < 0.9)

### Step 1: Setup ROCm SDK
```bash
# Verify ROCm SDK is installed
ls -la /opt/rocm_sdk_612/

# Activate SDK environment
source /opt/rocm_sdk_612/bin/activate_sdk.sh

# Verify PyTorch detects GPU
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
"
# Expected: CUDA available: True, GPU: AMD Radeon RX 5600 XT
```

### Step 2: Verify Data
```bash
python -c "
import h5py
with h5py.File('data/challenge2_train.h5', 'r') as f:
    print('C2 subjects:', len(f['data'].keys()))
    print('C2 channels:', f['data']['001']['eeg'].shape[0])
"
# Expected: 334 subjects, 104 channels
```

### Step 3: Run C2 Training
```bash
# Copy training script from checkpoint
cp checkpoints/sam_breakthrough_oct24/configs/train_c2_sam_real_data.py ./

# Run training with SDK (takes ~4 hours on GPU)
source /opt/rocm_sdk_612/bin/activate_sdk.sh
python train_c2_sam_real_data.py
```

### Step 4: Monitor Training
```bash
# In another terminal
tail -f training_sam_c2_sdk.log

# Look for:
# - "Device: cuda" (confirms GPU usage)
# - Decreasing validation NRMSE each epoch
# - Target: < 0.9 by final epoch

# Monitor GPU usage
watch -n 1 rocm-smi
```

### Step 5: Verify Results
```bash
# Check final validation NRMSE
grep "Best val NRMSE" training_sam_c2_sdk.log

# Expected: < 0.9 NRMSE
```

## Creating Submission

### Step 1: Verify Weights Exist
```bash
ls -lh weights_challenge_1_correct.pt  # C1: ~259K
ls -lh weights_challenge_2_correct.pt  # C2: ~124K
```

### Step 2: Test Submission Locally
```bash
python test_submission_verbose.py

# Should show:
# - Both models load successfully
# - Predictions generated for all test samples
# - No errors
```

### Step 3: Package Submission
```bash
zip submission_sam.zip \
    submission.py \
    weights_challenge_1_correct.pt \
    weights_challenge_2_correct.pt

# Verify package
unzip -l submission_sam.zip
```

### Step 4: Upload to Codabench
1. Go to competition page
2. Click "Submit"
3. Upload `submission_sam.zip`
4. Wait for evaluation (~15 minutes)

## Troubleshooting

### C1 Training Issues

**Issue**: "Out of memory" during training
```bash
# Solution: Reduce batch size in train_c1_sam_simple.py
# Change line 156: batch_size = 16  # was 32
```

**Issue**: Training very slow
```bash
# Solution: Use fewer CV folds
# Change line 154: n_folds = 3  # was 5
```

**Issue**: High validation NRMSE (> 0.5)
```bash
# Solution: Check data augmentation is enabled
# Verify lines 95-100 in train_c1_sam_simple.py
```

### C2 Training Issues

**Issue**: "HIP error: invalid device function"
```bash
# Solution: Use ROCm SDK
source /opt/rocm_sdk_612/bin/activate_sdk.sh
unset HSA_OVERRIDE_GFX_VERSION
python train_c2_sam_real_data.py
```

**Issue**: GPU out of memory
```bash
# Solution: Reduce batch size
# Change line 275: batch_size = 8  # was 16
```

**Issue**: Training not using GPU
```bash
# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Check device in script (line 273)
# Should be: device = "cuda" if torch.cuda.is_available() else "cpu"
```

### Submission Issues

**Issue**: Submission file too large
```bash
# Check file sizes
ls -lh weights_*.pt

# Should be:
# - C1: ~259K
# - C2: ~124K
# - Total zip: < 1MB
```

**Issue**: Submission fails validation
```bash
# Test locally first
python test_submission_verbose.py

# Check for errors in submission.py
# Verify model architectures match training
```

## Expected Timeline

### C1 Training
- Setup: 5 minutes
- Training: 4 hours (CPU)
- Validation: 30 minutes
- **Total**: ~5 hours

### C2 Training
- Setup: 10 minutes (ROCm SDK)
- Training: 4 hours (GPU)
- Validation: 45 minutes
- **Total**: ~5 hours

### Combined
- **End-to-end**: ~10 hours (if run sequentially)
- **Parallel**: ~5 hours (if run simultaneously on different machines)

## Validation Checklist

- [ ] Repository cloned successfully
- [ ] Dependencies installed
- [ ] Data downloaded and verified
- [ ] C1 training script runs without errors
- [ ] C1 validation NRMSE ≈ 0.3008 (±0.05)
- [ ] C1 weights saved (~259K)
- [ ] ROCm SDK activated for C2
- [ ] C2 training script runs on GPU
- [ ] C2 validation NRMSE < 0.9
- [ ] C2 weights saved (~124K)
- [ ] Submission package created
- [ ] Local submission test passes
- [ ] Uploaded to Codabench

## Performance Expectations

### C1 (EEGNeX + SAM)
- **Target**: Val NRMSE ≈ 0.30
- **Range**: 0.28 - 0.35 (acceptable)
- **Baseline**: 1.0015 (CompactCNN)
- **Improvement**: 70% better

### C2 (EEGNeX + SAM)
- **Target**: Val NRMSE < 0.9
- **Range**: 0.8 - 1.0 (acceptable)
- **Baseline**: 1.0087 (EEGNeX)
- **Improvement**: 10-20% expected

### Combined Submission
- **Conservative**: Overall ≈ 0.675
- **Optimistic**: Overall ≈ 0.585
- **Best Case**: Overall ≈ 0.540

## Additional Resources

### Training Logs
- C1: `checkpoints/sam_breakthrough_oct24/logs/training_sam_c1_cpu.log`
- C2: `checkpoints/sam_breakthrough_oct24/logs/training_sam_c2_sdk.log`

### Reference Weights
- C1: `checkpoints/sam_breakthrough_oct24/c1/sam_c1_best_model.pt`
- C2: `checkpoints/sam_breakthrough_oct24/c2/sam_c2_best_weights.pt`

### Documentation
- Checkpoint Info: `checkpoints/sam_breakthrough_oct24/docs/CHECKPOINT_INFO.md`
- Model Details: `checkpoints/sam_breakthrough_oct24/docs/MODEL_ARCHITECTURES.md`

## Contact & Support

**Repository**: https://github.com/hkevin01/eeg2025  
**Competition**: Decoding Brain Signals 2025  
**Last Updated**: October 24, 2025

---

If you encounter issues not covered here, please check the training logs and verify your environment matches the requirements exactly.
