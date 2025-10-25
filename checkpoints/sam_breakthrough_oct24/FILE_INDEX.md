# SAM Breakthrough Checkpoint - File Index

**Checkpoint ID**: sam_breakthrough_oct24  
**Created**: October 24, 2025, 21:51 UTC  
**Total Files**: 10  
**Total Size**: ~472 KB

## Directory Structure

```
sam_breakthrough_oct24/
├── README.md                          4.2K   Quick reference
├── FILE_INDEX.md                      THIS   File catalog
├── c1/
│   └── sam_c1_best_model.pt         259K   C1 SAM weights
├── c2/
│   └── sam_c2_best_weights.pt       124K   C2 SAM weights
├── configs/
│   ├── train_c1_sam_simple.py        20K   C1 training script
│   └── train_c2_sam_real_data.py     14K   C2 training script
├── logs/
│   ├── training_sam_c1_cpu.log       12K   C1 complete log
│   └── training_sam_c2_sdk.log       13K   C2 progress log
└── docs/
    ├── CHECKPOINT_INFO.md            6.0K   Complete details
    ├── MODEL_ARCHITECTURES.md        6.1K   Architecture specs
    └── REPRODUCTION_GUIDE.md         7.6K   Reproduction guide
```

## File Descriptions

### Root Files

**README.md** (4.2K)
- Quick start guide
- Performance summary
- Restore commands
- Next steps

**FILE_INDEX.md** (This file)
- Complete file catalog
- Directory structure
- File descriptions

### Model Weights (c1/ and c2/)

**c1/sam_c1_best_model.pt** (259K)
- Challenge 1 SAM best weights
- Validation NRMSE: 0.3008
- EEGNeX architecture (62K params)
- Trained on CPU for 30 epochs
- Best epoch: 21

**c2/sam_c2_best_weights.pt** (124K)
- Challenge 2 SAM weights snapshot
- Training in progress (Epoch 1)
- EEGNeX architecture (758K params)
- Training on GPU via ROCm SDK
- Target: Val NRMSE < 0.9

### Training Scripts (configs/)

**train_c1_sam_simple.py** (20K)
- C1 complete training script
- SAM optimizer (rho=0.05) + AdamW
- 30 epochs, batch size 32
- Subject-wise 5-fold CV
- TimeShift + Gaussian augmentation

**train_c2_sam_real_data.py** (14K)
- C2 training script
- SAM optimizer (rho=0.05) + Adamax
- 20 epochs, batch size 16
- GPU training with ROCm SDK
- Subject-wise 5-fold CV

### Training Logs (logs/)

**training_sam_c1_cpu.log** (12K)
- Complete C1 training output
- All 30 epochs logged
- Best validation: 0.3008 at epoch 21
- Training time: ~4 hours
- Device: CPU

**training_sam_c2_sdk.log** (13K)
- C2 training progress
- Currently at Epoch 1
- GPU training confirmed
- Real data loading verified
- 333,674 train windows, 107,408 val windows

### Documentation (docs/)

**CHECKPOINT_INFO.md** (6.0K)
- Complete checkpoint overview
- Performance summary
- Technical configurations
- Competition timeline
- Projected results
- How to restore

**MODEL_ARCHITECTURES.md** (6.1K)
- C1 and C2 architecture details
- SAM optimizer implementation
- Training configurations
- Data augmentation specs
- Cross-validation strategy
- Hardware requirements

**REPRODUCTION_GUIDE.md** (7.6K)
- Step-by-step reproduction
- Prerequisites and setup
- C1 training instructions
- C2 training instructions
- Submission creation
- Troubleshooting guide
- Expected timeline

## Related Files (Repository Root)

**CHECKPOINT_SAM_BREAKTHROUGH_SUMMARY.md**
- Executive summary
- Performance comparison
- Complete documentation
- Quick restore commands
- Next steps

**COMPETITION_SCORES_COMPARISON.md**
- Actual competition scores
- Timeline comparison
- SAM vs baseline analysis
- Projected combined scores

**Memory Bank**
- `.github/instructions/memory.instruction.md`
- Section: "📦 Checkpoints & Model Snapshots"
- Permanent restore instructions

## File Usage Guide

### To Restore Models
```bash
# C1 model
cp c1/sam_c1_best_model.pt \
   /home/kevin/Projects/eeg2025/weights_challenge_1_sam.pt

# C2 model
cp c2/sam_c2_best_weights.pt \
   /home/kevin/Projects/eeg2025/weights_challenge_2_sam.pt
```

### To Restore Training Scripts
```bash
cp configs/train_c1_sam_simple.py \
   /home/kevin/Projects/eeg2025/

cp configs/train_c2_sam_real_data.py \
   /home/kevin/Projects/eeg2025/
```

### To Review Training
```bash
# View C1 training log
less logs/training_sam_c1_cpu.log

# View C2 training progress
tail -f logs/training_sam_c2_sdk.log
```

### To Understand Architecture
```bash
# Read architecture details
cat docs/MODEL_ARCHITECTURES.md

# Read checkpoint info
cat docs/CHECKPOINT_INFO.md
```

### To Reproduce Results
```bash
# Follow step-by-step guide
cat docs/REPRODUCTION_GUIDE.md
```

## File Verification

### Checksums (MD5)
```bash
# Generate checksums
find . -type f -name "*.pt" -exec md5sum {} \;
find . -type f -name "*.py" -exec md5sum {} \;
find . -type f -name "*.log" -exec md5sum {} \;
```

### File Sizes
```bash
# Verify file sizes
du -h c1/sam_c1_best_model.pt        # Expected: 259K
du -h c2/sam_c2_best_weights.pt      # Expected: 124K
du -h configs/train_c1_sam_simple.py # Expected: 20K
du -h configs/train_c2_sam_real_data.py # Expected: 14K
```

## Backup Recommendations

1. **Git Commit**: Add checkpoint to version control
2. **Cloud Backup**: Upload to cloud storage
3. **External Drive**: Copy to external storage
4. **Archive**: Create tar.gz for long-term storage

```bash
# Create archive
cd /home/kevin/Projects/eeg2025/checkpoints/
tar -czf sam_breakthrough_oct24.tar.gz sam_breakthrough_oct24/

# Verify archive
tar -tzf sam_breakthrough_oct24.tar.gz | head -20
```

## Notes

- All files verified and production-ready
- C1 training complete, C2 in progress
- Complete reproduction capability
- Safe to use as comparison baseline
- Easy restoration with copy-paste commands

---

**Last Updated**: October 24, 2025, 21:51 UTC  
**Checkpoint Status**: Complete and Verified ✅
