# Weights Metadata

**Last Updated**: October 19, 2025, 14:54

## Current Versions

### Challenge 1: Response Time Prediction ✅
- **Current**: `challenge1/weights_challenge_1_current.pt`
- **Source**: `checkpoints/challenge1_tcn_competition_best.pth`
- **Status**: ✅ Ready for submission
- **Validation Loss**: 0.010170 (NRMSE)
- **Training Date**: October 17, 2025
- **Model**: TCN (196,225 params)
- **Architecture**: 5 TemporalBlocks with dilated convolutions [1,2,4,8,16]
- **Receptive Field**: 373 timepoints (3.73s at 100 Hz)

### Challenge 2: Externalizing Factor Prediction 🔄
- **Current**: `challenge2/weights_challenge_2_current.pt`
- **Status**: 🔄 Training in progress (Epoch 1/20, ~25% complete)
- **Training Started**: October 19, 2025, 13:52
- **Training PID**: 548497 (still running)
- **Watchdog PID**: 560789 (monitoring)
- **Model**: EEGNeX (lightweight CNN from braindecode)
- **Loss Function**: L1 (MAE)
- **Optimizer**: Adamax (lr=0.002)
- **Early Stopping**: Patience=5

### Multi-Release Versions
Located in `multi_release/`:
- `weights_challenge_1_multi_release.pt` - Trained on R1-R5 combined
- `weights_challenge_2_multi_release.pt` - Trained on R1-R5 combined

## Directory Structure

```
weights/
├── challenge1/                      # Challenge 1 weights (response time)
│   ├── weights_challenge_1_current.pt  ← Use for submission
│   └── weights_challenge_1_YYYYMMDD_HHMMSS.pt  (timestamped backups)
├── challenge2/                      # Challenge 2 weights (p_factor)
│   ├── weights_challenge_2_current.pt  ← Use for submission  
│   ├── weights_challenge_2.pt         (original)
│   └── weights_challenge_2_YYYYMMDD_HHMMSS.pt  (timestamped backups)
├── multi_release/                   # Multi-release trained versions
│   ├── weights_challenge_1_multi_release.pt
│   └── weights_challenge_2_multi_release.pt
├── checkpoints/                     # Training checkpoints (.pth files)
│   └── challenge1_tcn_competition_best.pth (2.4MB)
└── archive/                        # Old/deprecated weights
```

## Version History

### Challenge 1
| Version | Date | Status | Val Loss | File Size | Notes |
|---------|------|--------|----------|-----------|-------|
| v1.0 | Oct 17, 2025 | ✅ Ready | 0.010170 | 2.4MB | TCN model, competition ready |

### Challenge 2  
| Version | Date | Status | Val Loss | File Size | Notes |
|---------|------|--------|----------|-----------|-------|
| v0.1 | Oct 19, 2025 | 🔄 Training | TBD | 261KB | EEGNeX, epoch 1/20 in progress |

## Submission Instructions

For competition submission, always use the `*_current.pt` files:

```bash
# 1. Verify models work
python test_submission_verbose.py

# 2. Create submission package
zip -j submission.zip \
    submission.py \
    weights/challenge1/weights_challenge_1_current.pt \
    weights/challenge2/weights_challenge_2_current.pt

# 3. Verify package
unzip -l submission.zip
# Should show exactly 3 files

# 4. Upload to competition platform
```

## Backup Strategy

- **Timestamped Backups**: Created automatically on each organization
- **Original Files**: Preserved with dates
- **Checkpoints**: Maintained separately for reference
- **Easy Rollback**: Can restore any previous version

## Important Notes

⚠️ **While Training Active**:
- Do NOT modify or move weights files
- Do NOT interrupt training process (PID 548497)
- Watchdog monitoring active (PID 560789)
- Wait for training to complete before submission

✅ **Safe Operations**:
- Read/copy weight files
- Create new backups
- Update documentation

🎯 **For Final Submission**:
- Challenge 1: Use `challenge1/weights_challenge_1_current.pt` (already ready)
- Challenge 2: Wait for training to complete, then use `challenge2/weights_challenge_2_current.pt`

## File Sizes Reference

| File | Size | Type |
|------|------|------|
| challenge1_tcn_competition_best.pth | 2.4 MB | Checkpoint |
| weights_challenge_1_current.pt | 2.4 MB | Submission weights |
| weights_challenge_2.pt | 261 KB | Current training |
| weights_challenge_*_multi_release.pt | ~300 KB | Multi-release versions |

## Model Architectures

### TCN (Challenge 1)
- Input: 129 channels × 200 timepoints (2s)
- Layers: 5 TemporalBlocks
- Filters: 48 per layer
- Kernel size: 7
- Dilation: [1, 2, 4, 8, 16]
- Dropout: 0.3
- Total params: 196,225

### EEGNeX (Challenge 2)
- Input: 129 channels × 200 timepoints (2s)
- Architecture: Depthwise separable CNN
- Design: Lightweight, generalization-focused
- Total params: ~50,000 (estimated)
- From: braindecode library

---

*Auto-generated metadata - Last organization: 20251019_145544*
