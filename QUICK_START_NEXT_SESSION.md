# Quick Start - Next Session

## Status
✅ Memory-efficient training solution implemented  
✅ HDF5 preprocessing script ready  
✅ Challenge 2 memory safety added  
⏳ Waiting to test preprocessing and training  

## Immediate Actions (In Order)

### 1. Run Preprocessing (30-60 min)
```bash
cd /home/kevin/Projects/eeg2025
python scripts/preprocessing/cache_challenge1_windows.py
```

**Expected output:**
- `data/processed/R1_challenge1_windows.h5` (~2GB)
- `data/processed/R2_challenge1_windows.h5` (~3GB)
- `data/processed/R3_challenge1_windows.h5` (~3GB)
- `data/processed/R4_challenge1_windows.h5` (~4GB)
- Total: ~12GB on disk, minimal RAM usage

### 2. Test HDF5Dataset (1 min)
```bash
python src/utils/hdf5_dataset.py
```

**Should show:**
- Files loaded successfully
- Sample shapes correct
- DataLoader works with multiple workers

### 3. Create Memory-Efficient Training Script
Copy the Challenge 1 training script but use HDF5Dataset:

```python
from src.utils.hdf5_dataset import HDF5Dataset

# Instead of loading windows directly:
dataset = HDF5Dataset([
    'data/processed/R1_challenge1_windows.h5',
    'data/processed/R2_challenge1_windows.h5',
    'data/processed/R3_challenge1_windows.h5',
    'data/processed/R4_challenge1_windows.h5',
])

# Split train/val
train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

# Use with DataLoader (workers load in parallel)
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, num_workers=2)
```

### 4. Train Overnight
```bash
./train_challenge1_hdf5_tmux.sh  # Create this script
```

## Key Files

### Documentation
- `docs/strategy/COMPETITION_ACTION_PLAN.md` - Full roadmap
- `docs/strategy/MEMORY_EFFICIENT_TRAINING.md` - Technical details
- `docs/SESSION_SUMMARY_OCT18_MEMORY_SOLUTION.md` - Today's work

### Implementation  
- `scripts/preprocessing/cache_challenge1_windows.py` - Preprocess to HDF5
- `src/utils/hdf5_dataset.py` - Memory-efficient dataset
- `scripts/training/challenge2/train_challenge2_multi_release.py` - Memory-safe Challenge 2

## Competition Targets

**Current:**
- Challenge 1: 1.00 NRMSE (validation, trained with stimulus alignment)
- Challenge 2: 1.46 NRMSE (baseline)
- Overall: 1.23 NRMSE

**Next milestone (after HDF5 training):**
- Challenge 1: 0.85-0.95 NRMSE (train on all R1-R4 data)
- Challenge 2: 1.30-1.40 NRMSE (with regularization)
- Overall: 1.15-1.20 NRMSE

**Final goal:**
- Challenge 1: 0.70-0.75 NRMSE (with EEGNet + TTA)
- Challenge 2: 0.95-1.00 NRMSE (with focused work)
- Overall: 0.87-0.91 NRMSE (top 5 material!)

## Next Improvements (After HDF5 Works)

1. **EEGNet architecture** - Proven for EEG, wins competitions
2. **Data augmentation** - Gaussian noise, time shift, channel dropout
3. **TTA** (Test-Time Augmentation) - Free 5-10% improvement
4. **Ensemble** - Average multiple models
5. **Huber loss** - More robust to outliers
6. **Challenge 2 focus** - 47% behind, needs special attention

## Memory Safety Checklist

Before training:
- [ ] Check available RAM: `free -h`
- [ ] Close unnecessary programs
- [ ] Monitor during training: `watch -n 10 'free -h'`
- [ ] If crash: Reduce MAX_DATASETS_PER_RELEASE in training script

## Debugging Commands

```bash
# Check if preprocessing completed
ls -lh data/processed/*.h5

# Test loading a cached file
python -c "import h5py; f = h5py.File('data/processed/R1_challenge1_windows.h5', 'r'); print(f'Shape: {f[\"eeg\"].shape}'); f.close()"

# Monitor memory during training
watch -n 5 'ps aux | grep python | grep train_challenge'

# Check training log
tail -f logs/training_comparison/challenge1_hdf5_*.log
```

## Success Criteria

✅ Preprocessing completes without crash  
✅ HDF5Dataset test passes  
✅ Training runs overnight without crash  
✅ Memory stays below 80%  
✅ NRMSE improves or stays stable  

---

**Quick Start:** Run preprocessing, test loading, train overnight!
