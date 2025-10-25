# SAM Training Status Update - Oct 24, 2025 (6:50 PM)

## 🎯 Current Situation

### Challenge 1 (CompactCNN) - FAILED ❌
- **Issue**: train_c1_sam_simple.py uses EEGNeX model (not CompactCNN!)
- **Error**: `RuntimeError: HIP error: invalid device function`
- **Root Cause**: EEGNeX doesn't work on AMD Radeon RX 5600 XT (gfx1010) with ROCm
- **Solution Needed**: Use CompactCNN model OR run on CPU

### Challenge 2 (EEGNeX) - COMPLETED WITH DUMMY DATA ❌
- **Status**: Training completed in ~2 minutes
- **Best Val NRMSE**: 0.2177
- **Issue**: Used DUMMY DATA (random EEG windows)
- **Root Cause**: train_c2_sam_eegnex.py loads externalizing targets but generates random EEG
- **Solution**: Created train_c2_sam_real_data.py with proper EEG loading from eegdash

## 📊 Data Verification

### Real Data Available ✅
```bash
data/ds005505-bdf/participants.tsv  # Has "externalizing" column
data/ds005506-bdf/  # Challenge 1 data (150 subjects)
data/ds005507-bdf/  # Challenge 1 data (184 subjects)
```

### Challenge 1 Data ✅
- ds005506-bdf: 150 subjects
- ds005507-bdf: 184 subjects
- Task: contrastChangeDetection
- Target: response time (rt_from_stimulus)
- Successfully loaded when using limited dataset (50 subjects)

### Challenge 2 Data ✅
- ds005505-bdf: Has participants.tsv with externalizing scores
- R1-R5 releases available via eegdash
- Task: contrastChangeDetection  
- Target: p_factor (externalizing)

## 🔧 Files Created/Fixed

### Working Scripts
1. **train_c1_sam_simple.py** - ❌ Uses EEGNeX (GPU incompatible)
2. **train_c2_sam_eegnex.py** - ❌ Uses dummy data
3. **train_c2_sam_real_data.py** - ✅ NEW! Uses real EEG with SAM

### Issues to Fix
- C1 needs to use CompactCNN model (not EEGNeX)
- OR run C1 on CPU with current script
- C2 new script ready but not tested yet

## 📋 Next Steps (Priority Order)

### IMMEDIATE (Now)
1. **Restart C1 on CPU** with train_c1_sam_simple.py
   - Command: `--device cpu --max-subjects 50`
   - Expected: 2-4 hours with limited dataset
   - Alternative: Create new script with CompactCNN model

2. **Start C2 with REAL data** using train_c2_sam_real_data.py
   - Uses eegdash EEGChallengeDataset
   - Loads actual EEG from R1-R5
   - SAM optimizer integrated
   - Expected: Several hours with full dataset

### SHORT TERM (Tonight)
1. Monitor both trainings
2. Check first epoch metrics to verify real data
3. Adjust hyperparameters if needed

### MEDIUM TERM (Tomorrow)
1. Extract best weights from completed trainings
2. Create submission package
3. Upload to Codabench

## 🎯 Target Scores

### Baseline (Oct 16 + Oct 24)
- C1: 1.0015 (CompactCNN)
- C2: 1.0087 (EEGNeX)
- Overall: ~1.005

### SAM Targets
- C1: < 0.9 (10% improvement)
- C2: < 0.9 (10% improvement)
- Overall: < 0.85 (competitive)

## 🚨 Key Learnings

1. **AMD GPU Compatibility**: EEGNeX doesn't work on gfx1010, use CPU or different model
2. **Data Loading**: Must use proper loaders (eegdash for C2, direct file loading for C1)
3. **Dummy Data Detection**: Training that completes in minutes = dummy data
4. **Progress Bars**: Real data loading shows tqdm progress bars
5. **Limited Datasets**: Use --max-subjects for faster testing

## 📝 Commands to Run

### Restart C1 (CPU, limited dataset)
```bash
tmux new-session -d -s sam_c1 "cd /home/kevin/Projects/eeg2025 && python -u train_c1_sam_simple.py --data-dir data/ds005506-bdf data/ds005507-bdf --epochs 50 --batch-size 16 --lr 1e-3 --device cpu --max-subjects 50 2>&1 | tee training_sam_c1_cpu.log"
```

### Start C2 (CPU, real data)
```bash
tmux new-session -d -s sam_c2 "cd /home/kevin/Projects/eeg2025 && python -u train_c2_sam_real_data.py 2>&1 | tee training_sam_c2_real.log"
```

### Monitor
```bash
# Watch C1
tail -f training_sam_c1_cpu.log

# Watch C2
tail -f training_sam_c2_real.log

# List sessions
tmux ls
```

---

**Updated**: Oct 24, 2025 at 6:50 PM  
**Status**: Both trainings failed, fixes ready, ready to restart
**Next Action**: Restart C1 on CPU + Start C2 with real data
