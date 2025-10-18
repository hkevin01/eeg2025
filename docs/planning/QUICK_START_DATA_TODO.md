# 🚀 Quick Start: Data Acquisition TODO

**Created**: October 14, 2025  
**Status**: Ready to Execute  
**Time Estimate**: 2-4 hours

---

## Phase 1: Environment Setup (15 minutes)

### ✅ Step 1: Install Required Packages
```bash
pip install awscli boto3 requests tqdm
pip install mne mne-bids pandas numpy
```

**Verification**:
```bash
python -c "import mne; print(f'MNE version: {mne.__version__}')"
python -c "import boto3; print('AWS SDK ready')"
```

---

## Phase 2: Data Download (2-3 hours)

### ✅ Step 2: Create Data Directory
```bash
mkdir -p data/raw/hbn
mkdir -p data/processed
mkdir -p data/cache
```

### ✅ Step 3: Download Sample Data First
```bash
# Start with 1-2 subjects to test pipeline
python scripts/download_hbn_data.py \
    --subjects 2 \
    --output-dir data/raw/hbn \
    --verify
```

**Expected Output**:
```
✓ Downloaded 2 subjects
✓ Total size: ~2-3 GB
✓ All files verified
```

### ✅ Step 4: Verify Data Structure
```bash
python scripts/verify_data_structure.py \
    --data-dir data/raw/hbn \
    --verbose
```

---

## Phase 3: Quick Validation (30 minutes)

### ✅ Step 5: Test Data Loading
```bash
python tests/test_data_loading.py -v
```

**Expected Result**: All data loading tests pass ✓

### ✅ Step 6: Run Basic Preprocessing
```bash
python scripts/test_preprocessing_pipeline.py \
    --subject NDARAA536PTU \
    --visualize
```

**Check**: Should generate plots in `data/validation/`

### ✅ Step 7: Validate Against Known Values
```bash
python scripts/validate_data_statistics.py \
    --data-dir data/raw/hbn \
    --compare-baseline
```

---

## Phase 4: Integration Test (30 minutes)

### ✅ Step 8: End-to-End Pipeline Test
```bash
# Test full pipeline with 1 subject
python scripts/test_full_pipeline.py \
    --subject NDARAA536PTU \
    --challenge challenge1
```

### ✅ Step 9: Verify Model Input Shape
```bash
python -c "
from src.dataio.hbn_dataset import HBNDataset
dataset = HBNDataset('data/raw/hbn', task_type='cross_task')
sample = dataset[0]
print(f'EEG shape: {sample[0].shape}')
print(f'Labels: {sample[1]}')
print('✓ Data pipeline working!')
"
```

---

## 🎯 Success Criteria

After completing all steps, you should have:

- [x] **Environment Ready**: All packages installed
- [x] **Sample Data**: 2+ subjects downloaded and verified
- [x] **Data Loading**: Can load EEG data successfully
- [x] **Preprocessing**: Basic filtering/epoching works
- [x] **Pipeline Test**: End-to-end validation passes

---

## ⚠️ Common Issues & Solutions

### Issue 1: Download Fails
**Symptom**: Connection timeout or authentication error
**Solution**: 
```bash
# Check internet connection
ping fcon_1000.projects.nitrc.org

# Try with retry
python scripts/download_hbn_data.py --retry 3 --timeout 300
```

### Issue 2: MNE Can't Read Files
**Symptom**: "Not a valid BIDS dataset"
**Solution**:
```bash
# Verify BIDS structure
tree -L 3 data/raw/hbn
# Should show: sub-*/ses-*/eeg/
```

### Issue 3: Memory Error
**Symptom**: "Out of memory" during loading
**Solution**:
```bash
# Use memory-mapped loading
export MNE_USE_MEMMAP=1
# Or reduce batch size in config
```

---

## 📊 Progress Tracking

Track your progress with this simple checklist:

```bash
# Check current status
cat > check_status.sh << 'SCRIPT'
#!/bin/bash
echo "=== Data Acquisition Status ==="
echo ""

# Check packages
echo -n "✓ Packages: "
python -c "import mne, boto3" 2>/dev/null && echo "Installed" || echo "❌ Missing"

# Check data
echo -n "✓ Data Dir: "
[ -d "data/raw/hbn" ] && echo "Created" || echo "❌ Missing"

# Check files
echo -n "✓ Subjects: "
ls data/raw/hbn/sub-* 2>/dev/null | wc -l

# Check scripts
echo -n "✓ Scripts: "
[ -f "scripts/download_hbn_data.py" ] && echo "Ready" || echo "❌ Missing"

echo ""
echo "Run: bash check_status.sh"
SCRIPT

chmod +x check_status.sh
./check_status.sh
```

---

## 🚀 Next Steps After Completion

Once you've completed this checklist:

1. **Download Full Dataset** (optional, ~500GB):
   ```bash
   python scripts/download_hbn_data.py --all --parallel 4
   ```

2. **Run Full Test Suite**:
   ```bash
   pytest tests/test_data_*.py -v
   ```

3. **Start Training**:
   ```bash
   python src/training/train_cross_task.py --config configs/challenge1_baseline.yaml
   ```

---

## 📞 Need Help?

- Check logs: `tail -f data/logs/download.log`
- Verify setup: `python scripts/verify_environment.py`
- Debug mode: `python scripts/download_hbn_data.py --debug --verbose`

---

**Estimated Total Time**: 2-4 hours (depending on download speed)  
**Difficulty**: Beginner-Friendly  
**Priority**: 🔴 CRITICAL - Cannot proceed without data
