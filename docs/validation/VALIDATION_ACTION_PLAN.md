# 🎯 Better Validation Strategy - Action Plan

**Date:** October 28, 2025  
**Goal:** Fix validation to predict test performance  
**Timeline:** 4 days until competition deadline  

---

## 📋 Todo List

### Phase 1: Immediate Actions (Today - 1 hour)

```markdown
- [ ] Step 1: Upload current submission to get baseline test score
- [ ] Step 2: Analyze whether val 0.9954 correlates better than val 0.16
- [ ] Step 3: Decide if subject-aware validation is worth the time
```

### Phase 2: Subject-Aware Validation (If needed - 3-4 hours)

```markdown
- [ ] Step 4: Re-cache R1-R4 data WITH subject IDs (1-2 hours)
- [ ] Step 5: Create subject-aware training script (30 min)
- [ ] Step 6: Train with subject-level splits (1 hour)
- [ ] Step 7: Submit and compare test scores (10-15 min)
```

### Phase 3: Advanced Techniques (If time permits - 2-3 hours)

```markdown
- [ ] Step 8: Implement K-fold cross-validation (1 hour)
- [ ] Step 9: Try ensemble of multiple folds (1 hour)
- [ ] Step 10: Final submission with best approach
```

---

## 🚀 Detailed Action Plan

### Step 1: Upload Current Submission (NOW)

**Goal:** Get test score for ALL R-sets training (val 0.9954)

**Actions:**
1. Go to https://www.codabench.org/competitions/2948/
2. Upload `submission_all_rsets_v1.zip`
3. Wait ~10-15 minutes for evaluation
4. Download `scoring_result.zip`

**Expected outcomes:**
- **Best case:** C1 < 0.95 → Validation DOES correlate with test!
- **Baseline:** C1 ≈ 1.0 → No improvement, validation unreliable
- **Worst case:** C1 > 1.1 → Made things worse

**Decision:**
- If C1 < 0.95: Keep this approach, optimize hyperparameters
- If C1 ≈ 1.0-1.1: Need subject-aware validation
- If C1 > 1.1: Revert to quick_fix, try different strategy

---

### Step 2: Re-Cache Data WITH Subject IDs

**Why:** Fix subject leakage (same subject in train + val)

**Commands:**
```bash
# Run caching script (1-2 hours)
cd /home/kevin/Projects/eeg2025
python scripts/preprocessing/cache_challenge1_with_subjects.py

# Expected output:
# data/cached/challenge1_R1_windows_with_subjects.h5  (~900 MB)
# data/cached/challenge1_R2_windows_with_subjects.h5  (~950 MB)
# data/cached/challenge1_R3_windows_with_subjects.h5  (~1.2 GB)
# data/cached/challenge1_R4_windows_with_subjects.h5  (~2.1 GB)

# Verify subject IDs saved
python -c "
import h5py
with h5py.File('data/cached/challenge1_R1_windows_with_subjects.h5', 'r') as f:
    print('Keys:', list(f.keys()))
    print('Subjects shape:', f['subject_ids'].shape)
    print('Unique subjects:', len(set(f['subject_ids'][:])))
"
```

**Script already created:**
- `scripts/preprocessing/cache_challenge1_with_subjects.py`
- Extracts subject ID from raw.filenames path
- Saves as HDF5 dataset: `subject_ids`

**Next:** Create training script with subject-aware splits

---

### Step 3: Subject-Aware Training Script

**Goal:** Train with NO subject overlap between train/val

**Create:** `scripts/experiments/train_c1_subject_aware.py`

**Key changes from current script:**

```python
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

# Load data WITH subject IDs
def load_data_with_subjects(h5_files):
    all_eeg = []
    all_labels = []
    all_subjects = []
    
    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            all_eeg.append(f['eeg'][:])
            all_labels.append(f['labels'][:])
            all_subjects.append(f['subject_ids'][:])
    
    return (
        np.concatenate(all_eeg),
        np.concatenate(all_labels),
        np.concatenate(all_subjects)
    )

# Split by SUBJECT, not samples
def subject_aware_split(eeg, labels, subjects, test_size=0.1, random_state=42):
    unique_subjects = np.unique(subjects)
    print(f"Total subjects: {len(unique_subjects)}")
    
    # Split subjects
    train_subjects, val_subjects = train_test_split(
        unique_subjects,
        test_size=test_size,
        random_state=random_state
    )
    
    # Create boolean masks
    train_mask = np.isin(subjects, train_subjects)
    val_mask = np.isin(subjects, val_subjects)
    
    # Split data
    train_eeg = eeg[train_mask]
    train_labels = labels[train_mask]
    val_eeg = eeg[val_mask]
    val_labels = labels[val_mask]
    
    print(f"Train: {len(train_subjects)} subjects, {len(train_eeg)} samples")
    print(f"Val:   {len(val_subjects)} subjects, {len(val_eeg)} samples")
    
    # Verify no overlap
    overlap = set(train_subjects) & set(val_subjects)
    assert len(overlap) == 0, f"Subject overlap detected: {overlap}"
    print("✅ No subject overlap between train/val")
    
    return train_eeg, train_labels, val_eeg, val_labels

# Load data
cache_files = [
    'data/cached/challenge1_R1_windows_with_subjects.h5',
    'data/cached/challenge1_R2_windows_with_subjects.h5',
    'data/cached/challenge1_R3_windows_with_subjects.h5',
    'data/cached/challenge1_R4_windows_with_subjects.h5',
]
eeg, labels, subjects = load_data_with_subjects(cache_files)

# Subject-aware split
train_eeg, train_labels, val_eeg, val_labels = subject_aware_split(
    eeg, labels, subjects, test_size=0.1
)

# Rest of training is same as before...
```

**Benefits:**
- Forces model to generalize to NEW subjects
- Validation score becomes more predictive of test
- More realistic evaluation

**Trade-off:**
- Validation NRMSE will be higher (more difficult)
- But it will correlate better with test performance

---

### Step 4: Train and Submit

**Commands:**
```bash
# Start training in tmux (stable)
tmux new -s eeg_subject_aware

# Run training
cd /home/kevin/Projects/eeg2025
python scripts/experiments/train_c1_subject_aware.py

# Detach: Ctrl+B, then D
# Monitor: bash monitor_training.sh
# Attach: tmux attach -t eeg_subject_aware
```

**Expected:**
- Validation NRMSE: Higher than 0.9954 (more realistic)
- Training time: ~1 hour
- Output: `weights/compact_cnn_subject_aware_state.pt`

**Create submission:**
```python
# scripts/create_submission_subject_aware.py
import zipfile
import shutil
from pathlib import Path

submission_dir = Path('submission_subject_aware_v1')
submission_dir.mkdir(exist_ok=True)

# Copy files
shutil.copy('submission.py', submission_dir / 'submission.py')
shutil.copy(
    'weights/compact_cnn_subject_aware_state.pt',
    submission_dir / 'compact_cnn_c1_cross_r123_val4_state.pt'  # Competition expects this name
)
shutil.copy(
    'weights/weights_challenge_2.pt',
    submission_dir / 'weights_challenge_2.pt'
)

# Create ZIP
with zipfile.ZipFile('submission_subject_aware_v1.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in submission_dir.glob('*'):
        zf.write(f, f.name)

print("✅ Created: submission_subject_aware_v1.zip")
```

**Upload to Codabench** and check C1 score!

---

## 📊 Expected Results

### Current Random Split (all_rsets_v1)

| Metric | Train | Val | Test (Expected) |
|--------|-------|-----|-----------------|
| NRMSE | Low | 0.9954 | 0.95-1.10? |
| Subject overlap | ✅ | ❌ | ❌ |
| Predictive? | - | ❓ | Unknown |

### Subject-Aware Split (subject_aware_v1)

| Metric | Train | Val | Test (Expected) |
|--------|-------|-----|-----------------|
| NRMSE | Low | 1.05-1.15 | 1.00-1.10? |
| Subject overlap | ✅ | ✅ | ✅ |
| Predictive? | - | ✅ | Should correlate! |

**Key insight:** Subject-aware validation NRMSE should predict test NRMSE

---

## 🎯 Decision Tree

### After all_rsets_v1 submission:

**If C1 < 0.95:**
- ✅ Current approach works!
- Don't fix what's not broken
- Focus on hyperparameter tuning, architecture, ensemble

**If C1 = 0.95-1.0:**
- ⚠️ Small improvement over untrained (1.0015)
- Worth trying subject-aware validation
- Could push us to top 3 (C1 < 0.93)

**If C1 = 1.0-1.1:**
- ❌ No improvement or worse
- MUST implement subject-aware validation
- Current val doesn't predict test at all

**If C1 > 1.1:**
- 🚨 Training makes things worse!
- Revert to quick_fix (1.0065)
- Try completely different approach (features, architecture)

---

## ⏱️ Time Estimates

| Phase | Task | Time | Priority |
|-------|------|------|----------|
| 1 | Upload current submission | 2 min | ⭐⭐⭐ |
| 1 | Wait for results | 10-15 min | ⭐⭐⭐ |
| 1 | Analyze and decide | 5 min | ⭐⭐⭐ |
| 2 | Re-cache with subjects | 1-2 hours | ⭐⭐ |
| 2 | Create training script | 30 min | ⭐⭐ |
| 2 | Train model | 1 hour | ⭐⭐ |
| 2 | Create and upload submission | 15 min | ⭐⭐ |
| 3 | K-fold CV | 2-3 hours | ⭐ |
| 3 | Ensemble | 1-2 hours | ⭐ |

**Total:** 6-9 hours (spread over 2 days)

---

## 🚨 Critical Success Factors

### Must Do:

1. **Upload current submission FIRST** ⭐⭐⭐
   - Need test score to decide next steps
   - Don't waste time if current approach works

2. **Keep safety net** ⭐⭐⭐
   - quick_fix (1.0065) already submitted
   - Don't break what works
   - Always have backup

3. **Monitor time** ⭐⭐
   - 4 days until deadline
   - Each experiment takes time
   - Focus on highest-impact changes

### Nice to Have:

1. **Subject-aware validation**
   - Only if current submission fails
   - Takes 3-4 hours total
   - Should improve test correlation

2. **K-fold CV + Ensemble**
   - Only if we have time
   - More robust but time-consuming
   - Diminishing returns

---

## 📝 Documentation to Create

### After subject-aware training:

1. **SUBJECT_AWARE_TRAINING_OCT28.md**
   - Why we need it
   - How it works
   - Results comparison

2. **VALIDATION_COMPARISON_OCT28.md**
   - Random split vs subject-aware
   - Val vs test correlation
   - Which to use going forward

3. **Update STATUS_SUMMARY**
   - Current best approach
   - Next steps
   - Leaderboard position

---

## 🎓 Key Learnings

### Why Validation Failed:

1. **Subject leakage:** Same subject in train + val
2. **Distribution mismatch:** Val not representative of test
3. **Overfitting:** Model memorizes validation patterns

### How to Fix:

1. **Subject-level splits:** No subject overlap
2. **Multiple metrics:** Don't trust NRMSE alone
3. **Hold-out test:** Reserve R4, never touch during training
4. **K-fold CV:** Robust estimates across multiple folds

### Competition Strategy:

1. **Fast iteration:** 10-15 min per submission
2. **Use test as validation:** Only way to know for sure
3. **Keep backups:** Don't break working solutions
4. **Focus on impact:** Prioritize high-leverage changes

---

## 🚀 Next Immediate Action

**RIGHT NOW:**

1. Go to: https://www.codabench.org/competitions/2948/
2. Upload: `submission_all_rsets_v1.zip`
3. Wait: 10-15 minutes
4. Download: `scoring_result.zip`
5. Check: C1 score vs 1.0015 baseline
6. Decide: Continue current approach OR implement subject-aware

**Don't start re-caching until we know current submission results!**

---

*Created: October 28, 2025*  
*Status: Waiting for all_rsets_v1 test results*  
*Next: Decide based on C1 score*
