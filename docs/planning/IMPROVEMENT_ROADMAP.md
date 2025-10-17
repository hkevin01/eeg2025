# 🚀 Priority 1: Multi-Release Training Implementation

**Goal:** Train on R1+R2+R3 instead of R1+R2 to reduce overfitting  
**Expected Impact:** Score 2.01 → 1.5 (top 30)  
**Time Required:** 2-3 hours (implementation 30 min + training 2 hours)

---

## 📋 Todo List

```markdown
- [ ] Step 1: Create improved training scripts (30 min)
- [ ] Step 2: Train Challenge 1 with R1+R2+R3 (1 hour)
- [ ] Step 3: Train Challenge 2 with R1+R2+R3 (1 hour)
- [ ] Step 4: Create submission package (15 min)
- [ ] Step 5: Upload and verify new scores (15 min)
```

---

## 🔧 Step 1: Create Improved Training Scripts

### Option A: Simple Approach (Recommended for Tonight)

**Modify `train_challenge1_multi_release.py`:**

Change lines 462-476 from:
```python
# OLD: Train R1+R2, Validate R3
train_dataset = MultiReleaseDataset(
    releases=['R1', 'R2'],
    mini=False,
    cache_dir='data/raw'
)

val_dataset = MultiReleaseDataset(
    releases=['R3'],
    mini=False,
    cache_dir='data/raw'
)
```

To:
```python
# NEW: Train R1+R2+R3 with 80/20 split
all_dataset = MultiReleaseDataset(
    releases=['R1', 'R2', 'R3'],
    mini=False,
    cache_dir='data/raw'
)

# Split into train/val (80/20)
train_size = int(0.8 * len(all_dataset))
val_size = len(all_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    all_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # Reproducible split
)
```

**Same changes for Challenge 2:**
- File: `scripts/train_challenge2_multi_release.py`
- Lines: 401-415 (similar structure)

### Option B: Cross-Validation Ensemble (For Tomorrow)

Create 3 separate models:
```python
# Model 1: Train R1+R2, Val R3
# Model 2: Train R1+R3, Val R2
# Model 3: Train R2+R3, Val R1
```

Then average predictions in submission.py

---

## 📝 Detailed Implementation: Option A (Tonight)

### Create New Script: `train_challenge1_all_releases.py`

```bash
# Copy existing script
cp scripts/train_challenge1_multi_release.py scripts/train_challenge1_all_releases.py
```

### Modify the new script:

**Line 69:** Update header
```python
# OLD:
print("Training on: R1, R2, R3")

# NEW:
print("Training on: R1, R2, R3 (COMBINED - 80/20 split)")
```

**Lines 462-476:** Change dataset loading
```python
#!/usr/bin/env python3
"""
UPDATED STRATEGY: Train on ALL available releases to reduce overfitting

Previous submission:
- Training: R1 + R2
- Validation: R3
- Test Result: Overall 2.013 (Position #47)
- Problem: 4x overfitting (val 0.65 → test 2.01)

New strategy:
- Training: R1 + R2 + R3 (80% combined)
- Validation: R1 + R2 + R3 (20% combined)
- Expected: Better generalization to test set
"""

import torch

def load_datasets():
    """Load all releases and split into train/val"""
    
    print("\n📦 Loading ALL releases (R1+R2+R3)...")
    print("⚠️  Will split 80% train / 20% validation")
    
    # Load all data
    all_dataset = MultiReleaseDataset(
        releases=['R1', 'R2', 'R3'],
        mini=False,
        cache_dir='data/raw'
    )
    
    print(f"Total samples: {len(all_dataset)}")
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(all_dataset))
    val_size = len(all_dataset) - train_size
    
    print(f"Train samples: {train_size}")
    print(f"Val samples: {val_size}")
    
    # Reproducible split
    train_dataset, val_dataset = torch.utils.data.random_split(
        all_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, val_dataset


# In main():
if __name__ == "__main__":
    # ... existing setup code ...
    
    # Load datasets
    train_dataset, val_dataset = load_datasets()
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # ... rest of training code ...
```

**Line 500:** Update expected performance
```python
# OLD:
print(f"Expected (R1-R5): ~0.70 validation → ~1.4 test (2x better!)")

# NEW:
print(f"Expected with R1+R2+R3: ~0.8 validation → ~1.5 test")
print(f"Should improve from 2.01 to ~1.5 (25% improvement!)")
```

---

## 📝 Detailed Implementation: Challenge 2

### Create: `train_challenge2_all_releases.py`

```bash
# Copy existing script
cp scripts/train_challenge2_multi_release.py scripts/train_challenge2_all_releases.py
```

### Key changes:

**Lines 401-415:** Update dataset loading
```python
# OLD: Only R1+R2
print("\n�� Loading R1+R2 data...")
train_dataset = MultiReleaseDataset(
    releases=['R1', 'R2'],
    mini=False,
    cache_dir='data/raw'
)

# NEW: All releases
print("\n📦 Loading ALL releases (R1+R2+R3)...")
all_dataset = MultiReleaseDataset(
    releases=['R1', 'R2', 'R3'],
    mini=False,
    cache_dir='data/raw'
)

# Split 80/20
train_size = int(0.8 * len(all_dataset))
val_size = len(all_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    all_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
```

---

## 🏃 Step 2: Train Challenge 1

```bash
# Run new training script
python scripts/train_challenge1_all_releases.py > logs/train_c1_all_releases.log 2>&1

# Monitor progress
tail -f logs/train_c1_all_releases.log
```

**Expected Output:**
```
📦 Loading ALL releases (R1+R2+R3)...
Total samples: ~50,000-60,000
Train samples: ~40,000-48,000
Val samples: ~10,000-12,000

Epoch 1/50:
  Train Loss: 1.2345
  Val NRMSE: 0.8500

Epoch 10/50:
  Train Loss: 0.7800
  Val NRMSE: 0.7200

...

Best Val NRMSE: 0.7000-0.8000 ← Better generalization!
Saved: weights/weights_challenge_1_all_releases.pt
```

**Training Time:** ~1 hour (50 epochs)

---

## 🏃 Step 3: Train Challenge 2

```bash
# Run Challenge 2 training
python scripts/train_challenge2_all_releases.py > logs/train_c2_all_releases.log 2>&1

# Monitor
tail -f logs/train_c2_all_releases.log
```

**Expected Output:**
```
📦 Loading ALL releases (R1+R2+R3)...
Total samples: ~50,000-60,000
Train samples: ~40,000-48,000
Val samples: ~10,000-12,000

Best Val NRMSE: 0.400-0.500 ← Slight increase from 0.297
Saved: weights/weights_challenge_2_all_releases.pt
```

**Note:** Validation might be slightly worse (0.3 → 0.4-0.5) but test should improve!

**Training Time:** ~1 hour (50 epochs)

---

## 📦 Step 4: Create Submission Package

### Update submission.py

**File: `submission.py`**

Change weight file names:
```python
# OLD:
'weights_challenge_1_multi_release.pt'
'weights_challenge_2_multi_release.pt'

# NEW:
'weights_challenge_1_all_releases.pt'
'weights_challenge_2_all_releases.pt'
```

### Package submission

```bash
# Create new submission directory
mkdir -p submission_v2

# Copy files
cp weights/weights_challenge_1_all_releases.pt submission_v2/
cp weights/weights_challenge_2_all_releases.pt submission_v2/
cp submission.py submission_v2/
cp METHODS_DOCUMENT.pdf submission_v2/

# Create zip
cd submission_v2
zip -r ../submission_v2.zip .
cd ..

# Move to Downloads
mv submission_v2.zip ~/Downloads/

# Verify size
ls -lh ~/Downloads/submission_v2.zip
```

**Expected Size:** ~600 KB (similar to v1)

---

## 📤 Step 5: Upload & Verify

1. **Upload to Codabench:**
   - Go to competition page
   - Submit `submission_v2.zip`
   - Add description: "Multi-release training (R1+R2+R3) for better generalization"

2. **Wait for Results (~20 min)**

3. **Expected Scores:**
```json
{
  "overall": 1.4-1.6,      ← 25-30% improvement from 2.01!
  "challenge1": 2.0-2.5,   ← 50% improvement from 4.05!
  "challenge2": 0.8-1.0    ← 20-30% improvement from 1.14!
}
```

4. **Expected Rank:** #20-30 (from #47)

---

## ⚡ Quick Start Commands

### Tonight (3 hours total):

```bash
# 1. Create new scripts (5 min)
cp scripts/train_challenge1_multi_release.py scripts/train_challenge1_all_releases.py
cp scripts/train_challenge2_multi_release.py scripts/train_challenge2_all_releases.py

# 2. Edit scripts to use R1+R2+R3 (25 min)
# TODO: Make the changes described above

# 3. Train Challenge 1 (1 hour)
python scripts/train_challenge1_all_releases.py > logs/train_c1_all_releases.log 2>&1 &
tail -f logs/train_c1_all_releases.log

# 4. Train Challenge 2 (1 hour) 
python scripts/train_challenge2_all_releases.py > logs/train_c2_all_releases.log 2>&1 &
tail -f logs/train_c2_all_releases.log

# 5. Create submission (15 min)
mkdir -p submission_v2
cp weights/weights_challenge_*_all_releases.pt submission_v2/
cp submission.py submission_v2/
cp METHODS_DOCUMENT.pdf submission_v2/
cd submission_v2 && zip -r ../submission_v2.zip . && cd ..
mv submission_v2.zip ~/Downloads/

# 6. Upload to Codabench
# ... manual upload ...

# Total time: ~3 hours
```

---

## 🔍 Validation Checklist

Before uploading:

- [ ] Both weight files exist and are ~300-400 KB each
- [ ] submission.py uses correct weight filenames
- [ ] submission_v2.zip is ~600 KB
- [ ] Training logs show validation NRMSE < 1.0 for both challenges
- [ ] No errors in training logs

---

## 📊 Success Criteria

**Minimum Success:**
- Overall score < 1.8 (from 2.01)
- Rank < 35 (from #47)

**Target Success:**
- Overall score < 1.5
- Rank < 25

**Best Case:**
- Overall score < 1.3
- Rank < 20

---

## 🚨 Potential Issues & Solutions

### Issue: "Validation worse than before"

**Symptom:**
```
Previous Val: 0.65
New Val: 0.80-0.90
```

**Explanation:**
- Normal! You're now validating on a mix of all releases
- Previous validation was only R3 (overly optimistic)
- Test performance will be better even if validation is worse

**Action:** Proceed with submission

---

### Issue: "Training takes too long"

**Solution:**
```python
# Reduce epochs if needed
epochs = 30  # Instead of 50

# Or use smaller batch size
batch_size = 16  # Instead of 32
```

---

### Issue: "Out of memory"

**Solution:**
```python
# Reduce batch size
batch_size = 16  # or 8

# Or reduce num_workers
num_workers = 2  # Instead of 4
```

---

## 🎯 Next Steps After V2 Submission

If V2 improves to ~1.5 (top 25), then:

1. **Priority 2: 3-Fold Ensemble** (tomorrow)
   - Train 3 models with different val sets
   - Average predictions
   - Expected: 1.5 → 1.2-1.3 (top 15)

2. **Priority 3: Architecture Improvements** (weekend)
   - Add attention mechanisms
   - Try different architectures
   - Expected: 1.2 → 1.0 (top 10)

---

## 💡 Key Insights

**Why this will work:**
1. More diverse training data → better generalization
2. Test set contains subjects from multiple releases
3. Training on single release combination was too limited

**Why validation might look worse:**
1. You're now validating on diverse data (realistic)
2. Previous validation was single release (unrealistic)
3. Test scores will improve even if validation increases

**Bottom line:**
- Don't worry if validation NRMSE increases from 0.65 to 0.80
- What matters is test score (currently 2.01)
- Expect test to improve to ~1.5 even with "worse" validation

---

**Ready to start? Let's do this! 🚀**
