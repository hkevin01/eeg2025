# Challenge 2 Zero Variance Crisis

## The Problem
**ALL releases R2-R5 have ZERO VARIANCE for externalizing scores!**

### Evidence

| Release | Range | Mean | Std | Status |
|---------|-------|------|-----|--------|
| R1 Train | [0.325, 0.620] | - | >0 | ✅ VARIANCE |
| R2 Val | [0.620, 0.620] | 0.620 | 0.000 | ❌ ZERO VARIANCE |
| R3 Val | [-0.387, -0.387] | -0.387 | 0.000 | ❌ ZERO VARIANCE |
| R4 Val | [0.297, 0.297] | 0.297 | 0.000 | ❌ ZERO VARIANCE |
| R5 Val | [0.297, 0.297] | 0.297 | 0.000 | ❌ ZERO VARIANCE |

## Root Cause
Each release R2-R5 has all identical externalizing scores:
- R2: All = 0.620
- R3: All = -0.387
- R4: All = 0.297
- R5: All = 0.297

This makes Val NRMSE = 0.0 because:
```
NRMSE = RMSE / std(targets)
NRMSE = RMSE / 0.0 = undefined → shows as 0.0
```

## Solution Options

### Option 1: Use R1 Train/Val Split ✅ RECOMMENDED
Split R1 internally into train (80%) and validation (20%):
- Total R1 windows: 57,648
- Train: 46,118 windows (80%)
- Val: 11,530 windows (20%)

**Pros:**
- Both sets have proper variance
- Standard approach when no separate val set available
- Can still measure generalization

**Cons:**
- Smaller training set
- Val set from same subjects as train

### Option 2: Train on R1 Only, No Validation ❌ NOT RECOMMENDED
Just use all of R1 for training, no validation:

**Pros:**
- Maximum training data

**Cons:**
- No way to detect overfitting
- No early stopping
- No hyperparameter tuning
- Blind submission to competition

### Option 3: Use Multi-Release with Zero-Variance Val ❌ DOESN'T WORK
Keep using R2-R5 as validation:

**Cons:**
- Val NRMSE = 0.0 (useless metric)
- Can't detect overfitting
- Can't use early stopping
- Already tried this, doesn't work

## Decision: Option 1 (R1 Train/Val Split)

We will:
1. Use `torch.utils.data.random_split` to split R1 80/20
2. Train on 80% of R1
3. Validate on 20% of R1
4. Use early stopping based on validation NRMSE
5. Submit best model to competition for testing on R12

## Implementation Changes Required

1. Modify `MultiReleaseExternalizingDataset` to accept a `val_split` parameter
2. When `val_split=0.2`, internally split R1 into train/val
3. Keep Challenge 1 unchanged (it works with R3 validation)
4. Update Challenge 2 to use R1 with 80/20 split

## Expected Results

- Train NRMSE: Should decrease over epochs (0.8-1.2 range)
- Val NRMSE: Should be > 0 and similar to train (0.9-1.3 range)
- Both metrics will be meaningful for detecting overfitting

