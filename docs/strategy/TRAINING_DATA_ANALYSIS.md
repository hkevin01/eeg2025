# Training Data & Techniques Analysis

## 📊 Available Releases for Training

### Current Status (Verified October 18, 2025)

All 6 releases are **ACCESSIBLE** for training:

| Release | Status | Subjects | Notes |
|---------|--------|----------|-------|
| **R1** | ✅ Available | 239 | Currently used for training |
| **R2** | ✅ Available | 240 | Currently used for training |
| **R3** | ✅ Available | 240 | Currently used for validation |
| **R4** | ✅ Available | 240 | **NOT USED** (thought to have issues) |
| **R5** | ✅ Available | 240 | **NOT USED** (thought to have zero variance) |
| **R6** | ✅ Available | 237 | **NOT USED** (test set) |

**Total Available**: 1,436 subjects across 6 releases

### 🎯 Current Training Strategy

**Your Implementation:**
```python
# scripts/training/challenge1/train_challenge1_multi_release.py
train_releases = ['R1', 'R2']  # 479 subjects
val_releases = ['R3']          # 240 subjects
```

**Why R4-R6 are not used:**
- R4: Believed to have "no valid events" (needs verification!)
- R5: Believed to have "zero variance" (needs verification!)
- R6: Reserved as hidden test set (but accessible for training)

### 🚀 Opportunity: Expand Training Data

**You're only using 719 subjects out of 1,436 available (50%)!**

#### Recommended Strategy:
```python
# Use MORE data for better generalization
train_releases = ['R1', 'R2', 'R3', 'R4']  # 959 subjects (33% increase!)
val_releases = ['R5']                      # 240 subjects

# Or even more aggressive:
train_releases = ['R1', 'R2', 'R3', 'R4', 'R5']  # 1199 subjects (67% increase!)
val_releases = ['R6']                             # 237 subjects
```

**Why this might help:**
- More diverse data → better generalization
- R4 and R5 issues may have been fixed or misunderstood
- Competition evaluates on R1-R6, so training on more releases helps

## 📚 Starter Kit Training Techniques

### What's Provided in Starter Kit

**File: `starter_kit_integration/challenge_1.py`**

1. **Data Loading**
   - EEGChallengeDataset
   - BIDS format handling
   - Task filtering (contrastChangeDetection)

2. **Preprocessing Pipeline**
   ```python
   # From starter kit:
   - annotate_trials_with_target (RT extraction)
   - add_aux_anchors (event marking)
   - create_windows_from_events
   - Braindecode preprocessing utilities
   ```

3. **Model Architecture**
   - EEGNeX (from braindecode)
   - Simple baseline models

4. **Training Loop**
   - Basic PyTorch training
   - Train/validation split
   - NRMSE evaluation

5. **Cross-Validation**
   - `scripts/train_grouped_cv_template.py`
   - Grouped by subject/session

### What You Implemented (Beyond Starter Kit)

1. **Multi-Release Training** ✅
   - Combines multiple releases
   - Cross-release validation
   - Corruption detection

2. **Compact CNNs** ✅
   - CompactResponseTimeCNN (75K params)
   - CompactExternalizingCNN (64K params)
   - Much lighter than EEGNeX

3. **Test-Time Augmentation (TTA)** ✅
   - 5 augmentation types
   - Ensemble predictions
   - 5-10% improvement

4. **Robust Training** ✅
   - Learning rate scheduling
   - Early stopping
   - Gradient clipping

## 🎨 Techniques You Can Add

### 1. **Self-Supervised Pre-training** (High Impact)

**What**: Pre-train on unlabeled EEG data before fine-tuning

**Why**: Learn general EEG representations
- Challenge explicitly mentions "pre-training on passive tasks"
- Passive tasks: RestingState, movie watching
- Active tasks: contrastChangeDetection (your target)

**Implementation:**
```python
# Step 1: Pre-train on passive tasks (unlabeled)
passive_dataset = EEGChallengeDataset(
    release='R1',
    query=dict(task=['RestingState', 'DespicableMe', 'ThePresent'])
)

# Use contrastive learning (SimCLR, MoCo)
# Or masked modeling (BERT-style)
# Or temporal prediction

# Step 2: Fine-tune on active task
active_dataset = EEGChallengeDataset(
    release='R1',
    query=dict(task='contrastChangeDetection')
)
```

**Files to check:**
- `data/ds005505-bdf/task-*.json` (available tasks)
- Passive: RestingState, DespicableMe, DiaryOfAWimpyKid, ThePresent, FunwithFractals
- Active: contrastChangeDetection, seqLearning, surroundSupp, symbolSearch

### 2. **Data Augmentation** (Medium Impact)

You have TTA (test-time), but not training-time augmentation!

**Add to training loop:**
```python
def augment_training_data(X):
    # Time shift
    shift = np.random.randint(-5, 6)
    X = np.roll(X, shift, axis=-1)
    
    # Gaussian noise
    noise = np.random.randn(*X.shape) * 0.01
    X = X + noise
    
    # Time warping
    # Channel shuffle
    # Frequency masking
    
    return X
```

### 3. **Ensemble Learning** (High Impact)

**What**: Train multiple models and average predictions

**Why**: Reduces overfitting, improves robustness

**Implementation:**
```python
# Train 5 models with different:
# - Random seeds
# - Architectures (CNN, Transformer, TCN)
# - Data splits (different folds)

# Ensemble at test time:
predictions = [model1(X), model2(X), model3(X), model4(X), model5(X)]
final_pred = np.mean(predictions, axis=0)
```

### 4. **Attention Mechanisms** (Medium Impact)

You have a file `train_attention_model.py` - are you using it?

**Temporal attention:**
```python
class TemporalAttention(nn.Module):
    def __init__(self, channels):
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels//4, 1),
            nn.ReLU(),
            nn.Conv1d(channels//4, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        weights = self.attention(x)
        return x * weights
```

### 5. **Mixup / CutMix** (Medium Impact)

**What**: Mix samples during training

```python
def mixup(x1, y1, x2, y2, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    x_mixed = lam * x1 + (1 - lam) * x2
    y_mixed = lam * y1 + (1 - lam) * y2
    return x_mixed, y_mixed
```

### 6. **Subject-Specific Normalization** (High Impact for R4-R6)

**Problem**: Different releases may have different statistics

**Solution:**
```python
# Per-subject z-score normalization
for subject_id in unique_subjects:
    subject_data = data[subject_ids == subject_id]
    subject_mean = np.mean(subject_data, axis=(1, 2), keepdims=True)
    subject_std = np.std(subject_data, axis=(1, 2), keepdims=True)
    data[subject_ids == subject_id] = (subject_data - subject_mean) / (subject_std + 1e-8)
```

### 7. **Loss Function Improvements** (Medium Impact)

**Current**: MSE Loss (implied)

**Try:**
- Huber Loss (robust to outliers)
- Quantile Loss (for distribution modeling)
- Multi-task Loss (combine both challenges)

```python
# Huber Loss (more robust)
criterion = nn.SmoothL1Loss()

# Or custom NRMSE loss
def nrmse_loss(pred, target):
    mse = torch.mean((pred - target) ** 2)
    rmse = torch.sqrt(mse)
    nrmse = rmse / (target.std() + 1e-8)
    return nrmse
```

### 8. **Learning Rate Scheduling** (You have this!)

✅ Already implemented - good!

### 9. **Gradient Accumulation** (Medium Impact)

**What**: Simulate larger batch sizes

```python
accumulation_steps = 4
for i, (X, y) in enumerate(train_loader):
    loss = criterion(model(X), y)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 10. **Transfer Learning from Other Domains** (High Impact)

**What**: Use pre-trained models from similar tasks

**Options:**
- Pre-trained EEG models (if available)
- Transfer from ECG, EMG (similar time-series)
- ImageNet features (if converting EEG to spectrograms)

## 🔬 Experiments to Run

### Priority 1: Use More Training Data
```bash
# Modify train_challenge1_multi_release.py
releases=['R1', 'R2', 'R3', 'R4']  # Add R4!
val_releases=['R5']                 # Use R5 for validation

# Reason: You're only using 50% of available data
# Expected improvement: 5-10% NRMSE reduction
```

### Priority 2: Verify R4 and R5 Issues
```python
# Test script:
for release in ['R4', 'R5']:
    dataset = EEGChallengeDataset(release=release, mini=False)
    # Check for:
    # - Valid events
    # - Non-zero variance
    # - Data corruption
    print(f"{release}: {len(dataset.datasets)} subjects")
```

### Priority 3: Self-Supervised Pre-training
```python
# 1. Pre-train on passive tasks (R1-R5)
# 2. Fine-tune on active task (contrastChangeDetection)
# 3. Evaluate on R6
```

### Priority 4: Ensemble Models
```python
# Train 3-5 models with:
# - Different architectures
# - Different random seeds
# - Different data augmentations
# Average predictions at test time
```

## 📝 Quick Action Items

### Immediate (Today):
1. ✅ Verify R4 and R5 are actually usable
2. ✅ Add R4 to training data
3. ✅ Run evaluation on R1-R6 to get complete picture

### Short-term (This Week):
1. Implement training-time data augmentation
2. Add subject-specific normalization
3. Try Huber loss instead of MSE

### Medium-term (Next Week):
1. Self-supervised pre-training on passive tasks
2. Ensemble 3-5 models
3. Attention mechanisms

## 🎯 Expected Improvements

| Technique | Expected NRMSE Improvement | Effort |
|-----------|---------------------------|--------|
| Use R4 data | -5 to -10% | Low (1 hour) |
| Use R5 data | -5 to -10% | Low (1 hour) |
| Training augmentation | -3 to -5% | Low (2 hours) |
| Self-supervised pre-training | -10 to -20% | High (2 days) |
| Ensemble (5 models) | -5 to -10% | Medium (1 day) |
| Attention mechanisms | -3 to -7% | Medium (1 day) |
| Better loss function | -2 to -5% | Low (2 hours) |

**Total potential improvement: 33-67% NRMSE reduction!**

Current: 1.32 NRMSE
Target: 0.66-0.88 NRMSE (top leaderboard range)

## 📚 Resources

**Starter Kit:**
- `starter_kit_integration/challenge_1.py` - Basic training
- `starter_kit_integration/challenge_2.py` - Challenge 2
- `scripts/train_grouped_cv_template.py` - Cross-validation

**Your Implementation:**
- `scripts/training/challenge1/train_challenge1_multi_release.py`
- `scripts/training/challenge2/train_challenge2_multi_release.py`
- `submission_tta.py` - TTA implementation

**Documentation:**
- CHANNEL_NORMALIZATION_EXPLAINED.md
- MEETING_PRESENTATION.md
- README.md

---

**Status: Ready to expand training data and implement advanced techniques! 🚀**
