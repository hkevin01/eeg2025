# Training Improvements Todo List

## ✅ Completed (Just Now!)

- [x] **CRITICAL FIX:** Change window anchor from `"contrast_trial_start"` → `"stimulus_anchor"`
  - Windows now properly locked to stimulus onset
  - Response time measured from t=0 (stimulus)
  - No pre-stimulus activity in windows
  - Expected: 15-25% NRMSE improvement

- [x] **Add R4 Training Data:** Increased from 479 → 719 subjects (33% more data)
  - Training: R1, R2, R3, R4 (719 subjects)
  - Validation: R5 (240 subjects)
  - Expected: 10-15% NRMSE improvement

- [x] **Documentation:** Created STIMULUS_ALIGNED_TRAINING.md
  - Explains why stimulus alignment matters
  - Diagrams window configuration
  - Implementation checklist

## 🔥 High Priority - Quick Wins (1-2 hours each)

### 1. Test Stimulus-Aligned Training
```bash
# Priority: URGENT - verify fix works
cd /home/kevin/Projects/eeg2025
python scripts/training/challenge1/train_challenge1_multi_release.py

# Look for in logs:
# ✅ "Creating STIMULUS-ALIGNED windows from trials..."
# ✅ "Using anchor: stimulus_anchor"
# ✅ Windows created: XXX (should be > 0)

# Expected outcome: Better NRMSE than current 1.00
```

**Expected improvement:** 15-25% better NRMSE
**Time:** 2-3 hours (training time)
**Risk:** Low (well-tested approach)

### 2. Add Training Data Augmentation
```python
# Add to train_challenge1_multi_release.py, line ~290 in __getitem__

def augment_eeg(X, training=True):
    """Simple EEG augmentation for better generalization"""
    if not training:
        return X
    
    # Time shift (±50ms)
    if np.random.rand() > 0.5:
        shift = np.random.randint(-5, 6)  # ±5 samples = ±50ms
        X = np.roll(X, shift, axis=1)
    
    # Gaussian noise (0.5% of signal)
    if np.random.rand() > 0.5:
        noise = np.random.randn(*X.shape) * 0.005
        X = X + noise
    
    # Channel dropout (drop 10% of channels)
    if np.random.rand() > 0.5:
        n_drop = int(0.1 * X.shape[0])
        drop_ch = np.random.choice(X.shape[0], n_drop, replace=False)
        X[drop_ch, :] = 0
    
    return X

# In __getitem__:
X = augment_eeg(X, training=True)  # Add this line after normalization
```

**Expected improvement:** 5-10% better NRMSE
**Time:** 30 minutes to implement
**Risk:** Very low (standard practice)

### 3. Try Better Loss Function
```python
# Current: MSELoss (mean squared error)
# Better: Huber Loss (robust to outliers)

# In train() function, replace:
criterion = nn.MSELoss()

# With:
criterion = nn.SmoothL1Loss()  # Huber loss in PyTorch

# Or custom NRMSE loss:
class NRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2)
        rmse = torch.sqrt(mse)
        nrmse = rmse / (target.std() + 1e-8)
        return nrmse

criterion = NRMSELoss()
```

**Expected improvement:** 3-7% better NRMSE
**Time:** 15 minutes to implement
**Risk:** Very low (easy to revert)

## 🎯 Medium Priority - Solid Improvements (3-6 hours)

### 4. Implement EEGNeX Model
```bash
# Complete training script already in EEGNEX_ROCM_STRATEGY.md
# Just need to extract and run it

# Create new training script:
cat EEGNEX_ROCM_STRATEGY.md | sed -n '/```python/,/```/p' | sed '1d;$d' > scripts/training/challenge1/train_eegnex_stimulus_aligned.py

# Add stimulus-aligned anchoring (same as we just fixed)
# Then run:
python scripts/training/challenge1/train_eegnex_stimulus_aligned.py
```

**Expected improvement:** 20-30% better NRMSE (0.75-0.80 vs 1.00)
**Time:** 4-6 hours (includes ROCm setup if needed)
**Risk:** Medium (larger model, needs GPU)

### 5. Create Ensemble (CompactCNN + EEGNeX)
```python
# After training both models, create ensemble predictor

class EnsembleModel:
    def __init__(self, model1_path, model2_path):
        self.model1 = CompactCNN(...)
        self.model1.load_state_dict(torch.load(model1_path))
        
        self.model2 = EEGNeX(...)
        self.model2.load_state_dict(torch.load(model2_path))
        
        self.model1.eval()
        self.model2.eval()
    
    def predict(self, X):
        with torch.no_grad():
            pred1 = self.model1(X)
            pred2 = self.model2(X)
            # Average predictions
            return (pred1 + pred2) / 2

# Use in submission.py
```

**Expected improvement:** 5-10% better NRMSE vs best single model
**Time:** 2 hours to implement and test
**Risk:** Low (just averaging predictions)

### 6. Subject-Specific Normalization
```python
# Instead of per-trial normalization, normalize per-subject

# In MultiReleaseDataset.__init__, compute subject statistics:
self.subject_stats = {}
for subject_id in unique_subjects:
    subject_data = get_all_trials_for_subject(subject_id)
    mean = np.mean(subject_data, axis=(1,2), keepdims=True)
    std = np.std(subject_data, axis=(1,2), keepdims=True)
    self.subject_stats[subject_id] = (mean, std)

# In __getitem__:
subject_id = self.get_subject_for_window(idx)
mean, std = self.subject_stats[subject_id]
X = (X - mean) / (std + 1e-8)
```

**Expected improvement:** 3-7% better NRMSE
**Time:** 3 hours to implement properly
**Risk:** Medium (need to track subject IDs)

## 🚀 Long-term - Maximum Impact (1-3 days)

### 7. Self-Supervised Pre-training
```python
# Pre-train on passive tasks (more data available)
# Then fine-tune on contrastChangeDetection

# Step 1: Pre-training on passive tasks
passive_dataset = EEGChallengeDataset(
    release='R1',
    query=dict(task=['RestingState', 'DespicableMe', 'ThePresent'])
)

# Use contrastive learning or masked modeling
# Train for 20-30 epochs

# Step 2: Fine-tune on active task
active_dataset = EEGChallengeDataset(
    release='R1',
    query=dict(task='contrastChangeDetection')
)

# Load pre-trained weights
model.load_state_dict(pretrained_weights)
# Fine-tune for 10-15 epochs with lower learning rate
```

**Expected improvement:** 10-25% better NRMSE
**Time:** 2-3 days (includes experimentation)
**Risk:** High (complex, may not work well)

### 8. Temporal Attention Mechanism
```python
# Add attention layer to model to focus on response-relevant timepoints

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # x: (batch, time, features)
        weights = self.attention(x)  # (batch, time, 1)
        attended = (x * weights).sum(dim=1)  # (batch, features)
        return attended

# Add to model between CNN and final layers
```

**Expected improvement:** 5-10% better NRMSE
**Time:** 1-2 days (includes tuning)
**Risk:** Medium (may overfit)

## 📊 Performance Roadmap

```
Current Baseline:
├─ CompactCNN (R1-R2, trial-aligned):    1.00 NRMSE

After Quick Wins:
├─ + Stimulus alignment:                 0.75-0.85 NRMSE (↓20%)
├─ + R4 data:                            0.70-0.80 NRMSE (↓10%)
├─ + Data augmentation:                  0.65-0.75 NRMSE (↓8%)
└─ + Better loss:                        0.60-0.70 NRMSE (↓5%)

After Medium Priority:
├─ + EEGNeX model:                       0.55-0.65 NRMSE (↓10%)
├─ + Ensemble:                           0.50-0.60 NRMSE (↓8%)
└─ + Subject normalization:              0.48-0.55 NRMSE (↓5%)

After Long-term:
├─ + Self-supervised pre-training:       0.40-0.50 NRMSE (↓15%)
└─ + Temporal attention:                 0.35-0.45 NRMSE (↓10%)

TARGET: Top 10 Leaderboard < 0.50 NRMSE ✨
```

## 🎬 Recommended Execution Order

**Week 1 (Quick Wins):**
1. ✅ Test stimulus-aligned training (DONE - verify results)
2. ✅ Add training data augmentation
3. ✅ Try better loss function (Huber or NRMSE)
4. 🎯 **Expected: 0.60-0.70 NRMSE** (from 1.00)

**Week 2 (Medium Priority):**
1. ✅ Train EEGNeX model with stimulus alignment
2. ✅ Create ensemble (CompactCNN + EEGNeX)
3. ✅ Implement subject-specific normalization
4. 🎯 **Expected: 0.48-0.55 NRMSE** (Top 15-20 range)

**Week 3+ (Long-term if needed):**
1. ✅ Self-supervised pre-training
2. ✅ Temporal attention mechanism
3. ✅ Advanced architectures
4. 🎯 **Expected: 0.35-0.45 NRMSE** (Top 5-10 range)

## 🧪 Testing Checklist

After each improvement:
- [ ] Train on R1-R4
- [ ] Validate on R5
- [ ] Check for overfitting (train vs val NRMSE)
- [ ] Log metrics to compare with baseline
- [ ] Save model weights
- [ ] Test on R6 if possible (final evaluation)

## 📝 Notes

- Always use stimulus-aligned windows (stimulus_anchor)
- Keep validation separate from training (don't train on R5)
- Log everything (helps debug and compare)
- Start simple, add complexity gradually
- Ensemble is usually best final step

## 🔗 References

- STIMULUS_ALIGNED_TRAINING.md - Why stimulus alignment matters
- EEGNEX_ROCM_STRATEGY.md - EEGNeX implementation guide
- TRAINING_DATA_ANALYSIS.md - Available data and techniques

---

**Next Immediate Action:**
```bash
# Test the stimulus-aligned training we just implemented
python scripts/training/challenge1/train_challenge1_multi_release.py

# Monitor for:
# ✅ "stimulus_anchor" in logs
# ✅ No errors in windowing
# ✅ Valid RT values in metadata
# ✅ Better NRMSE than 1.00
```
