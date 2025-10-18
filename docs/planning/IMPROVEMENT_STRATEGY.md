# Algorithm Improvement Strategy - EEG 2025 Competition

**Date:** October 15, 2025  
**Current Scores:** C1: 0.4680 | C2: 0.0808 | Overall: 0.1970  
**Goal:** Improve Challenge 1 (biggest room for improvement)

---

## 🎯 Priority Analysis

### Challenge 1: Most Room for Improvement
- **Current:** NRMSE 0.4680 (30% weight)
- **Challenge 2:** NRMSE 0.0808 (70% weight) - already excellent!
- **Strategy:** Focus on Challenge 1 improvements

**Why Focus on Challenge 1?**
- More room to improve (0.47 → 0.35 = 26% better)
- Challenge 2 is already near-perfect (0.08)
- Even 10% C1 improvement = 3% overall improvement

---

## 🚀 HIGH PRIORITY IMPROVEMENTS (1-3 days)

### 1. Test-Time Augmentation (TTA) ⭐ HIGHEST ROI

**Concept:** Average predictions from multiple augmented versions

**Implementation:**
```python
def predict_with_tta(model, data, n_augmentations=5):
    """Test-time augmentation for robust predictions"""
    predictions = []
    
    # Original prediction
    predictions.append(model(data))
    
    # Augmented predictions
    for _ in range(n_augmentations - 1):
        # Add small Gaussian noise
        noise = torch.randn_like(data) * 0.02
        aug_data = data + noise
        predictions.append(model(aug_data))
    
    # Average predictions
    return torch.stack(predictions).mean(dim=0)
```

**Expected Gain:** 5-10% NRMSE reduction (0.47 → 0.42)  
**Time:** 2-3 hours  
**Risk:** LOW (can always revert)  
**Validation:** Test on validation set first

---

### 2. Weighted Ensemble (Multiple Models) ⭐ PROVEN EFFECTIVE

**Concept:** Train 3-5 models with different seeds, average with optimal weights

**Implementation:**
```python
# Train 3 models with different seeds
seeds = [42, 123, 456]
models = []
for seed in seeds:
    model = train_model(seed=seed)
    models.append(model)

# Find optimal weights (validation set)
from scipy.optimize import minimize

def ensemble_loss(weights, models, X_val, y_val):
    weights = weights / weights.sum()  # Normalize
    preds = sum(w * m.predict(X_val) for w, m in zip(weights, models))
    return nrmse(y_val, preds)

# Optimize weights
initial_weights = np.ones(len(models)) / len(models)
result = minimize(ensemble_loss, initial_weights, 
                  args=(models, X_val, y_val),
                  bounds=[(0, 1)] * len(models))

optimal_weights = result.x / result.x.sum()
```

**Expected Gain:** 5-8% NRMSE reduction (0.47 → 0.43)  
**Time:** 4-6 hours (training + optimization)  
**Risk:** MEDIUM (increases complexity)  
**Note:** We already have ensemble models trained!

---

### 3. Frequency Domain Features ⭐ UNTAPPED POTENTIAL

**Concept:** Add spectral features to complement time-domain

**Implementation:**
```python
import scipy.signal as signal

def extract_frequency_features(data, fs=100):
    """Extract frequency domain features"""
    features = []
    
    for ch in range(data.shape[0]):  # For each channel
        # Compute power spectral density
        freqs, psd = signal.welch(data[ch], fs=fs, nperseg=min(256, len(data[ch])))
        
        # Band powers
        delta = psd[(freqs >= 0.5) & (freqs < 4)].mean()
        theta = psd[(freqs >= 4) & (freqs < 8)].mean()
        alpha = psd[(freqs >= 8) & (freqs < 13)].mean()
        beta = psd[(freqs >= 13) & (freqs < 30)].mean()
        gamma = psd[(freqs >= 30) & (freqs < 50)].mean()
        
        features.extend([delta, theta, alpha, beta, gamma])
    
    return np.array(features)

# Hybrid model: CNN + frequency features
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = ImprovedResponseTimeCNN()  # Time-domain
        self.freq_fc = nn.Linear(129 * 5, 128)  # Frequency features
        self.fusion = nn.Linear(128 + 128, 1)  # Combine both
    
    def forward(self, x):
        # Time-domain features
        time_features = self.cnn.feature_extractor(x)
        
        # Frequency features (compute on-the-fly or pre-compute)
        freq_features = extract_frequency_features_batch(x)
        freq_features = self.freq_fc(freq_features)
        
        # Fusion
        combined = torch.cat([time_features, freq_features], dim=1)
        return self.fusion(combined)
```

**Expected Gain:** 10-20% NRMSE reduction (0.47 → 0.38-0.42)  
**Time:** 6-8 hours (implementation + training)  
**Risk:** MEDIUM (requires validation)  
**Note:** Most competitors likely use time-domain only

---

### 4. Subject-Level Features (Metadata) ⭐ EASY WIN

**Concept:** Include age, sex as auxiliary inputs

**Why It Helps:** Response times correlate with age/development

**Implementation:**
```python
class MetadataEnhancedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = ImprovedResponseTimeCNN()
        
        # Metadata branch
        self.meta_fc = nn.Sequential(
            nn.Linear(2, 32),  # age, sex
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Fusion
        self.fusion = nn.Linear(128 + 16, 1)
    
    def forward(self, x, age, sex):
        # CNN features
        cnn_features = self.cnn.feature_extractor(x)
        
        # Metadata features
        metadata = torch.stack([age, sex], dim=1)
        meta_features = self.meta_fc(metadata)
        
        # Combine
        combined = torch.cat([cnn_features, meta_features], dim=1)
        return self.fusion(combined)
```

**Expected Gain:** 5-10% NRMSE reduction (0.47 → 0.42-0.44)  
**Time:** 3-4 hours  
**Risk:** LOW (easy to implement)  
**Data:** Already in participants.tsv

---

## 🔧 MEDIUM PRIORITY IMPROVEMENTS (3-7 days)

### 5. Attention Mechanisms

**Concept:** Let model learn which channels/timepoints are important

**Implementation:**
```python
class ChannelAttention(nn.Module):
    """Attention over EEG channels"""
    def __init__(self, num_channels=129):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(num_channels, num_channels // 4),
            nn.ReLU(),
            nn.Linear(num_channels // 4, num_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [batch, channels, time]
        # Pool over time
        pooled = x.mean(dim=2)  # [batch, channels]
        
        # Compute attention weights
        weights = self.attention(pooled)  # [batch, channels]
        
        # Apply attention
        return x * weights.unsqueeze(2)

class TemporalAttention(nn.Module):
    """Attention over time"""
    def __init__(self, seq_len=200):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(seq_len, seq_len // 4),
            nn.ReLU(),
            nn.Linear(seq_len // 4, seq_len),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [batch, channels, time]
        # Pool over channels
        pooled = x.mean(dim=1)  # [batch, time]
        
        # Compute attention weights
        weights = self.attention(pooled)  # [batch, time]
        
        # Apply attention
        return x * weights.unsqueeze(1)
```

**Expected Gain:** 8-15% NRMSE reduction (0.47 → 0.40-0.43)  
**Time:** 6-8 hours  
**Risk:** MEDIUM (adds complexity)

---

### 6. Multi-Task Learning (Response Time + Trial Type)

**Concept:** Predict both response time AND trial correctness simultaneously

**Why It Helps:** Shared representations, regularization

**Implementation:**
```python
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared encoder
        self.encoder = ImprovedResponseTimeCNN().feature_extractor
        
        # Task-specific heads
        self.rt_head = nn.Linear(128, 1)  # Response time
        self.correct_head = nn.Linear(128, 1)  # Correctness (binary)
    
    def forward(self, x):
        features = self.encoder(x)
        rt = self.rt_head(features)
        correct = torch.sigmoid(self.correct_head(features))
        return rt, correct

# Training with multi-task loss
def train_multitask(model, data, rt_labels, correct_labels):
    rt_pred, correct_pred = model(data)
    
    loss_rt = F.mse_loss(rt_pred, rt_labels)
    loss_correct = F.binary_cross_entropy(correct_pred, correct_labels)
    
    # Weighted combination
    total_loss = loss_rt + 0.5 * loss_correct  # Tune weight
    return total_loss
```

**Expected Gain:** 5-12% NRMSE reduction (0.47 → 0.41-0.45)  
**Time:** 8-10 hours  
**Risk:** MEDIUM (requires correctness labels)  
**Data:** May need to extract from CCD task events

---

### 7. Deeper Data Augmentation

**Concept:** More sophisticated augmentation strategies

**New Augmentations:**
```python
class AdvancedAugmentation:
    def __init__(self):
        pass
    
    def channel_dropout(self, data, p=0.1):
        """Randomly zero out channels"""
        mask = torch.rand(data.shape[0]) > p
        return data * mask.unsqueeze(1)
    
    def time_masking(self, data, max_mask=20):
        """Mask random time segments"""
        mask_len = random.randint(1, max_mask)
        mask_start = random.randint(0, data.shape[1] - mask_len)
        data[:, mask_start:mask_start+mask_len] = 0
        return data
    
    def mixup(self, data1, data2, label1, label2, alpha=0.2):
        """Mixup augmentation"""
        lam = np.random.beta(alpha, alpha)
        mixed_data = lam * data1 + (1 - lam) * data2
        mixed_label = lam * label1 + (1 - lam) * label2
        return mixed_data, mixed_label
    
    def frequency_shift(self, data, max_shift=2):
        """Apply small frequency shift"""
        # Use resampling to shift frequency content
        shift_factor = 1 + random.uniform(-max_shift/100, max_shift/100)
        # Resample (pseudo-code, use scipy.signal.resample)
        return resample_signal(data, shift_factor)
```

**Expected Gain:** 3-8% NRMSE reduction (0.47 → 0.43-0.46)  
**Time:** 4-6 hours  
**Risk:** LOW (can A/B test)

---

## 🧪 EXPERIMENTAL IMPROVEMENTS (7-14 days)

### 8. Transformer Architecture

**Concept:** Replace CNN with attention-based model

**Pros:**
- Captures long-range dependencies
- State-of-the-art in many domains

**Cons:**
- Needs more data (we have 420 samples)
- Harder to train
- More parameters

**Expected Gain:** -10% to +15% (highly uncertain)  
**Time:** 12-16 hours  
**Risk:** HIGH (may overfit)  
**Recommendation:** Try ONLY if other methods fail

---

### 9. Pre-training on RestingState Data

**Concept:** Pre-train on Challenge 2 data (2,315 samples), then fine-tune for Challenge 1

**Why It Might Help:** Transfer learned EEG representations

**Implementation:**
```python
# Step 1: Pre-train on RestingState (Challenge 2 data)
pretrain_model = ExternalizingCNN()
train_on_challenge2_data(pretrain_model)

# Step 2: Transfer weights to Challenge 1 model
challenge1_model = ImprovedResponseTimeCNN()
# Copy convolutional layers
challenge1_model.conv1.weight = pretrain_model.conv1.weight
challenge1_model.conv2.weight = pretrain_model.conv2.weight
# ... etc

# Step 3: Fine-tune on Challenge 1 data
fine_tune_on_challenge1_data(challenge1_model, lr=1e-5)
```

**Expected Gain:** 5-15% NRMSE reduction (0.47 → 0.40-0.45)  
**Time:** 6-10 hours  
**Risk:** MEDIUM (may not transfer well - different tasks)

---

### 10. Spatial Filtering (Source Localization)

**Concept:** Transform channel-level EEG to brain region activity

**Why It Might Help:** More interpretable, closer to underlying neural activity

**Implementation:** Use MNE-Python's source estimation

**Expected Gain:** 0-20% (highly uncertain)  
**Time:** 10-15 hours  
**Risk:** HIGH (complex, may not help)  
**Recommendation:** LOW priority

---

## 📊 IMPROVEMENT ROADMAP

### Week 1 (Days 1-7): Quick Wins

**Days 1-2:**
- [x] Validation experiments (DONE!)
- [ ] Test-time augmentation (2-3 hours)
- [ ] Expected: 0.47 → 0.42 (-11%)

**Days 3-4:**
- [ ] Subject metadata features (3-4 hours)
- [ ] Weighted ensemble (4-6 hours)
- [ ] Expected: 0.42 → 0.38 (-9%)

**Days 5-7:**
- [ ] Frequency domain features (6-8 hours)
- [ ] Advanced augmentation (4-6 hours)
- [ ] Expected: 0.38 → 0.34 (-11%)

**Week 1 Target:** 0.47 → 0.34 (-28% improvement!)

### Week 2 (Days 8-14): Advanced Techniques

**Days 8-10:**
- [ ] Attention mechanisms (6-8 hours)
- [ ] Multi-task learning (8-10 hours)
- [ ] Expected: 0.34 → 0.30 (-12%)

**Days 11-14:**
- [ ] Pre-training experiments (6-10 hours)
- [ ] Final optimization & tuning
- [ ] Expected: 0.30 → 0.28 (-7%)

**Week 2 Target:** 0.34 → 0.28 (-18% improvement!)

### Final Days (15-18): Polish & Submit

- [ ] Best model selection
- [ ] Final ensemble
- [ ] Documentation update
- [ ] Submit improved version

---

## 🎯 REALISTIC IMPROVEMENT PROJECTIONS

### Conservative Estimate
```
Current:  0.4680 (Challenge 1)
+ TTA:    0.4430 (-5%)
+ Meta:   0.4210 (-5%)
Final:    0.4210 (-10% total)

Overall:  0.1970 → 0.1890 (-4% overall)
```

### Optimistic Estimate
```
Current:  0.4680 (Challenge 1)
+ TTA:    0.4210 (-10%)
+ Meta:   0.3790 (-10%)
+ Freq:   0.3220 (-15%)
+ Ens:    0.3060 (-5%)
Final:    0.3060 (-35% total)

Overall:  0.1970 → 0.1483 (-25% overall)
```

### Dream Scenario
```
Current:  0.4680 (Challenge 1)
All methods combined perfectly:
Final:    0.2800 (-40% total)

Overall:  0.1970 → 0.1405 (-29% overall)
```

---

## 💡 RECOMMENDATIONS

### If You Have 1-2 Days:
1. ⭐ Test-time augmentation (highest ROI)
2. ⭐ Subject metadata features (easy win)
3. ⭐ Weighted ensemble (models already trained!)

**Expected:** 0.47 → 0.40 (-15%)

### If You Have 3-5 Days:
1. All above +
2. ⭐ Frequency domain features
3. ⭐ Advanced augmentation
4. ⭐ Attention mechanisms

**Expected:** 0.47 → 0.34 (-28%)

### If You Have 7-14 Days:
1. All above +
2. Multi-task learning
3. Pre-training experiments
4. Hyperparameter optimization
5. Final ensemble of all approaches

**Expected:** 0.47 → 0.28-0.32 (-30-40%)

---

## 🔬 VALIDATION STRATEGY

### Before Implementing Any Improvement:
1. ✅ Establish baseline on validation set
2. ✅ Implement improvement
3. ✅ Test on same validation set
4. ✅ If improvement > 3%, keep it
5. ✅ If improvement < 3%, analyze why
6. ✅ Combine improvements incrementally

### Testing Protocol:
```python
# Always use same validation set
np.random.seed(42)
train_idx, val_idx = train_test_split(...)

# Test each improvement
baseline_nrmse = test_model(baseline_model, X_val, y_val)
improved_nrmse = test_model(improved_model, X_val, y_val)

improvement_pct = (baseline_nrmse - improved_nrmse) / baseline_nrmse * 100
print(f"Improvement: {improvement_pct:.1f}%")

if improvement_pct > 3:
    print("✅ Keep improvement!")
else:
    print("❌ Revert or debug")
```

---

## 🎯 PRIORITY MATRIX

| Improvement | Expected Gain | Time | Risk | Priority |
|-------------|---------------|------|------|----------|
| Test-Time Aug | 5-10% | 2-3h | LOW | 🔴 HIGHEST |
| Subject Meta | 5-10% | 3-4h | LOW | 🔴 HIGHEST |
| Weighted Ensemble | 5-8% | 4-6h | MED | 🟠 HIGH |
| Frequency Features | 10-20% | 6-8h | MED | 🟠 HIGH |
| Advanced Aug | 3-8% | 4-6h | LOW | 🟡 MEDIUM |
| Attention | 8-15% | 6-8h | MED | 🟡 MEDIUM |
| Multi-Task | 5-12% | 8-10h | MED | 🟡 MEDIUM |
| Pre-training | 5-15% | 6-10h | MED | 🟢 LOW |
| Transformer | -10-15% | 12-16h | HIGH | 🟢 LOW |
| Source Local | 0-20% | 10-15h | HIGH | ⚪ SKIP |

---

## 🚀 START HERE

### Immediate Next Script to Write:

**File:** `scripts/test_time_augmentation.py`

```python
#!/usr/bin/env python3
"""
Test-Time Augmentation for Challenge 1
Predict with multiple augmented versions and average
"""

import torch
import numpy as np
from train_challenge1_improved import ImprovedResponseTimeCNN, AugmentedResponseTimeDataset

def predict_with_tta(model, data, n_augmentations=5, noise_std=0.02):
    """Test-time augmentation"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # Original
        pred = model(data)
        predictions.append(pred)
        
        # Augmented versions
        for _ in range(n_augmentations - 1):
            # Add noise
            noise = torch.randn_like(data) * noise_std
            aug_data = data + noise
            
            pred = model(aug_data)
            predictions.append(pred)
    
    # Average
    return torch.stack(predictions).mean(dim=0)

# Test and compare
if __name__ == "__main__":
    # Load model
    model = ImprovedResponseTimeCNN()
    model.load_state_dict(torch.load('checkpoints/response_time_model.pth'))
    
    # Load validation data
    dataset = AugmentedResponseTimeDataset(...)
    
    # Compare baseline vs TTA
    baseline_preds = []
    tta_preds = []
    
    for data, label in dataset:
        # Baseline
        baseline_pred = model(data.unsqueeze(0))
        baseline_preds.append(baseline_pred.item())
        
        # TTA
        tta_pred = predict_with_tta(model, data.unsqueeze(0), n_augmentations=5)
        tta_preds.append(tta_pred.item())
    
    # Compare NRMSE
    baseline_nrmse = compute_nrmse(labels, baseline_preds)
    tta_nrmse = compute_nrmse(labels, tta_preds)
    
    print(f"Baseline NRMSE: {baseline_nrmse:.4f}")
    print(f"TTA NRMSE: {tta_nrmse:.4f}")
    print(f"Improvement: {(baseline_nrmse - tta_nrmse)/baseline_nrmse*100:.1f}%")
```

---

## 📚 Resources for Implementation

- **Test-Time Aug:** https://arxiv.org/abs/1904.12848
- **Attention:** https://arxiv.org/abs/1706.03762
- **Frequency Features:** https://mne.tools/stable/auto_tutorials/time-freq/20_sensors_time_frequency.html
- **Multi-Task:** https://arxiv.org/abs/1706.05098
- **Mixup:** https://arxiv.org/abs/1710.09412

---

**Ready to start improving? Begin with test-time augmentation! 🚀**
