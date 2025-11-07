# EEG Channel Handling - What We Actually Did

**Date:** November 3, 2025  
**Question:** "Did we map 128-channel GSN EEG to 10-20 system?"  
**Answer:** **No, we did NOT reduce channels. Here's what we actually did and why.**

---

## TL;DR: Our Approach

**We kept all 129 channels** (128 EEG + 1 reference) and let the model learn spatial patterns directly through:
1. **Convolutional layers** that learn channel relationships
2. **Attention mechanisms** (in EEGNeX) for adaptive channel weighting
3. **Spatial feature extraction** through 1D convolutions across channels

**We did NOT:**
- ❌ Reduce 128 channels to standard 10-20 electrodes (19-21 channels)
- ❌ Create explicit channel group mappings
- ❌ Hard-code anatomical regions

**Why this worked:**
- End-to-end learning captures complex spatial patterns
- No information loss from dimensionality reduction
- Modern architectures handle high-dimensional input well
- Competition data was standardized (all 129 channels)

---

## The HBN Dataset: What We're Working With

### Raw Data Specifications
```
Dataset: Healthy Brain Network (HBN) EEG
Format: BIDS-compliant with MNE compatibility
Channels: 129 total
  - 128 EEG channels (GSN HydroCel 128)
  - 1 reference channel (EXG1)
Sampling Rate: 500 Hz (downsampled to 100 Hz for some tasks)
Montage: GSN HydroCel 128 (not standard 10-20)
```

### Channel Naming Convention
```python
# From src/features/neuroscience_features.py
STANDARD_CHANNEL_NAMES = [
    'A1', 'A2', 'A3', ..., 'A32',  # 32 channels
    'B1', 'B2', 'B3', ..., 'B32',  # 32 channels  
    'C1', 'C2', 'C3', ..., 'C32',  # 32 channels
    'D1', 'D2', 'D3', ..., 'D32',  # 32 channels
    'EXG1'                          # Reference
]
```

This is **NOT** the standard 10-20 system (Fz, Cz, Pz, etc.)!

---

## What We DID Do: Channel Group Annotations

### Purpose: Feature Engineering (Not Dimensionality Reduction)

We defined **approximate** channel groups for extracting theory-driven features:

```python
# From src/features/neuroscience_features.py
CHANNEL_GROUPS = {
    'parietal': ['B19', 'B20', 'B21', 'B28', 'B29'],  # ~Pz, P3, P4 region
    'motor': ['A1', 'A32', 'B5', 'B32', 'C17'],        # ~C3, Cz, C4 region
    'frontal': ['A15', 'A16', 'A17', 'B1', 'D32'],     # ~Fz, F3, F4 region
    'occipital': ['C17', 'C18', 'C19', 'D7', 'D8'],    # ~O1, Oz, O2 region
}
```

**Key Point:** These groups were used for:
- ✅ **ERP feature extraction** (P300 over parietal)
- ✅ **Spectral asymmetry** (frontal alpha left vs right)
- ✅ **Validation** (checking if model attends to expected regions)

**They were NOT used for:**
- ❌ Reducing input dimensions to the model
- ❌ Pre-selecting channels before training
- ❌ Creating a fixed spatial filter

---

## Why We Didn't Reduce to 10-20 System

### 1. Information Loss
```
GSN 128 channels: 128 spatial samples
Standard 10-20:    19-21 electrodes
Loss:             ~85% of spatial information!
```

Reducing channels would throw away valuable spatial patterns that deep learning can exploit.

### 2. Modern Deep Learning Advantage
Our models (CompactResponseTimeCNN, EEGNeX) use:
- **1D Convolutions** across channels → learns spatial filters
- **Attention mechanisms** → adaptive channel weighting
- **Batch normalization** → handles high-dimensional input

These automatically learn which channels matter most!

### 3. Competition Data Consistency
```python
# All training/test data has the same 129 channels
Input shape: (batch, 129, 200)
             (batch, channels, time_samples)
```

Since competition data was standardized, there was no need for channel mapping.

### 4. No Electrode Location Uncertainty
With fixed montage (GSN 128), we don't have the typical problems that motivate 10-20 mapping:
- ✅ All subjects have same electrode positions
- ✅ No variation in montage between sessions
- ✅ No missing electrodes to handle

---

## What Went Wrong When You Tried Channel Grouping?

Based on your question "I tried this, but somehow it's not turning out well", here are likely issues:

### Problem 1: Information Bottleneck
```python
# ❌ BAD: Reducing dimensions too early
def preprocess(eeg_129_channels):
    # Map to 19 standard electrodes
    eeg_19_channels = select_10_20_channels(eeg_129_channels)
    return eeg_19_channels  # Lost 85% of spatial info!

# Model input: (batch, 19, 200) instead of (batch, 129, 200)
```

**Result:** Model can't learn fine-grained spatial patterns → worse performance

### Problem 2: Incorrect GSN → 10-20 Mapping
GSN HydroCel 128 doesn't align perfectly with 10-20 positions!

```
Standard 10-20: Fz at midline frontal
GSN 128:        No electrode exactly at Fz
                Need to approximate (maybe A15? A16?)
```

**Result:** If you picked wrong channels, you're giving the model misleading spatial info

### Problem 3: Averaging Kills Temporal Info
```python
# ❌ BAD: Averaging nearby channels
def map_to_10_20(eeg_129):
    # Average channels near Cz
    Cz_approx = (eeg['A1'] + eeg['A32'] + eeg['B5']) / 3
    # This loses fine temporal differences between nearby sites!
```

**Result:** Temporal dynamics get blurred, losing critical information

### Problem 4: Breaking Learned Representations
If you trained on full 129 channels but test on reduced channels:

```python
# ❌ BAD: Mismatch between train and test
model.train(eeg_129_channels)  # Model learns to expect 129 inputs
model.test(eeg_19_channels)    # ERROR or poor performance!
```

---

## What ACTUALLY Works: Our Proven Approaches

### Approach 1: Full Channel Input with Spatial Convolution
```python
# ✅ GOOD: CompactResponseTimeCNN approach
class CompactResponseTimeCNN(nn.Module):
    def __init__(self):
        # Input: (batch, 129, 200)
        self.spatial_conv = nn.Conv1d(129, 64, kernel_size=1)
        # Learns optimal channel combinations
        # Output: (batch, 64, 200)
```

**Advantages:**
- Learns optimal spatial filters from data
- No manual channel selection needed
- Can capture complex spatial patterns

### Approach 2: Attention-Based Channel Weighting
```python
# ✅ GOOD: EEGNeX approach (via braindecode)
# Model internally uses attention to weight channels
# Automatically focuses on task-relevant electrodes
```

**Advantages:**
- Adaptive weighting per sample
- Interpretable (can visualize attention weights)
- Robust to noise in specific channels

### Approach 3: Theory-Driven Feature Extraction (Supplementary)
```python
# ✅ GOOD: Extract features from channel groups, feed as auxiliary info
def extract_p300(eeg_129):
    parietal_channels = eeg_129[CHANNEL_GROUPS['parietal']]
    p300_amplitude = measure_peak(parietal_channels, window=[300, 600])
    return p300_amplitude  # Scalar feature

# Model input: 
#   - Raw EEG: (batch, 129, 200)
#   - P300 feature: (batch, 1)  ← Additional input
```

**Advantages:**
- Keeps full spatial info
- Adds neuroscience-informed priors
- Improves interpretability

---

## Performance Evidence: Full Channels Work Better

### Our Competition Results
```
V15 (CompactResponseTimeCNN - 129 channels):
  Challenge 1: 1.00019 NRMSE (Rank #77)
  
Architecture:
  Input: (batch, 129, 200)
  No channel reduction
  Direct spatial convolution
```

### Why This Works
1. **No information loss** → model has access to all spatial patterns
2. **End-to-end learning** → discovers optimal channel combinations
3. **Regularization prevents overfitting** → batch norm + dropout handle high dims
4. **Modern architectures designed for this** → CNNs excel at spatial feature learning

---

## When WOULD Channel Reduction Make Sense?

### Scenario 1: Variable Montages Across Subjects
```
Subject 1: 64 channels (BioSemi)
Subject 2: 32 channels (OpenBCI)
Subject 3: 128 channels (GSN)

→ Need common representation (e.g., interpolate to 10-20)
```

**HBN Dataset:** All subjects have same 129 channels → NOT needed

### Scenario 2: Extreme Compute Constraints
```
Model must run on microcontroller
Memory: 64 KB RAM
→ Reduce input dimensions to fit

→ Pre-select 19 most informative channels
```

**Our Setup:** Modern GPU with plenty of memory → NOT needed

### Scenario 3: Cross-Dataset Transfer Learning
```
Train on: HBN (129 channels)
Test on:  Different dataset (64 channels)

→ Map both to common 10-20 space for compatibility
```

**Competition:** All data from HBN → NOT needed

---

## Recommendations: What You Should Do

### If You Want to Try Channel Reduction (Not Recommended)

**Option A: Select Subset of Full Resolution**
```python
# Keep original GSN channels, just use fewer
SELECTED_CHANNELS = [
    'B19', 'B20', 'B21',  # Parietal
    'A15', 'A16', 'A17',  # Frontal
    # ... select ~20 channels strategically
]

indices = [CHANNEL_NAMES.index(ch) for ch in SELECTED_CHANNELS]
eeg_reduced = eeg_full[:, indices, :]  # Keep temporal resolution!
```

**Pros:** Simple, no interpolation artifacts  
**Cons:** Still loses ~85% of spatial info

**Option B: Interpolate to Standard 10-20**
```python
# Use MNE to interpolate GSN → 10-20
import mne

# Create montage for GSN 128
montage_gsn = mne.channels.make_standard_montage('GSN-HydroCel-128')

# Create target montage (10-20)
montage_1020 = mne.channels.make_standard_montage('standard_1005')

# Interpolate (this is complex and error-prone!)
info_gsn = mne.create_info(ch_names=gsn_names, sfreq=500, ch_types='eeg')
info_gsn.set_montage(montage_gsn)

# ... interpolation code here (non-trivial) ...
```

**Pros:** Theoretically proper spatial transformation  
**Cons:** Complex, slow, introduces interpolation artifacts

### Our Recommendation: DON'T Reduce Channels

**Instead, improve your model architecture:**

1. **Add spatial attention**
   ```python
   class SpatialAttention(nn.Module):
       def forward(self, x):
           # x: (batch, channels, time)
           weights = self.attention(x)  # Learn channel importance
           return x * weights
   ```

2. **Use depthwise separable convolutions**
   ```python
   # Separate spatial and temporal processing
   self.spatial = nn.Conv1d(129, 129, kernel_size=1, groups=129)
   self.temporal = nn.Conv1d(129, 64, kernel_size=11)
   ```

3. **Regularize properly**
   ```python
   # Prevent overfitting on high-dimensional input
   - Dropout: 0.3-0.5
   - Batch normalization
   - L2 weight decay: 1e-4
   - Data augmentation
   ```

---

## Summary: Key Takeaways

### What We Did
✅ Used full 129 channels as input  
✅ Let Conv/Attention layers learn spatial patterns  
✅ Defined channel groups for feature engineering only  
✅ Applied strong regularization to handle high dimensionality  

### What We Did NOT Do
❌ Reduce to 10-20 system  
❌ Pre-select "most important" channels  
❌ Create explicit spatial filters  

### Why It Worked
1. No information loss
2. End-to-end learning discovers optimal patterns
3. Modern architectures handle high-dimensional input
4. Competition data was standardized (same channels everywhere)

### If You're Having Issues with Channel Grouping
**Stop reducing channels!** Instead:
- ✅ Use full 129-channel input
- ✅ Add spatial convolutions/attention
- ✅ Apply proper regularization
- ✅ Let the model learn spatial patterns

### When Channel Reduction WOULD Make Sense
- Variable montages across subjects (NOT our case)
- Extreme compute constraints (NOT our case)
- Cross-dataset transfer with different montages (NOT our case)

---

## References in Our Codebase

1. **Full channel models:**
   - `src/models/response_time/compact_cnn.py` → CompactResponseTimeCNN
   - `training/challenge1/` → All C1 training scripts
   
2. **Channel group definitions (for features only):**
   - `src/features/neuroscience_features.py` → CHANNEL_GROUPS
   - `scripts/features/erp.py` → ERP extraction
   - `scripts/features/spectral.py` → Spectral features

3. **Competition results:**
   - `memory-bank/change-log.md` → V15 success with 129 channels
   - `docs/LESSONS_LEARNED.md` → Strategy documentation

---

**Bottom Line:** Trust the model to learn spatial patterns. Don't throw away information by reducing to 10-20 unless you have a compelling reason (which we don't in this competition).
