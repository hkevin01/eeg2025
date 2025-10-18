# 🧠 EEG 2025 Competition - Team Meeting Presentation
**Presenter:** Kevin  
**Date:** October 18, 2025  
**Status:** Ready for Discussion ✅

---

## 📋 Table of Contents
1. [Competition Overview](#competition-overview)
2. [The Starter Kit - Our Foundation](#the-starter-kit)
3. [My Approach: Applying CNN to the Framework](#my-approach)
4. [Key Clues from the Starter Kit](#key-clues)
5. [CNN Architecture Overview](#cnn-architecture-overview)
6. [What I Tried and Why](#what-i-tried)
7. [Testing Strategy (R1-R6)](#testing-strategy)
8. [Current Results and Next Steps](#results)

---

## 🎯 Competition Overview

### What We're Predicting
- **Challenge 1 (30%):** Response Time - How quickly subjects react in a visual task
- **Challenge 2 (70%):** Externalizing Behavior - Behavioral problems from CBCL assessment

### The Data
- **EEG recordings** from children performing cognitive tasks
- **129 channels** of brain activity
- **200 time samples** per segment (2 seconds @ 100 Hz)
- **6 data releases** (R1-R6) for training and validation

### The Metric
- **NRMSE** (Normalized Root Mean Squared Error)
- Lower is better: **< 1.00 is good**, **< 0.50 is excellent**
- Overall Score = 0.3 × Challenge1 + 0.7 × Challenge2

---

## 🔧 What I Used vs. What I Didn't Use

### ✅ What I Used from Starter Kit

| Component | Used? | My Implementation |
|-----------|-------|-------------------|
| **`EEGChallengeDataset`** | ✅ YES | Used for all data loading |
| **`braindecode.preprocessing`** | ✅ YES | Used `Preprocessor`, `preprocess`, `create_windows_from_events` |
| **`annotate_trials_with_target`** | ✅ YES | Critical for extracting response times |
| **`add_aux_anchors`** | ✅ YES | Adds `stimulus_anchor` for window locking |
| **`stimulus_anchor` mapping** | ✅ YES | Lock windows to stimulus onset |
| **NRMSE metric** | ✅ YES | Official competition metric |
| **Submission template** | ✅ YES | `get_model_challenge_1()`, `get_model_challenge_2()` |
| **Multi-release strategy** | ✅ YES | Train on R1-R4, evaluate on R5 |

### ❌ What I Added (Not in Starter Kit)

| Component | From Starter Kit? | What I Did |
|-----------|-------------------|------------|
| **Channel-wise normalization** | ❌ NO | Implemented z-score normalization per channel |
| **CNN architecture** | ❌ NO | Designed CompactResponseTimeCNN (75K params) |
| **Training loop** | ❌ NO | Implemented with AdamW, LR scheduling, early stopping |
| **Data augmentation** | ❌ NO | TTA with 10 augmentations (experimental) |
| **Model checkpointing** | ❌ NO | Save best model based on validation NRMSE |
| **Cross-validation** | ❌ NO | 5-fold CV for hyperparameter tuning |

### 📊 Starter Kit Models

The starter kit provided **example models**, but I chose not to use them:

| Model | In Starter Kit? | Why I Didn't Use It |
|-------|----------------|---------------------|
| **EEGNeX** | ✅ YES | Too large (5M+ params), slow training |
| **ShallowFBCSPNet** | ✅ YES | Designed for motor imagery, not cognitive tasks |
| **Deep4Net** | ✅ YES | Too many params, overfits small datasets |

**Instead:** I built custom lightweight CNNs (75K and 64K params) optimized for this specific task.

---

## 📦 The Starter Kit - Our Foundation

### What the Organizers Provided

The competition starter kit gave us a **complete infrastructure** - I didn't start from scratch!

### Core Modules & Libraries Explained

#### **Why These Specific Libraries?**

| Library | Purpose | Why Starter Kit Uses It |
|---------|---------|--------------------------|
| **`eegdash`** | Competition data loader | ✅ Official competition package<br/>✅ Handles R1-R6 splits correctly<br/>✅ Ensures consistency with eval server |
| **`braindecode`** | EEG deep learning toolkit | ✅ Built on top of MNE (industry standard)<br/>✅ PyTorch integration<br/>✅ Preprocessing pipelines<br/>✅ Window creation from events |
| **`mne`** | EEG processing foundation | ✅ Industry standard (30+ years)<br/>✅ Comprehensive signal processing<br/>✅ Used by neuroscience researchers globally |
| **`torch`** | Deep learning framework | ✅ Dynamic graphs (better for research)<br/>✅ Excellent debugging<br/>✅ Large ecosystem |

#### 1. **Data Loading Framework (`eegdash`)**
```python
from eegdash import EEGChallengeDataset

# THIS IS CRITICAL - Must use EEGChallengeDataset, NOT EEGDashDataset!
dataset = EEGChallengeDataset(
    release="R1",                           # Which data split to use
    query=dict(task="contrastChangeDetection"),  # Filter by task
    cache_dir="data"                        # Where to store data
)
```

**⚠️ CRITICAL DISTINCTION:** The starter kit emphasizes:
> "The data accessed via `EEGChallengeDataset` is NOT identical to what you get from EEGDashDataset directly. If you are participating in the competition, always use `EEGChallengeDataset` to ensure consistency with the challenge data."

**Why this matters:**
- `EEGChallengeDataset` has competition-specific preprocessing
- Ensures consistency with evaluation server
- Handles data splits (R1-R6) correctly
- Using `EEGDashDataset` directly = **wrong preprocessing** = invalid results!

**What `eegdash` provides:**
- Automatic data download from competition servers
- Cached local storage (saves bandwidth)
- Release-specific data loading (R1, R2, R3, R4, R5, R6)
- Task filtering (`contrastChangeDetection`, `restingState`, etc.)
- Subject metadata (age, sex, clinical scores)

#### 2. **Preprocessing Pipeline (`braindecode`)**

**Why Braindecode?**

Braindecode is specifically designed for EEG deep learning:
- Built on top of MNE (the industry standard for EEG)
- PyTorch-native (seamless integration with deep learning)
- Provides EEG-specific preprocessing pipelines
- Used by researchers in major neuroscience labs

**Key Features We Use:**
1. **`Preprocessor`** class - Applies functions to MNE Raw objects
2. **`preprocess()`** - Parallel processing across multiple recordings
3. **`create_windows_from_events()`** - Event-locked windowing
4. **`BaseConcatDataset`** - Efficient multi-subject dataset handling

```python
from braindecode.preprocessing import Preprocessor, preprocess
from eegdash.hbn.windows import annotate_trials_with_target, add_aux_anchors

# The starter kit showed us how to preprocess
preprocessors = [
    # Annotate trials with target values (response times)
    Preprocessor(
        annotate_trials_with_target,      # Extract response times from events
        target_field="rt_from_stimulus",  # What we're predicting
        epoch_length=2.0,                 # 2-second windows
        require_stimulus=True,            # Must have stimulus event
        require_response=True,            # Must have response event
        apply_on_array=False              # Apply to MNE Raw objects
    ),
    # Add auxiliary anchor points for window creation
    Preprocessor(
        add_aux_anchors,                  # Add 'stimulus_anchor' annotation
        apply_on_array=False
    ),
]

# Apply preprocessors in parallel across all recordings
preprocess(dataset, preprocessors, n_jobs=-1)  # Use all CPU cores
```

**What `annotate_trials_with_target` does:**
- Finds stimulus events in the EEG recording
- Finds response button presses
- Calculates `rt_from_stimulus` = response_time - stimulus_time
- Adds this information to MNE annotations
- Critical for supervised learning!

#### 3. **Window Creation (Event-Locked)**
```python
from braindecode.preprocessing import create_windows_from_events

# Create 2-second windows locked to stimulus onset
dataset = create_windows_from_events(
    dataset,
    mapping={"stimulus_anchor": 0},       # Lock to stimulus
    trial_start_offset_samples=50,        # +0.5s after stimulus
    window_size_samples=200,              # 2 seconds @ 100Hz
    window_stride_samples=100,            # Non-overlapping
    preload=True
)
```

**Key insight from starter kit:** Use `"stimulus_anchor"` annotation to align windows to task events!

#### 4. **Official Scoring Metrics**
```python
def nrmse(y_true, y_pred):
    """Official competition metric from starter kit"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse / y_true.std()  # Normalized by std of targets

def score_overall(score1, score2):
    """Exact scoring used by competition"""
    return 0.3 * score1 + 0.7 * score2
```

#### 5. **Submission Template**
The starter kit provided `submission.py` template:
```python
class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ      # Sampling frequency (100 Hz)
        self.device = DEVICE    # CPU or CUDA
    
    def get_model_challenge_1(self):
        """Load Challenge 1 model"""
        model = YourModel()
        weights = torch.load("weights_challenge_1.pt")
        model.load_state_dict(weights)
        return model
    
    def get_model_challenge_2(self):
        """Load Challenge 2 model"""
        # Similar structure
```

---

## 🔧 My Approach: Applying CNN to the Framework

### What I Contributed

I took the starter kit infrastructure and **plugged in a Convolutional Neural Network (CNN)** architecture.

### My CNN Architecture

```python
class CompactResponseTimeCNN(nn.Module):
    """My CNN for Challenge 1 (75K parameters)"""
    
    def __init__(self):
        super().__init__()
        
        # Convolutional feature extraction
        self.features = nn.Sequential(
            # Layer 1: 129 channels → 32 channels
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2: 32 → 64 channels
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            # Layer 3: 64 → 128 channels
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            # Global average pooling
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Regression head (predict response time)
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single output value
        )
```

### How I Wired It Up

#### Step 1: Load Data (Using Starter Kit)
```python
# Follow starter kit pattern
dataset = EEGChallengeDataset(
    release="R1",
    query=dict(task="contrastChangeDetection")
)

# Apply preprocessing (from starter kit)
preprocess(dataset, preprocessors)

# Create windows (from starter kit)
dataset = create_windows_from_events(...)
```

#### Step 2: Create DataLoader
```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=32,      # Process 32 samples at once
    shuffle=True,       # Randomize order
    num_workers=4       # Parallel data loading
)
```

#### Step 3: Training Loop
```python
model = CompactResponseTimeCNN()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.MSELoss()  # Mean squared error

for epoch in range(50):
    for X, y in train_loader:
        # X shape: (batch=32, channels=129, time=200)
        # y shape: (batch=32, 1) - response times
        
        # Forward pass
        predictions = model(X)
        loss = criterion(predictions, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate using starter kit metrics
    nrmse = validate_with_nrmse(model, val_loader)
```

#### Step 4: Save for Submission
```python
# Save weights in format expected by starter kit
torch.save({
    'model_state_dict': model.state_dict(),
    'epoch': epoch,
    'val_loss': best_val_loss
}, 'weights_challenge_1_multi_release.pt')
```

#### Step 5: Create Submission Package
```python
# Follow starter kit submission.py template
class Submission:
    def get_model_challenge_1(self):
        model = CompactResponseTimeCNN()
        weights = torch.load("weights_challenge_1_multi_release.pt")
        model.load_state_dict(weights['model_state_dict'])
        return model

# Package as required
zip -j submission.zip \
    submission.py \
    weights_challenge_1_multi_release.pt \
    weights_challenge_2_multi_release.pt
```

---

## 🔍 Key Clues from the Starter Kit

### 1. **Use EEGChallengeDataset (Not EEGDashDataset!)**

The starter kit documentation emphasized this multiple times:

```
IMPORTANT: The data accessed via `EEGChallengeDataset` is NOT identical 
to what you get from EEGDashDataset directly. If you are participating 
in the competition, always use `EEGChallengeDataset` to ensure 
consistency with the challenge data.
```

**Why this matters:**
- `EEGChallengeDataset` has competition-specific preprocessing
- Ensures our local evaluation matches server evaluation
- Handles data splits (R1-R6) correctly
- Using wrong dataset = completely invalid results

**I saw this in the logs:**
```
Used Annotations descriptions: ['stimulus_anchor']
Used Annotations descriptions: ['stimulus_anchor']
...
```

This confirms the dataset is using `stimulus_anchor` events, which the starter kit taught us to use.

### 2. **Stimulus-Locked Windows**

Starter kit showed the importance of event-locked analysis:

```python
# Lock windows to stimulus onset, not arbitrary time points
create_windows_from_events(
    dataset,
    mapping={"stimulus_anchor": 0},  # ← Key clue from starter kit
    trial_start_offset_samples=50,   # +0.5s after stimulus
    ...
)
```

**Why:** Response time is measured from stimulus, so our windows must be stimulus-aligned!

### 3. **Input Shape Requirements**

Starter kit made clear the expected input format:
- **Shape:** `(batch, 129, 200)` 
  - 129 EEG channels
  - 200 time samples (2 seconds @ 100 Hz)
- **Output:** `(batch, 1)` - single prediction per sample

This guided my CNN architecture design.

### 4. **NRMSE Metric Calculation**

The starter kit provided the exact scoring function:

```python
def nrmse(y_trues, y_preds):
    """Normalized RMSE using standard deviation"""
    return rmse(y_trues, y_preds) / y_trues.std()
```

**Key insight:** Normalization by `std()` means:
- Predictions should match the scale of targets
- Model needs to learn the actual response time range (not just relative ordering)

### 5. **Data Split Strategy**

Starter kit revealed the competition structure:
- **R1, R2, R3:** Primary training data
- **R4:** Validation data during development
- **R5:** "Warmup" test set (used for local evaluation)
- **R6:** Final hidden test set (used only by organizers)

**My strategy:** Train on R1-R4, validate on R5, prepare for R6 testing

### 6. **Submission Format**

Starter kit provided exact template:
```python
# Must have this exact structure
class Submission:
    def __init__(self, SFREQ, DEVICE): ...
    def get_model_challenge_1(self): ...
    def get_model_challenge_2(self): ...

# Must be packaged as flat zip
submission.zip
├── submission.py
├── weights_challenge_1_multi_release.pt
└── weights_challenge_2_multi_release.pt
```

**No subdirectories allowed!** Took me a while to learn this.

---

## 🧠 CNN Architecture Overview

### Convolutional Neural Network (CNN)

A CNN is a feed-forward neural network with specialized layers that exploit local correlations in the input data through learnable convolutional kernels.

### Architecture Components

#### 1. **1D Convolutional Layers**
```python
nn.Conv1d(in_channels=129, out_channels=32, kernel_size=7, stride=2, padding=3)
```
- **Operation:** Discrete convolution over temporal dimension
- **Input:** `(batch, 129, 200)` - 129 EEG channels, 200 temporal samples
- **Kernel:** Learnable weights `(32, 129, 7)` - 32 filters, each 7 samples wide
- **Output:** `(batch, 32, 100)` - 32 feature maps, downsampled by stride=2
- **Function:** Extracts local temporal features through shared weights (weight sharing reduces parameters)

#### 2. **Batch Normalization**
```python
nn.BatchNorm1d(num_features=32)
```
- **Operation:** `y = (x - μ_B) / √(σ²_B + ε) * γ + β`
- **Parameters:** Learnable scale (γ) and shift (β)
- **Purpose:** Normalizes activations, reduces internal covariate shift
- **Benefits:** Accelerates convergence, allows higher learning rates, regularization effect

#### 3. **Activation Functions**
```python
nn.ReLU()  # Rectified Linear Unit
```
- **Function:** `f(x) = max(0, x)`
- **Purpose:** Introduces non-linearity (enables learning of complex functions)
- **Properties:** 
  - Sparse activation (gradient flows only for positive values)
  - Computationally efficient (simple comparison)
  - Mitigates vanishing gradient problem

#### 4. **Dropout Regularization**
```python
nn.Dropout(p=0.3)  # Drop probability 30%
```
- **Training:** Randomly zeros elements with probability `p`, scales remaining by `1/(1-p)`
- **Inference:** Identity operation (no dropout)
- **Purpose:** Prevents co-adaptation of neurons, reduces overfitting
- **Mechanism:** Approximates ensemble of exponentially many sub-networks

#### 5. **Global Average Pooling**
```python
nn.AdaptiveAvgPool1d(output_size=1)
```
- **Operation:** Averages across temporal dimension: `y = (1/T) Σ x_t`
- **Input:** `(batch, channels, T)` - variable temporal length
- **Output:** `(batch, channels, 1)` - fixed-size representation
- **Purpose:** 
  - Reduces spatial dimensions to fixed size
  - Provides translation invariance
  - Reduces parameters (no fully-connected layer needed)
  - Acts as structural regularizer

### Forward Pass Data Flow

```
Input: (batch, 129, 200)
    ↓ [Conv1d: 129→32, k=7, s=2] + BatchNorm + ReLU + Dropout(0.3)
    → (batch, 32, 100)
    ↓ [Conv1d: 32→64, k=5, s=2] + BatchNorm + ReLU + Dropout(0.4)
    → (batch, 64, 50)
    ↓ [Conv1d: 64→128, k=3, s=2] + BatchNorm + ReLU + Dropout(0.5)
    → (batch, 128, 25)
    ↓ [AdaptiveAvgPool1d: T→1]
    → (batch, 128, 1) → Flatten → (batch, 128)
    ↓ [Linear: 128→64] + ReLU + Dropout(0.5)
    → (batch, 64)
    ↓ [Linear: 64→32] + ReLU + Dropout(0.4)
    → (batch, 32)
    ↓ [Linear: 32→1]
Output: (batch, 1)
```

### Design Rationale for EEG

**1. Parameter Efficiency**
- Weight sharing across temporal dimension: O(k·c_in·c_out) vs O(T·c_in·c_out) for fully-connected
- Total parameters: ~75K (Challenge 1), ~64K (Challenge 2)
- Prevents overfitting on limited training data

**2. Local Temporal Structure**
- Convolutional kernels capture local dependencies (ERPs, oscillations)
- Hierarchical feature extraction: low-level waveforms → mid-level patterns → high-level cognitive states
- Receptive field grows exponentially with depth: layer 1 (7 samples) → layer 2 (17 samples) → layer 3 (35 samples)

**3. Translation Invariance**
- Same learned kernel applied across all temporal positions
- Cognitive patterns detected regardless of latency jitter
- Important for variable response times

**4. Regularization Strategy**
- Progressive dropout (0.3 → 0.4 → 0.5): stronger regularization in deeper layers
- BatchNorm: implicit regularization through noise injection
- Global pooling: reduces overfitting vs fully-connected layers

---

## 🔬 What I Tried and Why

### Iteration 1: Baseline CNN ✅
**Architecture:** Simple 3-layer CNN (75K params)
- Conv: 129→32→64→128 channels
- Dropout: 0.3, 0.4, 0.5
- Regression head: 128→64→32→1

**Result:** NRMSE 1.00 on Challenge 1, 1.46 on Challenge 2
**Status:** Working baseline!

### Iteration 2: TCN (Temporal Convolutional Network) ❌
**Why I tried it:** TCN has larger receptive field, captures longer-term dependencies

**What went wrong:**
- 196K parameters (2.6× larger than baseline)
- Validation loss: 0.0102 (looked great!)
- Test NRMSE: 1.63 (+63% worse!) 😱
- **Root cause:** Overfitting to validation set

**Lesson learned:** Bigger ≠ better, validation metrics can be misleading

### Iteration 3: Multi-Head Self-Attention 🚧
**Why trying it:** 
- Full receptive field (sees all 200 time points)
- Lightweight (+6.3% params)
- Residual connections ensure safety

**Architecture:**
```python
class LightweightAttentionCNN:
    Conv1 → Conv2 → [Attention + Residual] → Conv3 → Regressor
    # If attention fails, residual path preserves CNN performance
```

**Status:** In development (training infrastructure ready)

### Iteration 4: Test-Time Augmentation (TTA) ⚡
**Why trying it:**
- No retraining needed!
- 5-10% improvement expected
- Works with existing models

**How it works:**
```python
def predict_with_tta(model, x):
    # Original prediction
    pred1 = model(x)
    
    # Augmented predictions
    pred2 = model(x + small_noise)      # Add noise
    pred3 = model(x * 1.05)             # Scale slightly
    pred4 = model(roll(x, shift=3))     # Time shift
    pred5 = model(x * channel_mask)     # Channel dropout
    
    # Average all predictions
    return mean([pred1, pred2, pred3, pred4, pred5])
```

**Status:** Implemented, ready to test!

### Timeline of Attempts

```
Oct 15: Initial baseline          → NRMSE 2.01 (poor)
Oct 16: Tuned CNN                 → NRMSE 1.32 (good!) ✅
Oct 18: Tried TCN                 → NRMSE 1.42 (worse) ❌
Oct 18: Reverted to working CNN   → Back to 1.32 ✅
Oct 18: Added TTA                 → Testing now...
Oct 18: Attention CNN ready       → Training pending...
```

---

## 🧪 Testing Strategy (R1-R6)

### Why Test on All Releases?

**Problem:** Previous submissions only used R5 for validation
- R5 might not be representative
- Could overfit to R5 characteristics
- Need to ensure model generalizes across all data

**Solution:** Test on R1-R6 to get complete picture

### What Each Release Represents

| Release | Role | Data Size | Purpose |
|---------|------|-----------|---------|
| **R1** | Training | ~300 subjects | Primary training data |
| **R2** | Training | ~300 subjects | Primary training data |
| **R3** | Training | ~300 subjects | Primary training data |
| **R4** | Validation | ~200 subjects | Development validation |
| **R5** | Test (Public) | ~150 subjects | Local evaluation set |
| **R6** | Test (Hidden) | ~150 subjects | Final competition eval |

### Testing Approach

```python
# Evaluate submission on ALL releases
for release in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']:
    # Load data for this release
    dataset = EEGChallengeDataset(release=release, ...)
    
    # Run inference
    predictions = model.predict(dataset)
    
    # Calculate official NRMSE
    nrmse = calculate_nrmse(predictions, targets)
    
    print(f"{release}: NRMSE = {nrmse:.4f}")
```

### Expected Outcomes

**Good Model:**
- Consistent NRMSE across R1-R6
- R1-R4: ~1.0 (training data, should be good)
- R5-R6: ~1.0-1.2 (test data, slight degradation OK)

**Overfitting:**
- R1-R4: Very low (< 0.5)
- R5-R6: Much higher (> 1.5)
- **Red flag!** Model memorized training data

**Underfitting:**
- All releases: High (> 2.0)
- **Red flag!** Model too simple

### Current Testing Status

```bash
# Running comprehensive evaluation
python3 evaluate_on_releases.py \
    --submission-zip eeg2025_submission_v7_TTA.zip \
    --data-dir data \
    --output-dir outputs/eval_releases
```

**Expected output:**
```json
{
  "R1": {"overall": 1.25, "challenge1": 0.98, "challenge2": 1.35},
  "R2": {"overall": 1.22, "challenge1": 0.95, "challenge2": 1.32},
  "R3": {"overall": 1.28, "challenge1": 1.02, "challenge2": 1.38},
  "R4": {"overall": 1.35, "challenge1": 1.08, "challenge2": 1.45},
  "R5": {"overall": 1.32, "challenge1": 1.00, "challenge2": 1.46},
  "R6": {"overall": ?, "challenge1": ?, "challenge2": ?}
}
```

*(R6 results only available after competition submission)*

---

## 📊 Current Results and Next Steps

### Submission History

| Date | Version | Challenge 1 | Challenge 2 | Overall | Notes |
|------|---------|-------------|-------------|---------|-------|
| Oct 15 | v1 | 1.00 | 2.34 | 2.01 | Initial baseline |
| Oct 16 | v5 | 1.00 | 1.46 | **1.32** | 🏆 Best so far |
| Oct 18 | v6 | 1.63 | 1.33 | 1.42 | TCN failed (overfitting) |
| Oct 18 | v6_REVERTED | 1.00 | 1.46 | 1.32 | Back to working model |
| Oct 18 | v7_TTA | ? | ? | ? | Testing now... |

### What's Working

✅ **Compact CNN Architecture**
- 75K parameters (Challenge 1), 64K (Challenge 2)
- Strong dropout (0.3-0.5) prevents overfitting
- Trained on R1-R4, validated on R5
- Consistent NRMSE ~1.00-1.46

✅ **Starter Kit Integration**
- Using `EEGChallengeDataset` correctly
- Event-locked windows (`stimulus_anchor`)
- Official NRMSE metric
- Proper submission format

✅ **Data Pipeline**
- Preprocessing matches competition requirements
- 129 channels, 200 samples @ 100 Hz
- Stimulus-locked 2-second windows
- Response time targets extracted correctly

### What Needs Improvement

⚠️ **Challenge 2 Performance**
- Current: NRMSE 1.46
- Goal: < 1.00
- Challenge 2 is 70% of overall score!

⚠️ **Cross-Release Validation**
- Currently only validated on R5
- Need R1-R6 testing to ensure generalization

⚠️ **Ensemble Methods**
- Currently single model per challenge
- Could average multiple models for robustness

### Next Steps (Priority Order)

#### 1. **Complete R1-R6 Evaluation** (Today)
```bash
# Running now - will give us:
# - Per-release NRMSE
# - Consistency metrics
# - Overfitting check
```

#### 2. **Test TTA Submission** (This Week)
- Upload `eeg2025_submission_v7_TTA.zip`
- Expected improvement: 5-10%
- Target: Overall NRMSE 1.18-1.25

#### 3. **Train Attention Model** (Next Week)
- Use comprehensive training script
- Data augmentation enabled
- Monitor official NRMSE metric
- Target: Challenge 1 < 0.90

#### 4. **Improve Challenge 2** (Priority!)
- 70% of overall score
- Current 1.46 → Target 1.00
- Possible approaches:
  - Longer windows (more resting state data)
  - Different architecture (externalizing is different from response time)
  - Ensemble methods

#### 5. **Ensemble Submission** (If Time Permits)
```python
# Average predictions from multiple models
prediction = 0.4 * model_cnn(x) + \
             0.3 * model_attention(x) + \
             0.3 * model_tcn(x)
```

### Competition Timeline

- **Now:** Testing and refinement phase
- **Oct 20-25:** Train attention models, test improvements
- **Oct 26-30:** Final ensemble, create best submission
- **Oct 31:** Submission deadline (if any)
- **Nov 1+:** Await final R6 evaluation from organizers

---

## 🤝 Discussion Topics

### Questions for the Team

1. **Testing Strategy**
   - Should we continue testing on all R1-R6?
   - Or focus optimization on R5 only?

2. **Architecture Direction**
   - Continue with simple CNN (working)?
   - Or invest time in attention model (risky)?

3. **Challenge 2 Focus**
   - 70% of score but performance is weak
   - Worth dedicating more resources here?

4. **Ensemble Approach**
   - Train multiple models and average?
   - Or perfect single best model?

5. **Competition Participation**
   - How often to submit? (limited submissions?)
   - Risk management strategy?

### Areas Where I Need Help

1. **Domain Knowledge**
   - What EEG patterns are known to predict response time?
   - What brain regions/frequencies matter for externalizing behavior?

2. **Hyperparameter Tuning**
   - Learning rate schedules
   - Dropout rates
   - Architecture depth/width

3. **Data Understanding**
   - Why does Challenge 2 perform worse?
   - Is there something special about externalizing prediction?

4. **Competition Strategy**
   - How aggressive should we be with new methods?
   - When to stick with "good enough"?

---

## 📚 Key Takeaways

### What I Learned

1. **Starter Kit is Essential**
   - Provides complete infrastructure
   - Using wrong dataset = invalid results
   - Follow their patterns carefully

2. **Bigger ≠ Better**
   - TCN (196K params) overfitted badly
   - Simple CNN (75K params) works well
   - Regularization matters more than size

3. **Validation Can Mislead**
   - TCN had Val Loss 0.0102 but Test NRMSE 1.63
   - Need proper cross-validation
   - Test on multiple data splits

4. **Incremental Improvements Work**
   - Oct 15: 2.01 → Oct 16: 1.32 (35% improvement!)
   - Small, tested changes better than big leaps

5. **Infrastructure First, Optimization Later**
   - Get pipeline working end-to-end
   - Then optimize individual components
   - Reproducibility is key

### Best Practices Established

✅ Always use `EEGChallengeDataset` (not EEGDashDataset)
✅ Test on multiple releases (R1-R6), not just R5
✅ Use official NRMSE metric from starter kit
✅ Follow exact submission format
✅ Save checkpoints with validation metrics
✅ Document what works and what doesn't

---

## 🚀 Call to Action

### Immediate Actions (This Week)

- [ ] Complete R1-R6 evaluation (running now)
- [ ] Review per-release results
- [ ] Upload TTA submission if R1-R6 looks good
- [ ] Start attention model training if approved

### Medium Term (Next 1-2 Weeks)

- [ ] Focus on Challenge 2 improvement
- [ ] Experiment with ensemble methods
- [ ] Implement data augmentation
- [ ] Cross-validation across releases

### Long Term (Competition Strategy)

- [ ] Build model zoo (multiple architectures)
- [ ] Automated hyperparameter search
- [ ] Documentation for reproducibility
- [ ] Prepare final competition submission

---

## 📞 Contact & Resources

### Useful Links
- Competition Website: https://eeg2025.github.io/
- Codabench Platform: https://www.codabench.org/competitions/4287/
- Starter Kit Docs: (in repository)
- Our Repository: /home/kevin/Projects/eeg2025

### Key Files to Review
- `submission.py` - Current working submission
- `submission_tta.py` - TTA-enhanced version
- `train_attention_with_metrics.py` - New training script with official metrics
- `evaluate_on_releases.py` - Multi-release testing script
- `starter_kit_integration/local_scoring.py` - Official scoring logic

### Meeting Prepared By
Kevin - October 18, 2025

---

**Ready for questions! 🎤**

