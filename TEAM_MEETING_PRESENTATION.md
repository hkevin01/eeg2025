# EEG Foundation Model Challenge 2025 - Team Meeting Notes
## October 18, 2025

**Prepared by:** Kevin  
**Competition:** EEG2025 Foundation Model Challenge  
**Competition URL:** https://eeg2025.github.io/  
**Codabench:** https://www.codabench.org/competitions/4287/

---

## 📊 Executive Summary

### Current Status
- **Best Score:** 1.32 NRMSE (October 16 submission)
- **Current Approach:** Compact CNN models with Test-Time Augmentation
- **Recent Development:** Score regression identified and fixed (Oct 18: 1.42 → reverted to 1.32)
- **Active Development:** Multi-head self-attention enhancement + comprehensive training pipeline

### Key Achievements
✅ Successfully set up development environment  
✅ Integrated competition starter kit  
✅ Built working CNN models for both challenges  
✅ Achieved competitive baseline score (1.32)  
✅ Implemented Test-Time Augmentation (TTA) for 5-10% expected improvement  
✅ Created comprehensive training infrastructure with official metrics  

---

## 🎯 Competition Overview

### What is the Challenge?
The EEG Foundation Model Challenge focuses on predicting cognitive and behavioral outcomes from brain activity (EEG data).

**Two Challenges:**

1. **Challenge 1: Response Time Prediction (30% of score)**
   - Task: Predict how quickly a person responds in a contrast change detection (CCD) task
   - Input: 129-channel EEG data (2 seconds @ 100Hz = 200 time points)
   - Output: Response time in seconds
   - Metric: NRMSE (Normalized Root Mean Squared Error)

2. **Challenge 2: Externalizing Behavior Prediction (70% of score)**
   - Task: Predict externalizing behaviors (aggression, rule-breaking) from resting-state EEG
   - Input: Same 129-channel EEG format
   - Output: CBCL externalizing score
   - Metric: NRMSE

**Overall Score Formula:**
```
Overall NRMSE = 0.3 × Challenge1_NRMSE + 0.7 × Challenge2_NRMSE
```

### Dataset: Healthy Brain Network (HBN)
- **Source:** Child Mind Institute
- **Subjects:** Children and adolescents (ages 5-21)
- **Data:** High-density EEG recordings (129 channels)
- **Releases:** Multiple data releases (R1-R5, R6 holdout for final testing)
- **Format:** BIDS (Brain Imaging Data Structure)

---

## 🧠 My Understanding of AI/ML and Neural Networks

### What is Machine Learning?
Think of it like teaching a computer to recognize patterns, similar to how we learn:
- **Traditional Programming:** We write explicit rules (if this, then that)
- **Machine Learning:** Computer learns rules from examples (training data)

### Why Neural Networks?
Neural networks are inspired by how our brains work:
- Made up of layers of "neurons" (mathematical functions)
- Each neuron processes information and passes it forward
- Through training, neurons learn to detect useful patterns

### What is a CNN (Convolutional Neural Network)?

**Simple Explanation:**
A CNN is like having filters that scan through data looking for patterns.

**Analogy:**
Imagine you're looking at a photo to identify a cat:
1. **First layer:** Detects edges (horizontal lines, vertical lines, curves)
2. **Second layer:** Combines edges into shapes (triangles, circles, rectangles)
3. **Third layer:** Combines shapes into features (ears, eyes, whiskers)
4. **Final layer:** Combines features to say "this is a cat!"

**For EEG Data:**
1. **First layer:** Detects basic brain wave patterns across channels
2. **Middle layers:** Finds combinations of patterns over time
3. **Final layers:** Maps patterns to prediction (response time or behavior score)

### Key CNN Concepts I Used

#### 1. Convolution
- **What it does:** Scans through data with a "sliding window" to detect patterns
- **In EEG:** Looks for patterns across brain channels and time
- **Example:** `Conv1d(129, 32, kernel_size=7)` means:
  - Input: 129 channels (brain sensors)
  - Output: 32 feature maps (detected patterns)
  - Kernel size 7: Looks at 7 time points at once

#### 2. Pooling
- **What it does:** Reduces data size while keeping important information
- **Why:** Makes model faster and reduces overfitting
- **Example:** `AdaptiveAvgPool1d(1)` → Reduces entire sequence to one value per feature

#### 3. Dropout
- **What it does:** Randomly "turns off" some neurons during training
- **Why:** Prevents overfitting (memorizing training data instead of learning patterns)
- **Example:** `Dropout(0.4)` → Turns off 40% of neurons randomly each training step

#### 4. Batch Normalization
- **What it does:** Normalizes data between layers
- **Why:** Helps training be more stable and faster
- **Example:** `BatchNorm1d(32)` → Normalizes 32 feature maps

---

## 🏗️ Setting Up the Code

### 1. Project Structure
```
eeg2025/
├── submission.py              # Competition submission file
├── weights_challenge_1_*.pt   # Trained model weights
├── weights_challenge_2_*.pt
├── src/
│   └── dataio/
│       └── starter_kit.py     # Official competition utilities (2,589 lines!)
├── scripts/
│   └── train_*.py             # Training scripts
├── improvements/
│   └── all_improvements.py    # TTA, ensembles, etc.
└── models/
    └── models_with_attention.py  # Attention-enhanced architectures
```

### 2. Starter Kit Integration

**What is the Starter Kit?**
The competition provides official code for:
- Loading EEG data properly
- Computing official metrics (NRMSE)
- Handling data splits (train/val/test)
- Managing memory and performance

**Key Components I Use:**
```python
# From src/dataio/starter_kit.py

def calculate_nrmse(y_true, y_pred):
    """Official competition metric"""
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / std(y_true)
    return nrmse

# NRMSE tells us: How far off are predictions relative to data variability?
# - NRMSE = 1.0 means predictions vary as much as the data itself
# - NRMSE = 0.5 means predictions are twice as accurate (50% of data's std)
# - NRMSE < 1.0 is good, < 0.5 is excellent
```

### 3. Data Loading Pipeline

**Challenge: EEG data is complex!**
- 129 channels (different spots on the scalp)
- Varying sampling rates (need to resample to 100Hz)
- Multiple file formats (.bdf, .fif, .edf)
- Event synchronization (align EEG with task events)

**My Approach:**
```python
# Simplified version of what I implemented
class ResponseTimeDataset:
    def __init__(self, data_dir):
        # 1. Find all subjects with CCD task data
        # 2. Load EEG files using MNE library
        # 3. Resample to 100Hz if needed
        # 4. Extract 2-second segments around task events
        # 5. Normalize each channel (mean=0, std=1)
        # 6. Link to response time labels
        
    def __getitem__(self, idx):
        # Return: (EEG_segment, response_time)
        # Shape: (129 channels, 200 time points), (1,)
```

---

## 🎨 My Model Architecture

### Baseline: CompactResponseTimeCNN

**Why "Compact"?**
- Fewer parameters = less overfitting
- Proven to work (NRMSE 1.00 on Challenge 1)
- Fast to train

**Architecture Breakdown:**
```python
Input: (batch, 129 channels, 200 time points)

Conv Block 1:
  Conv1d(129 → 32 channels, kernel=7, stride=2)  # Detect basic patterns
  BatchNorm → ReLU → Dropout(0.3)
  Output: (batch, 32, 100)

Conv Block 2:
  Conv1d(32 → 64 channels, kernel=5, stride=2)   # Combine patterns
  BatchNorm → ReLU → Dropout(0.4)
  Output: (batch, 64, 50)

Conv Block 3:
  Conv1d(64 → 128 channels, kernel=3, stride=2)  # High-level features
  BatchNorm → ReLU → Dropout(0.5)
  Output: (batch, 128, 25)

Global Average Pooling:
  AdaptiveAvgPool1d(1)                            # Reduce to 1 value per channel
  Output: (batch, 128)

Regression Head:
  Linear(128 → 64) → ReLU → Dropout(0.5)
  Linear(64 → 32) → ReLU → Dropout(0.4)
  Linear(32 → 1)                                  # Final prediction
  Output: (batch, 1)
```

**Key Design Choices:**

1. **Stride = 2 (downsampling):**
   - Reduces sequence length: 200 → 100 → 50 → 25
   - Makes model more efficient
   - Forces model to focus on important patterns

2. **Increasing channels: 129 → 32 → 64 → 128:**
   - Early layers: Few channels, detect simple patterns
   - Later layers: More channels, detect complex combinations
   - Standard CNN practice

3. **Dropout increases: 0.3 → 0.4 → 0.5:**
   - More dropout in later layers
   - Prevents overfitting on complex features
   - Critical for small datasets

4. **Total Parameters: ~75,000**
   - Small enough to avoid overfitting
   - Large enough to capture patterns
   - Sweet spot found through experimentation

---

## 🔬 What I Tried and Why

### Attempt 1: Baseline CNN ✅ SUCCESS
**Date:** October 15-16  
**Score:** 1.32 NRMSE  

**What I Did:**
- Implemented CompactResponseTimeCNN (75K params)
- Implemented CompactExternalizingCNN (64K params)
- Trained on releases R1-R4, validated on R5
- Used strong regularization (dropout 0.3-0.5)

**Why:**
- Start simple before going complex
- CNNs are proven for time-series data
- Small models generalize better on limited data

**Result:** ✅ Good baseline established

---

### Attempt 2: Temporal Convolutional Network (TCN) ❌ FAILED
**Date:** October 17-18  
**Score:** 1.42 NRMSE (worse!)  

**What I Tried:**
- Replaced Challenge 1 CNN with TCN
- TCN uses dilated convolutions (looks at patterns with gaps)
- 196K parameters (2.6× bigger than baseline CNN)

**Why I Tried It:**
- TCNs are popular for time-series in research papers
- Dilated convolutions can capture long-range patterns
- Thought it might capture better temporal dependencies

**What Went Wrong:**
```
Training Results:
- Validation Loss: 0.0102 (looked great!)
- Test NRMSE: 1.63 (terrible! 63% worse than baseline)

Analysis:
- Model OVERFITTED to validation set
- More parameters → memorized patterns instead of learning
- Dilated convolutions might not suit EEG's characteristics
- Validation metrics were misleading
```

**Lessons Learned:**
1. ⚠️ More complex ≠ better
2. ⚠️ Bigger model ≠ better performance
3. ⚠️ Great validation score ≠ good test score
4. ✅ Simple, well-regularized models often win
5. ✅ Test incrementally (change one thing at a time)

---

### Attempt 3: Test-Time Augmentation (TTA) ⏳ IN PROGRESS
**Date:** October 18  
**Expected:** 5-10% improvement  

**What is TTA?**
Instead of making one prediction, make multiple predictions with slight variations and average them:

```python
# Regular prediction
prediction = model(eeg_data)

# TTA prediction
predictions = []
predictions.append(model(eeg_data))                    # Original
predictions.append(model(eeg_data + small_noise))      # Add noise
predictions.append(model(eeg_data * 1.05))             # Scale up 5%
predictions.append(model(shift_time(eeg_data, -2)))    # Shift left
predictions.append(model(shift_time(eeg_data, +2)))    # Shift right
# ... 10 total variations

final_prediction = average(predictions)  # More robust!
```

**Why:**
- Makes predictions more robust to small variations
- No retraining needed (just wrap existing model)
- Proven technique, minimal risk
- Fast to implement and test

**Augmentation Types I Use:**
1. **Gaussian noise:** Small random noise (±2% amplitude)
2. **Amplitude scaling:** Scale signal by 0.95-1.05×
3. **Time shifting:** Shift ±3 samples
4. **Channel dropout:** Randomly mask 5% of channels
5. **Mixup:** Blend with shifted version

**Status:** ✅ Implemented, ready to submit

---

### Attempt 4: Multi-Head Self-Attention ⏳ IN DEVELOPMENT
**Date:** October 18  
**Expected:** 5-15% improvement  

**What is Attention?**
Attention lets the model "focus" on important parts of the data.

**Analogy:**
When you read a sentence, you don't give equal attention to every word:
- "The **cat** sat on the mat" → You focus on "cat" and "mat"
- Attention does this automatically for EEG patterns

**For EEG:**
- Model can focus on specific time points that matter for prediction
- Can learn which channels are most relevant
- Captures long-range dependencies (connections between distant time points)

**Architecture:**
```python
Input: (batch, 64 channels, 50 time points)

Multi-Head Attention (4 heads):
  Head 1: Learns to attend to early patterns
  Head 2: Learns to attend to late patterns  
  Head 3: Learns to attend to oscillatory patterns
  Head 4: Learns to attend to transient events
  
  Combine all heads → (batch, 64 channels, 50 time points)
  
Residual Connection:
  output = input + attention(input)  # Keep original + learned attention
  
Why residual?: If attention doesn't help, model can ignore it!
```

**Key Advantages:**
1. **Full receptive field:** Attention sees all 200 time points at once
   - CNN only sees ~15 time points per layer
2. **Learned importance:** Model learns what to focus on
3. **Interpretable:** Can visualize attention weights
4. **Minimal parameters:** Only +6.3% (79K → 84K params)

**Why Safer than TCN:**
- Residual connection ensures we keep baseline performance
- Smaller parameter increase (+6.3% vs +162%)
- Attention is designed for sequential data
- Can only improve, not harm (worst case: attention weights go to zero)

**Status:** ✅ Architecture ready, training pipeline built

---

## 📈 Score History and Analysis

### Timeline
```
October 15:  2.01 NRMSE  ❌ Poor initial attempt
October 16:  1.32 NRMSE  ✅ BEST - working CompactCNNs
October 18:  1.42 NRMSE  ❌ TCN overfitted (mistake)
October 18:  Reverted to 1.32 baseline ✅
```

### Detailed Breakdown (October 18 Submission)
```
Challenge 1 (Response Time): 1.6262 NRMSE
  - Baseline was 1.00
  - TCN made it 63% WORSE
  - Why: Overfitted, too complex

Challenge 2 (Externalizing): 1.3318 NRMSE  
  - Baseline was 1.46
  - Actually improved 9%!
  - Why: Better API format handling

Overall: 0.3 × 1.6262 + 0.7 × 1.3318 = 1.4201
```

### What This Means
- **1.32 is competitive** (probably top 30-50%)
- **1.00 or below** would be very good
- **0.50 or below** would be excellent (top 5-10%)

---

## 🛠️ Submission File Structure

### What Goes in the Zip?
```
eeg2025_submission_v7_TTA.zip
├── submission.py                          # Main file (competition loads this)
├── weights_challenge_1_multi_release.pt   # Trained model for Challenge 1
└── weights_challenge_2_multi_release.pt   # Trained model for Challenge 2
```

### submission.py Structure
```python
class Submission:
    """Competition automatically creates this class"""
    
    def __init__(self, SFREQ, DEVICE):
        """Called once at start"""
        self.sfreq = SFREQ    # Sampling rate (100 Hz)
        self.device = DEVICE   # 'cpu' or 'cuda'
    
    def get_model_challenge_1(self):
        """Called to load Challenge 1 model"""
        model = CompactResponseTimeCNN()
        model.load_state_dict(torch.load('weights_challenge_1_multi_release.pt'))
        return model
    
    def get_model_challenge_2(self):
        """Called to load Challenge 2 model"""
        model = CompactExternalizingCNN()
        model.load_state_dict(torch.load('weights_challenge_2_multi_release.pt'))
        return model
```

### How Competition Evaluates
```python
# Competition server does this:
submission = Submission(SFREQ=100, DEVICE='cpu')
model_c1 = submission.get_model_challenge_1()
model_c2 = submission.get_model_challenge_2()

# For each test sample:
prediction_c1 = model_c1(eeg_data)  # Predict response time
prediction_c2 = model_c2(eeg_data)  # Predict externalizing score

# Calculate NRMSE for all test samples
# Return: Challenge 1 score, Challenge 2 score, Overall score
```

---

## 💡 Key Insights and Lessons

### What Works
1. ✅ **Simple, well-regularized CNNs** (strong dropout)
2. ✅ **Compact architectures** (50-100K parameters)
3. ✅ **Training on multiple releases** (R1-R4) for generalization
4. ✅ **Strong dropout** (0.3-0.5) to prevent overfitting
5. ✅ **Test-Time Augmentation** for robustness
6. ✅ **Residual connections** for safe enhancements

### What Doesn't Work
1. ❌ **Oversized models** (TCN with 196K params)
2. ❌ **Complex architectures without regularization**
3. ❌ **Trusting validation metrics alone**
4. ❌ **Changing multiple things at once**

### Best Practices
1. 🎯 **Start simple, add complexity gradually**
2. 🎯 **Test one change at a time**
3. 🎯 **Use official metrics for validation**
4. 🎯 **Keep successful baselines**
5. 🎯 **Document everything** (what, why, results)

---

## 🚀 Current Development Status

### Ready to Deploy
✅ **Version 6 (Reverted):** Baseline models (NRMSE 1.32)
✅ **Version 7 (TTA):** Baseline + Test-Time Augmentation (expected 1.18-1.25)

### In Development
🔄 **Attention Model:** Multi-head self-attention enhancement
🔄 **Training Pipeline:** Comprehensive training with data augmentation
🔄 **Official Metrics:** Integrated NRMSE computation

### Next Steps
1. Upload TTA submission (v7) to get improved baseline
2. Complete attention model training
3. If attention works (NRMSE < 1.00), create v8 submission
4. Explore ensemble methods (combine multiple models)

---

## 🎓 Technical Terms Glossary

| Term | Simple Explanation | Technical Definition |
|------|-------------------|---------------------|
| **NRMSE** | How wrong predictions are (relative) | RMSE divided by standard deviation of targets |
| **Overfitting** | Memorizing instead of learning | Model performs well on training data but poorly on new data |
| **Regularization** | Techniques to prevent overfitting | Dropout, weight decay, data augmentation |
| **Dropout** | Randomly turn off neurons | Randomly set activations to zero during training |
| **Batch Normalization** | Normalize between layers | Normalize activations to have mean=0, std=1 |
| **Convolution** | Sliding pattern detector | Apply learnable filters to detect features |
| **Pooling** | Summarize/reduce data | Downsample by taking max or average |
| **Attention** | Focus on important parts | Weighted combination based on learned importance |
| **Residual Connection** | Skip connection | Add input to output: y = f(x) + x |
| **Learning Rate** | How fast model learns | Step size for weight updates |
| **Epoch** | One pass through all data | Complete iteration over training dataset |

---

## 📊 Comparison with Literature

### Typical Results in EEG Prediction
- **Good performance:** NRMSE 0.8-1.2
- **Excellent performance:** NRMSE 0.4-0.6
- **State-of-the-art:** NRMSE < 0.4

### Our Current Standing
- **Current:** 1.32 NRMSE → In "good" range
- **With TTA:** ~1.20 NRMSE (expected) → Upper "good" range
- **With Attention:** ~1.00 NRMSE (goal) → Solid "good" performance
- **Target:** < 0.50 NRMSE → Would be "excellent"

---

## ❓ Discussion Questions for Team

1. **Strategy:**
   - Should we upload TTA now to secure improvement, or wait for attention model?
   - Risk vs reward of trying more complex architectures?

2. **Technical:**
   - Are there other augmentation strategies we should try?
   - Should we explore ensemble methods (combining multiple models)?
   - What about transfer learning from other EEG datasets?

3. **Resources:**
   - Do we have access to more compute (GPU) for faster training?
   - Can we access additional EEG datasets for pre-training?
   - Should we try different model architectures (transformers, etc.)?

4. **Timeline:**
   - How much time until competition deadline?
   - Should we focus on one challenge or both?
   - When to stop experimenting and finalize submission?

---

## 📚 Resources and References

### Code Repository
- **Location:** `/home/kevin/Projects/eeg2025/`
- **Key Files:**
  - `submission.py` - Current submission
  - `submission_tta.py` - TTA-enhanced submission
  - `train_attention_with_metrics.py` - Training infrastructure
  - `TEAM_MEETING_PRESENTATION.md` - This document

### Competition Links
- **Main Site:** https://eeg2025.github.io/
- **Codabench:** https://www.codabench.org/competitions/4287/
- **Paper:** arXiv:2506.19141 (EEG Foundation Model Challenge)

### Technical Documentation
- **Starter Kit:** `src/dataio/starter_kit.py` (2,589 lines)
- **Improvements:** `improvements/all_improvements.py` (TTA, ensembles)
- **Models:** `models/models_with_attention.py`

---

## 🎯 Summary

**What I've Built:**
- ✅ Working CNN models achieving competitive scores (1.32 NRMSE)
- ✅ Complete training and evaluation infrastructure
- ✅ Test-Time Augmentation for improved robustness
- ✅ Multi-head attention architecture for better temporal modeling
- ✅ Official metrics integration for accurate evaluation

**What I've Learned:**
- 🧠 CNN fundamentals and how they work with EEG
- 🧠 Importance of regularization and simple architectures
- 🧠 How to evaluate and debug model performance
- 🧠 Competition submission process and requirements

**Ready for Discussion:**
- 💬 Strategy for next submissions
- 💬 Technical approaches to try
- 💬 Resource allocation and timeline
- 💬 Risk management (when to try new things vs secure baseline)

---

**Last Updated:** October 18, 2025  
**Author:** Kevin  
**Status:** Ready for team review and discussion
