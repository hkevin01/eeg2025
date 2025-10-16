# Competition Action Plan - Updated from Official Site
## October 15, 2025

## 🎯 Competition Overview (from https://eeg2025.github.io/)

### Challenge 1: Cross-Task Transfer Learning
**Task**: Predict **response time** (regression) from CCD task
- **Input**: EEG windows (129 channels, 200 samples @ 100Hz = 2 seconds)
- **Output**: Response time (continuous value)
- **Metric**: NRMSE (Normalized RMSE)
- **Optional**: Pre-train on passive tasks (SuS), then fine-tune on CCD
- **Note**: SuS pretraining no longer mandatory

### Challenge 2: Externalizing Factor Prediction  
**Task**: Predict **externalizing factor only** (NOT p_factor, attention, internalizing)
- **Input**: EEG windows (129 channels, 200 samples @ 100Hz = 2 seconds)
- **Output**: Externalizing score (continuous value)
- **Metric**: NRMSE (Normalized RMSE)
- **Goal**: Subject-invariant representation

**IMPORTANT UPDATE**: Other clinical factors removed to streamline competition!

## 📊 Our Current Status

### ✅ What We Have:
1. **12 HBN subjects** with RestingState EEG
2. **Age prediction model**: MAE=0.30yr, Corr=0.9851 ✅
3. **Clinical prediction model**: 
   - Trained on 4 factors (p_factor, attention, internalizing, externalizing)
   - Mean correlation: 0.9763
   - **Externalizing**: 0.975 correlation, MAE=0.135 ✅

### ❌ What We Don't Have:
1. **CCD task data** for Challenge 1
2. **Submission.py** format implementation  
3. **NRMSE evaluation** metrics
4. **Inference-only** code structure

## 🚀 IMMEDIATE ACTION ITEMS

### Priority 1: Fix Challenge 2 Model (1-2 hours)

Our existing model predicts 4 factors, but competition only wants **externalizing**!

**Tasks**:
- [ ] Create Challenge 2 submission-ready model
- [ ] Train model that outputs ONLY externalizing (not 4 factors)
- [ ] Use 2-second windows (200 samples @ 100Hz)
- [ ] Test with local_scoring.py
- [ ] Create submission.py format

**Expected Output**:
- Model trained on externalizing only
- NRMSE < 0.5 (target)
- submission.py with get_model_challenge_2()

### Priority 2: Download CCD Data for Challenge 1 (2-4 hours)

**Tasks**:
- [ ] Query HBN for subjects with CCD task
- [ ] Download targeted subjects (aim for 50+)
- [ ] Verify CCD data quality
- [ ] Extract response time labels

**Expected Output**:
- 50+ subjects with CCD task
- Response time labels extracted
- Data ready for Challenge 1 training

### Priority 3: Train Challenge 1 Model (3-4 hours)

**Tasks**:
- [ ] Create Challenge 1 trainer
- [ ] Use 2-second windows from CCD
- [ ] Train response time prediction
- [ ] Achieve NRMSE < 0.5
- [ ] Create submission format

### Priority 4: Create Submission Package (1 hour)

**Tasks**:
- [ ] Implement Submission class
- [ ] get_model_challenge_1() method
- [ ] get_model_challenge_2() method
- [ ] Test with local_scoring.py
- [ ] Create submission.zip

## 📋 Detailed Steps

### Step 1: Retrain Challenge 2 (Externalizing Only)

```python
# scripts/train_challenge2_externalizing.py
# - Load HBN data
# - Extract externalizing factor ONLY
# - Create 2-second windows (200 samples @ 100Hz)
# - Train model: Input(129, 200) -> Output(1)
# - Save as externalizing_model.pth
```

### Step 2: Download CCD Data

```bash
# Use eegdash to download CCD task data
python scripts/download_ccd_data.py \
    --release R5,R6,R7 \
    --task contrastChangeDetection \
    --min_subjects 50 \
    --output data/raw/hbn_ccd
```

### Step 3: Train Challenge 1

```python
# scripts/train_challenge1_response_time.py
# - Load CCD data
# - Extract response time labels
# - Create stimulus-locked 2-second windows
# - Train model: Input(129, 200) -> Output(1)
# - Save as response_time_model.pth
```

### Step 4: Create Submission

```python
# submission.py (matching official format)
class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ  # 100 Hz
        self.device = DEVICE
    
    def get_model_challenge_1(self):
        model = load_model('response_time_model.pth')
        return model.to(self.device)
    
    def get_model_challenge_2(self):
        model = load_model('externalizing_model.pth')
        return model.to(self.device)
```

### Step 5: Test Locally

```bash
# Use official local_scoring.py
python local_scoring.py \
    --submission-zip submission.zip \
    --data-dir data/raw/hbn \
    --output-dir results/local_test
```

## 🎯 Success Criteria

### Challenge 1:
- ✅ Model trains without errors
- ✅ NRMSE < 0.5 (competitive)
- ✅ Inference-only submission works
- ✅ Local scoring passes

### Challenge 2:
- ✅ Model predicts externalizing only (not 4 factors)
- ✅ NRMSE < 0.3 (our correlation is 0.975!)
- ✅ Inference-only submission works
- ✅ Local scoring passes

### Submission:
- ✅ submission.zip format correct
- ✅ Both models load successfully
- ✅ Predictions match expected format
- ✅ Ready for Codabench upload

## 📈 Timeline

| Task | Estimated Time | Priority | Status |
|------|----------------|----------|--------|
| Retrain Challenge 2 (externalizing only) | 1-2 hours | 🔴 Critical | ⭕ Not Started |
| Download CCD data | 2-4 hours | 🔴 Critical | ⭕ Not Started |
| Train Challenge 1 | 3-4 hours | 🟠 High | ⭕ Blocked by data |
| Create submission.py | 1 hour | 🟠 High | ⭕ Not Started |
| Test with local_scoring | 30 min | 🟡 Medium | ⭕ Not Started |
| Upload to Codabench | 15 min | 🟡 Medium | ⭕ Not Started |

**Total Estimated Time**: 8-12 hours

## 🔧 Technical Requirements

### Model Requirements:
- **Input Shape**: (batch, 129, 200) - 129 channels, 200 samples
- **Output Shape**: (batch, 1) - single continuous value
- **Sampling Rate**: 100 Hz
- **Window Duration**: 2 seconds
- **Device**: CPU or CUDA

### Submission Format:
```
submission.zip
├── submission.py (required)
├── response_time_model.pth
├── externalizing_model.pth
└── (optional) any additional files
```

### Evaluation Metrics:
```python
# Challenge 1 & 2: NRMSE
NRMSE = RMSE(y_true, y_pred) / std(y_true)

# Lower is better
# Target: < 0.5 for competitive results
```

## 🚀 LET'S START!

**Next immediate action:**
1. Create `train_challenge2_externalizing.py` (externalizing only)
2. Retrain model on externalizing factor
3. Test inference format
4. Move to Challenge 1 data acquisition

---

*Based on: https://eeg2025.github.io/*  
*Starter Kit: https://github.com/eeg2025/startkit*  
*Last Updated: October 15, 2025*
