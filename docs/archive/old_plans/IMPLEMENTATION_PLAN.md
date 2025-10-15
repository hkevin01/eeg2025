# Complete Implementation Plan - EEG 2025 Competition

## ✅ Current Status

### What Works
- ✅ Dataset loader (SimpleEEGDataset) - Tested and working
- ✅ Model architecture (SimpleTransformer) - Tested and working
- ✅ Training loop - Confirmed working
- ✅ Data: 12 subjects with EEG data in HBN dataset

### What We Know
- Training 1 epoch on 2 subjects (~6414 windows) takes ~5 minutes
- Full dataset will have ~38,000 windows (12 subjects)
- Estimated training time: ~30-45 minutes per epoch on CPU
- For 5 epochs: ~2.5-3.5 hours total

## 📋 Step-by-Step Implementation

### Step 1: Foundation Model Training ⭐⭐⭐ (PRIORITY)

**Objective:** Train a foundation transformer model on all HBN data

**Files Needed:**
- `scripts/train_foundation.py` - Main training script
- Uses: `scripts/models/eeg_dataset_simple.py`

**Configuration:**
```python
{
    'hidden_dim': 128,        # Model capacity
    'n_heads': 8,             # Attention heads
    'n_layers': 4,            # Transformer layers
    'batch_size': 16,         # Batch size
    'epochs': 10,             # Training epochs
    'learning_rate': 1e-4,    # Learning rate
}
```

**Expected Output:**
- `checkpoints/foundation_best.pth` - Best model checkpoint
- `logs/training_history.json` - Training metrics

**Estimated Time:** 4-6 hours

**Command:**
```bash
python3 scripts/train_foundation.py > logs/foundation_training.log 2>&1 &
tail -f logs/foundation_training.log
```

---

### Step 2: Challenge 1 - Age Prediction (Regression) ⭐⭐⭐

**Objective:** Predict continuous age from EEG data

**Files Needed:**
- `scripts/challenge1_age_prediction.py` - Transfer learning for age
- Uses foundation model + regression head

**Data:**
- Load from: `data/raw/hbn/participants.tsv`
- Target: `Age` column (continuous)

**Model Architecture:**
```python
Foundation Transformer (frozen or fine-tuned)
    ↓
Global Average Pooling
    ↓
Regression Head: Linear(hidden_dim → 1)
    ↓
Age Prediction (years)
```

**Training:**
- Loss: MSE (Mean Squared Error) or Huber Loss
- Metric: Pearson Correlation
- Target: r > 0.3

**Submission Format:**
```csv
ID,Predicted
subject_001,12.5
subject_002,14.2
...
```

---

### Step 3: Challenge 2 - Sex Classification (Binary) ⭐⭐⭐

**Objective:** Classify biological sex from EEG data

**Files Needed:**
- `scripts/challenge2_sex_classification.py` - Transfer learning for sex

**Data:**
- Load from: `data/raw/hbn/participants.tsv`
- Target: `Sex` column (M/F → 0/1)

**Model Architecture:**
```python
Foundation Transformer (frozen or fine-tuned)
    ↓
Global Average Pooling
    ↓
Classification Head: Linear(hidden_dim → 2)
    ↓
Softmax → [P(Female), P(Male)]
```

**Training:**
- Loss: CrossEntropyLoss
- Metric: AUROC (Area Under ROC Curve)
- Target: AUROC > 0.7

**Submission Format:**
```csv
ID,Predicted
subject_001,0.85
subject_002,0.32
...
```

---

## 🛠️ Implementation Order

### Phase 1: Foundation (Current - 4-6 hours)
1. ✅ Verify dataset loader works
2. ✅ Verify model architecture works
3. ✅ Verify training loop works
4. 🔄 Run full foundation training
   - **Status:** Ready to start
   - **Command:** `python3 scripts/train_foundation.py`
   - **Monitor:** `tail -f logs/foundation_training.log`

### Phase 2: Challenge 1 (After foundation - 2-3 hours)
1. Load participants.tsv and extract age labels
2. Create regression dataset
3. Load foundation model
4. Add regression head
5. Train with MSE loss
6. Evaluate with Pearson correlation
7. Generate predictions for test set
8. Create submission CSV

### Phase 3: Challenge 2 (After Challenge 1 - 2-3 hours)
1. Load participants.tsv and extract sex labels
2. Create classification dataset
3. Load foundation model
4. Add classification head
5. Train with CrossEntropy loss
6. Evaluate with AUROC
7. Generate predictions for test set
8. Create submission CSV

### Phase 4: Submission (Final - 30 minutes)
1. Verify submission formats
2. Test locally
3. Upload to competition platform
4. Monitor results

---

## 📊 Expected Performance

### Foundation Model
- **Val Accuracy:** 60-75% (baseline task)
- **Val Loss:** 0.4-0.6

### Challenge 1 (Age Prediction)
- **Pearson r:** 0.3-0.5 (Target: > 0.3)
- **MAE:** 2-4 years

### Challenge 2 (Sex Classification)
- **AUROC:** 0.7-0.85 (Target: > 0.7)
- **Accuracy:** 70-80%

---

## 🚀 Quick Start

### Option A: Run Everything (Recommended)
```bash
# Step 1: Foundation training (4-6 hours)
nohup python3 scripts/train_foundation.py > logs/foundation.log 2>&1 &

# Monitor
tail -f logs/foundation.log

# Step 2: After foundation completes, run Challenge 1
nohup python3 scripts/challenge1_age_prediction.py > logs/challenge1.log 2>&1 &

# Step 3: After Challenge 1 completes, run Challenge 2
nohup python3 scripts/challenge2_sex_classification.py > logs/challenge2.log 2>&1 &
```

### Option B: Interactive (For Debugging)
```bash
# Run each step interactively to see progress
python3 scripts/train_foundation.py
python3 scripts/challenge1_age_prediction.py
python3 scripts/challenge2_sex_classification.py
```

---

## 📝 Files to Create

1. ✅ `scripts/train_foundation.py` - Full foundation training
2. ⭕ `scripts/challenge1_age_prediction.py` - Age regression
3. ⭕ `scripts/challenge2_sex_classification.py` - Sex classification
4. ⭕ `scripts/evaluate_model.py` - Model evaluation utilities
5. ⭕ `scripts/create_submission.py` - Submission file generation

---

## �� Success Criteria

- [x] Dataset loads successfully
- [x] Model trains without errors
- [ ] Foundation model trains to completion
- [ ] Challenge 1 achieves Pearson r > 0.3
- [ ] Challenge 2 achieves AUROC > 0.7
- [ ] Submission files generated
- [ ] Files uploaded to competition

---

## 💡 Tips for Success

1. **Start Small, Scale Up**
   - Test with 2 subjects first
   - Then scale to full dataset

2. **Monitor Training**
   - Use `tail -f logs/*.log` to watch progress
   - Check for overfitting (train vs val gap)

3. **Save Checkpoints**
   - Save every epoch
   - Keep best model based on validation

4. **Transfer Learning Strategy**
   - Option A: Freeze foundation, train head only (fast, good baseline)
   - Option B: Fine-tune entire model (slower, potentially better)

5. **Hyperparameter Tuning**
   - Learning rate: Try 1e-4, 5e-5, 1e-5
   - Batch size: Adjust based on memory
   - Epochs: 10-20 for foundation, 5-10 for challenges

---

## 🔧 Troubleshooting

### If Training is Slow
- Reduce batch size
- Reduce model size (hidden_dim, n_layers)
- Use fewer subjects for initial testing

### If Memory Issues
- Reduce batch size
- Reduce hidden_dim
- Use gradient accumulation

### If Accuracy is Low
- Train longer (more epochs)
- Increase model capacity
- Try different learning rates
- Check data quality

---

## 📈 Timeline Summary

| Phase | Task | Duration | Status |
|-------|------|----------|--------|
| 1 | Foundation Training | 4-6 hrs | 🔄 Ready |
| 2 | Challenge 1 (Age) | 2-3 hrs | ⭕ Pending |
| 3 | Challenge 2 (Sex) | 2-3 hrs | ⭕ Pending |
| 4 | Evaluation & Submission | 0.5 hrs | ⭕ Pending |
| **Total** | **End-to-End** | **9-12 hrs** | **In Progress** |

---

**Generated:** October 14, 2025  
**Status:** Foundation training ready to start  
**Next Action:** Run `python3 scripts/train_foundation.py`
