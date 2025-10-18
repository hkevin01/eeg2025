# Competition Implementation TODO - UPDATED
## October 15, 2025

## 🔍 Data Availability Assessment

### ✅ What We HAVE:
- 12 subjects with RestingState EEG data
- Clinical scores: p_factor, attention, internalizing, externalizing (132/136 available)
- Age labels (6.4 - 14.0 years)
- Movie-watching tasks: DespicableMe, DiaryOfAWimpyKid, ThePresent
- 4,530 EEG segments total
- Excellent age prediction model (MAE=0.30yr, Corr=0.9851)

### ❌ What We DON'T HAVE:
- SuS (Sustained Attention) task data
- CCD (Contrast Change Detection) task data
- MW, SL, SyS cognitive task data

## 🎯 REVISED PRIORITY FOCUS

### HIGH PRIORITY: Challenge 2 - Psychopathology Prediction

**Status**: ✅ READY TO TRAIN  
**Goal**: Predict clinical scores (p_factor, attention, internalizing, externalizing) from RestingState EEG

**Why we can do this:**
1. We have 12 subjects with RestingState EEG
2. We have clinical scores for these subjects
3. Similar to our successful age prediction approach
4. Challenge 2 trainer is already implemented

**Tasks**:
- [ ] Adapt Challenge 2 trainer for RestingState-only data
- [ ] Train psychopathology prediction model
- [ ] Target metrics:
  - P-factor correlation > 0.2
  - Individual factor MAE < 10
- [ ] Compare with age prediction success
- [ ] Document results

### MEDIUM PRIORITY: Enhanced Age Prediction

**Status**: ⭕ Can Improve  
**Goal**: Further improve our already excellent age prediction

**Tasks**:
- [ ] Add cross-validation
- [ ] Test ensemble methods
- [ ] Add confidence intervals
- [ ] Visualize predictions vs actual
- [ ] Create inference script for new data

### LOW PRIORITY: Download More Data for Challenge 1

**Status**: ⭕ Data Acquisition Needed  
**Goal**: Get subjects with SuS and CCD tasks

**Tasks**:
- [ ] Download more HBN subjects targeting SuS/CCD availability
- [ ] Verify task availability before training
- [ ] Then train Challenge 1

## 📝 EXECUTABLE PLAN

### Step 1: Train Challenge 2 (Psychopathology Prediction)

I'll create a simplified Challenge 2 trainer that works with our current data:

```bash
# Create adapted Challenge 2 trainer
python scripts/train_challenge2_resting.py \
    --data_root data/raw/hbn \
    --use_resting_state \
    --clinical_factors p_factor,attention,internalizing,externalizing \
    --batch_size 16 \
    --max_epochs 30 \
    --output_dir runs/challenge2_resting
```

**Expected Outcomes**:
- Model predicts 4 clinical factors from EEG
- Correlations similar to or better than age prediction (0.98)
- Saved checkpoint for inference
- Documented metrics

### Step 2: Enhance Age Prediction Model

```bash
# Add cross-validation to age prediction
python scripts/train_age_cv.py \
    --data_root data/raw/hbn \
    --n_folds 5 \
    --model simple_cnn \
    --output_dir runs/age_cv

# Create inference script
python scripts/predict_age.py \
    --checkpoint checkpoints/simple_cnn_age.pth \
    --input_eeg new_eeg_file.set \
    --output predictions.csv
```

### Step 3: Download Additional Data (if needed)

```bash
# Check which subjects have SuS/CCD
python scripts/query_hbn_subjects.py --has_task SuS,CCD

# Download specific subjects
bash scripts/download_hbn_data.sh --subjects sub-XXXX,sub-YYYY
```

## 🎯 Success Criteria

### Challenge 2 (Psychopathology)
- ✅ Model trains without errors
- ✅ P-factor correlation > 0.2 (target: 0.5+)
- ✅ Attention correlation > 0.2
- ✅ Internalizing correlation > 0.2
- ✅ Externalizing correlation > 0.2
- ✅ Results documented

### Enhanced Age Prediction
- ✅ Cross-validation complete
- ✅ Mean CV score matches or exceeds single model
- ✅ Inference script works on new data
- ✅ Confidence intervals calculated

## 📊 Expected Timeline

| Task | Est. Time | Priority |
|------|-----------|----------|
| Create Challenge 2 RestingState trainer | 30 min | 🔴 High |
| Train Challenge 2 model | 2-3 hours | 🔴 High |
| Add age prediction CV | 30 min | 🟡 Medium |
| Run age CV | 1-2 hours | 🟡 Medium |
| Create inference script | 30 min | 🟡 Medium |
| Download more subjects | 2-4 hours | 🟢 Low |

**Total High Priority Time**: ~3-4 hours  
**Total Medium Priority Time**: ~2-3 hours

## 🚀 LET'S START!

Next immediate action:
1. Create `scripts/train_challenge2_resting.py`
2. Adapt Challenge 2 trainer to work with RestingState data only
3. Train the model
4. Document results

---

**Note**: We're pivoting from Challenge 1 (can't do without SuS/CCD data) to Challenge 2 (ready to train now). This is the pragmatic approach given our current dataset.
