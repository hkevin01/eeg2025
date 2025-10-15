# Competition Implementation TODO - October 15, 2025

## 🎯 Current Focus: Training Challenge Models

Based on our README.md and competition implementation plan, here's what needs to be done:

### ✅ Already Complete
- [x] HBN dataset downloaded (12 subjects with RestingState data)
- [x] Age prediction model trained (MAE=0.30yr, Corr=0.9851)
- [x] Competition trainers implemented (challenge1_trainer.py, challenge2_trainer.py)
- [x] Data loading pipelines ready
- [x] Official metrics implemented
- [x] P0 and P1 tasks complete

### 🔥 HIGH PRIORITY - Next Steps

#### 1. Challenge 1: Cross-Task Transfer Training
**Status**: ⭕ Not Started  
**Goal**: Train SuS → CCD transfer learning model

**Tasks**:
- [ ] Verify we have both SuS and CCD task data from HBN
- [ ] Run challenge1 training script
- [ ] Achieve target metrics:
  - RT Correlation > 0.3
  - Success Balanced Accuracy > 0.6
- [ ] Save best checkpoint
- [ ] Document results

**Commands**:
```bash
# Check available task data
python scripts/check_available_tasks.py

# Train Challenge 1
python scripts/train_challenge1.py \
    --data_root data/raw/hbn \
    --source_task SuS \
    --target_task CCD \
    --use_progressive_unfreezing \
    --batch_size 16 \
    --max_epochs 30 \
    --output_dir runs/challenge1
```

#### 2. Challenge 2: Psychopathology Prediction
**Status**: ⭕ Not Started  
**Goal**: Predict P-factor from multi-task EEG

**Tasks**:
- [ ] Verify CBCL clinical scores available
- [ ] Run challenge2 training script
- [ ] Achieve target metrics:
  - P-factor correlation > 0.2
  - Clinical score MAE < 10
- [ ] Save best checkpoint
- [ ] Document results

**Commands**:
```bash
# Train Challenge 2
python scripts/train_challenge2.py \
    --data_root data/raw/hbn \
    --tasks RS,SuS,CCD \
    --use_clinical_scores \
    --batch_size 16 \
    --max_epochs 30 \
    --output_dir runs/challenge2
```

#### 3. Evaluation & Submission Generation
**Status**: ⭕ Not Started  
**Goal**: Generate competition-compliant predictions

**Tasks**:
- [ ] Run evaluation on test set
- [ ] Generate submission CSV files
- [ ] Validate submission format
- [ ] Document final metrics

**Commands**:
```bash
# Generate submissions
python scripts/evaluate_competition.py \
    --challenge challenge1 \
    --checkpoint runs/challenge1/best_model.ckpt \
    --data_root data/raw/hbn \
    --output_dir submissions/challenge1

python scripts/evaluate_competition.py \
    --challenge challenge2 \
    --checkpoint runs/challenge2/best_model.ckpt \
    --data_root data/raw/hbn \
    --output_dir submissions/challenge2
```

### 📊 Current Dataset Status
- **Subjects**: 12 with RestingState EEG
- **Age Range**: 6.4 - 14.0 years
- **Segments**: 4,530 total
- **Tasks Available**: Need to verify SuS, CCD, MW, SL, SyS availability

### 🚨 Known Constraints
- **Hardware**: CPU-only (AMD GPU unstable)
- **Training Time**: Expect 2-4 hours per challenge on CPU
- **Batch Size**: Keep at 16 due to memory constraints
- **Epochs**: Start with 30, can adjust based on convergence

### 📈 Success Criteria

**Challenge 1**:
- ✅ Model trains without errors
- ✅ RT correlation > 0.3
- ✅ Success accuracy > 0.6
- ✅ Submission file generated

**Challenge 2**:
- ✅ Model trains without errors
- ✅ P-factor correlation > 0.2
- ✅ Clinical MAE < 10
- ✅ Submission file generated

### 🔄 Workflow

1. **Verify Data** → Check task availability
2. **Train Challenge 1** → SuS → CCD transfer
3. **Train Challenge 2** → Multi-task → P-factor
4. **Evaluate Both** → Generate submissions
5. **Document Results** → Update README with metrics

### 📝 Notes
- Age prediction already successful (0.30yr MAE) - use insights for challenges
- CPU-only training worked well - continue this approach
- Real labels critical - ensure we're using official HBN labels
- Competition infrastructure ready - just need to execute training
