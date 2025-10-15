# 🚀 P2 Next Phase: Foundation Model Training

**Date**: October 14, 2025  
**Status**: ✅ GPU Safeguards Complete → Now scaling foundation training  
**Priority**: 🔴 Critical

---

## ✅ What We Accomplished

### GPU Safeguards Implementation
- ✅ Created timeout protection for GPU operations
- ✅ Automatic CPU fallback when GPU hangs
- ✅ Successfully tested on minimal dataset (2 epochs, 10 samples)
- ✅ Model training works perfectly on CPU
- ✅ All crash prevention measures in place

**Key Achievement**: **ZERO crashes** - system remains stable regardless of GPU issues

---

## 🎯 Next Phase Objectives

### Phase P2.2: Foundation Model Training (Full Scale)

**Goal**: Train a robust foundation model on all available HBN data

**Success Criteria**:
- ✅ Train on all 10 subjects (4,904 windows)
- ✅ Achieve >60% validation accuracy
- ✅ Model converges (loss decreasing)
- ✅ Save best checkpoint
- ✅ Complete in <6 hours on CPU

---

## 📋 Todo List

```markdown
### Phase 1: Data Preparation (30 min)
- [ ] Verify all 10 subjects have valid data
- [ ] Create train/val/test splits (70/15/15)
- [ ] Implement data caching for speed
- [ ] Calculate dataset statistics

### Phase 2: Foundation Training (3-4 hours)
- [ ] Configure training hyperparameters
- [ ] Start foundation model training on CPU
- [ ] Monitor training progress (loss, accuracy)
- [ ] Save checkpoints every epoch
- [ ] Identify best model

### Phase 3: Model Evaluation (30 min)
- [ ] Evaluate on validation set
- [ ] Calculate metrics (accuracy, loss, confusion matrix)
- [ ] Visualize training curves
- [ ] Document model performance

### Phase 4: Challenge 1 - CCD (1 hour)
- [ ] Load pretrained foundation model
- [ ] Create CCD-specific head
- [ ] Fine-tune on CCD data
- [ ] Evaluate: Pearson r, AUROC
- [ ] Generate submission file

### Phase 5: Challenge 2 - P-factors (1 hour)
- [ ] Load pretrained foundation model
- [ ] Create 4-output regression head
- [ ] Fine-tune on psychopathology data
- [ ] Evaluate: Average Pearson r
- [ ] Generate submission file

### Phase 6: Optimization (2 hours)
- [ ] Profile inference latency
- [ ] Apply quantization (FP32→FP16)
- [ ] Optimize data loading
- [ ] Benchmark final performance
- [ ] Update documentation
```

---

## 🏗️ Technical Plan

### Training Configuration

```python
CONFIG = {
    # Data
    'data_dir': 'data/raw/hbn',
    'max_subjects': 10,  # All available
    'train_split': 0.70,
    'val_split': 0.15,
    'test_split': 0.15,
    
    # Model
    'hidden_dim': 256,
    'n_heads': 8,
    'n_layers': 4,
    'dropout': 0.1,
    
    # Training
    'batch_size': 16,  # Balanced for CPU
    'epochs': 15,      # Enough for convergence
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'early_stopping_patience': 5,
    
    # Safety
    'checkpoint_every': 1,
    'validate_every': 1,
    'max_training_time': 6 * 3600,  # 6 hours max
}
```

### Expected Timeline

| Task | Duration | Start | End |
|------|----------|-------|-----|
| Data prep | 30 min | Now | +0.5h |
| Foundation training | 4 hours | +0.5h | +4.5h |
| Evaluation | 30 min | +4.5h | +5h |
| Challenge 1 | 1 hour | +5h | +6h |
| Challenge 2 | 1 hour | +6h | +7h |
| Optimization | 2 hours | +7h | +9h |

**Total**: ~9 hours to complete all P2 tasks

---

## 🚀 Execution Commands

### Step 1: Prepare Data
```bash
python3 scripts/prepare_foundation_data.py \
    --data-dir data/raw/hbn \
    --output-dir data/processed \
    --splits 0.7 0.15 0.15 \
    --cache
```

### Step 2: Train Foundation Model
```bash
python3 scripts/train_foundation_full.py \
    --data-dir data/processed \
    --config configs/foundation_training.yaml \
    --output-dir outputs/foundation \
    --device cpu \
    --verbose
```

### Step 3: Monitor Training
```bash
# In another terminal
watch -n 5 'tail -20 logs/foundation_training.log'
```

### Step 4: Evaluate Model
```bash
python3 scripts/evaluate_foundation.py \
    --checkpoint checkpoints/foundation_best.pth \
    --data-dir data/processed \
    --output results/foundation_eval.json
```

### Step 5: Challenge 1
```bash
python3 scripts/challenge1_ccd.py \
    --foundation checkpoints/foundation_best.pth \
    --data-dir data/challenge1 \
    --output submissions/challenge1.csv
```

### Step 6: Challenge 2
```bash
python3 scripts/challenge2_pfactors.py \
    --foundation checkpoints/foundation_best.pth \
    --data-dir data/challenge2 \
    --output submissions/challenge2.csv
```

---

## 📊 Success Metrics

### Foundation Model
- **Training Loss**: Should decrease from ~0.69 to <0.3
- **Validation Accuracy**: Target >60%, stretch >70%
- **Training Time**: <6 hours on CPU
- **Model Size**: ~10-20MB
- **Convergence**: Loss plateaus with early stopping

### Challenge 1 (CCD)
- **Pearson r**: >0.3 (competitive)
- **AUROC**: >0.7 (good discrimination)
- **Submission**: Valid CSV format

### Challenge 2 (P-factors)
- **Average Pearson r**: >0.2 (competitive)
- **Individual r's**: All >0.15
- **Submission**: Valid CSV format

---

## 🔧 Files to Create

### New Scripts Needed
1. `scripts/prepare_foundation_data.py` - Data preprocessing
2. `scripts/train_foundation_full.py` - Full training pipeline
3. `scripts/evaluate_foundation.py` - Model evaluation
4. `scripts/challenge1_ccd.py` - Challenge 1 implementation
5. `scripts/challenge2_pfactors.py` - Challenge 2 implementation
6. `scripts/monitor_training.py` - Real-time monitoring

### Configuration Files
1. `configs/foundation_training.yaml` - Training config
2. `configs/challenge1_config.yaml` - Challenge 1 config
3. `configs/challenge2_config.yaml` - Challenge 2 config

---

## ⚠️ Risk Mitigation

### Known Risks
1. **Training Time**: May take longer than 4 hours
   - **Mitigation**: Run overnight, implement checkpointing
   
2. **Convergence**: Model may not converge
   - **Mitigation**: Start with small learning rate, use early stopping
   
3. **Overfitting**: Limited data (10 subjects)
   - **Mitigation**: Strong regularization (dropout, weight decay)
   
4. **Memory**: CPU RAM constraints
   - **Mitigation**: Batch size 16, gradient accumulation if needed

---

## 🎯 Immediate Next Steps

**RIGHT NOW**:
1. ✅ Review this plan
2. ⭕ Create `scripts/train_foundation_full.py`
3. ⭕ Start foundation training
4. ⭕ Monitor progress

**Command to start**:
```bash
python3 scripts/train_foundation_full.py 2>&1 | tee logs/foundation_$(date +%Y%m%d_%H%M%S).log &
```

---

## 📈 Progress Tracking

**Overall P2 Progress**: 40% → Target: 100%

- [x] P2.0: GPU Safeguards (100%)
- [ ] P2.1: Data Acquisition (20% - have 10 subjects)
- [ ] P2.2: Foundation Training (40% - ready to train)
- [ ] P2.3: Challenge 1 (0%)
- [ ] P2.4: Challenge 2 (0%)
- [ ] P2.5: Optimization (0%)

**Target Completion**: End of Day (9 hours from now)

