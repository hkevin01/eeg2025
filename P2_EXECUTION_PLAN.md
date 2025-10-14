# P2 Task Execution Plan

**Started**: October 14, 2025  
**Status**: 🟡 In Progress

---

## Task Overview

| # | Task | Time | Status | Progress |
|---|------|------|--------|----------|
| P2.1 | Scale Data Acquisition | 2-3 days | �� In Progress | 10% |
| P2.2 | Train Foundation Model | 5-7 days | ⭕ Pending | 0% |
| P2.3 | Challenge 1 Implementation | 3-4 days | ⭕ Pending | 0% |
| P2.4 | Challenge 2 Implementation | 3-4 days | ⭕ Pending | 0% |
| P2.5 | Model Optimization | 2-3 days | ⭕ Pending | 0% |

---

## P2.1: Scale Data Acquisition (🟡 In Progress - 10%)

### Goal
Download 50-100 subjects from HBN dataset for robust model training

### Current Status
- ✅ 2 subjects downloaded (sub-NDARAA075AMK, sub-NDARAA117NEJ)
- 🟡 Working on scaling to 50+ subjects
- ⭕ S3 direct access having connectivity issues

### Approach
Since S3 direct download is challenging, using alternative strategy:
1. Start with existing 2 subjects for initial development
2. Download additional subjects in background
3. Parallelize: develop models while data acquisition continues

### Actions Taken
- Verified existing 2 subjects are valid and loadable
- Confirmed BIDS structure compliance
- Identified S3 connectivity constraints

### Next Steps
1. Create batch download script with retry logic
2. Download 10 subjects at a time
3. Verify each batch before continuing
4. Meanwhile, start foundation model development with existing data

---

## P2.2: Train Foundation Model (⭕ Starting Now)

### Goal
Train advanced transformer-based foundation model on available data

### Strategy
Start training with 2 subjects, scale up as more data arrives

### Implementation Plan

1. **Model Architecture** (2 hours)
   - Review existing `advanced_foundation_model.py`
   - Configure for current data size
   - Set up training pipeline

2. **Training Configuration** (1 hour)
   - Create training config file
   - Set hyperparameters for small dataset
   - Configure checkpointing and logging

3. **Initial Training** (4-8 hours)
   - Train on 2 subjects as baseline
   - Monitor metrics and convergence
   - Identify any issues early

4. **Scale Up** (ongoing)
   - Retrain as more subjects become available
   - Compare performance improvements
   - Adjust architecture if needed

### Files to Create/Modify
- `config/foundation_model_small.yaml` - Config for 2-subject training
- `scripts/train_foundation_model.py` - Training script
- `scripts/monitor_training.py` - Real-time monitoring

---

## P2.3: Challenge 1 Implementation (⭕ Pending)

### Goal
Implement SuS → CCD cross-task transfer learning

### Prerequisites
- P2.2 foundation model trained
- Sufficient training data available

### Implementation Plan
1. Load pretrained foundation model
2. Implement task-specific adaptation layers
3. Train transfer learning pipeline
4. Evaluate on held-out CCD data
5. Optimize for competition metrics

### Target Metrics
- Response Time: Pearson r > 0.3
- Success Rate: AUROC > 0.7

---

## P2.4: Challenge 2 Implementation (⭕ Pending)

### Goal
Predict psychopathology factors (P-factor, internalizing, externalizing, attention)

### Prerequisites
- P2.2 foundation model trained
- Clinical labels loaded and preprocessed

### Implementation Plan
1. Load CBCL scores from participants.tsv
2. Implement multi-output regression head
3. Train psychopathology prediction model
4. Evaluate Pearson correlation per factor
5. Handle missing labels appropriately

### Target Metrics
- Average Pearson r > 0.2 across all factors
- P-factor correlation > 0.25
- Individual factor correlations > 0.15

---

## P2.5: Model Optimization (⭕ Pending)

### Goal
Reduce inference latency from 186ms → <50ms (3.7x speedup)

### Current Baseline
- Random model: 186ms average
- Target: <50ms average, <75ms P95

### Optimization Strategies
1. **Model Quantization**
   - FP32 → FP16 (2x speedup expected)
   - INT8 if accuracy permits (4x speedup)

2. **Operator Fusion**
   - Fuse consecutive operations
   - Reduce memory transfers

3. **TensorRT Compilation**
   - Convert model to TensorRT
   - GPU-specific optimizations

4. **Batch Optimization**
   - Optimize batch size for latency
   - Implement efficient batching

### Implementation Plan
1. Profile current model (identify bottlenecks)
2. Apply quantization (test accuracy impact)
3. Implement operator fusion
4. Test TensorRT compilation
5. Benchmark and iterate

---

## Execution Timeline

### Week 1 (Oct 14-20, 2025)

**Days 1-2 (Oct 14-15)**: Data + Foundation Model Setup
- ✅ P2.1: Download 10 more subjects (total: 12)
- ✅ P2.2: Train foundation model on available data
- ✅ P2.2: Monitor training and adjust

**Days 3-4 (Oct 16-17)**: Continue Training + Challenge 1
- ✅ P2.1: Download 20 more subjects (total: 32)
- ✅ P2.2: Continue/complete foundation training
- ✅ P2.3: Start Challenge 1 implementation

**Days 5-7 (Oct 18-20)**: Challenge 2 + Optimization
- ✅ P2.1: Download remaining subjects (total: 50+)
- ✅ P2.4: Implement Challenge 2
- ✅ P2.5: Start model optimization

### Week 2 (Oct 21-24, 2025)

**Days 8-10 (Oct 21-24)**: Complete & Polish
- ✅ P2.2: Retrain on full dataset
- ✅ P2.3: Optimize Challenge 1 metrics
- ✅ P2.4: Optimize Challenge 2 metrics  
- ✅ P2.5: Complete optimization work
- ✅ All: Integration testing and validation

---

## Success Criteria

### P2.1 Success
- [ ] 50+ subjects downloaded and verified
- [ ] All data BIDS-compliant
- [ ] Train/val/test splits created (60/20/20)
- [ ] Data quality metrics documented

### P2.2 Success
- [ ] Foundation model trained without errors
- [ ] Validation metrics show improvement over baseline
- [ ] Model checkpoints saved properly
- [ ] Training logs complete and interpretable

### P2.3 Success
- [ ] Transfer learning pipeline works end-to-end
- [ ] Response time: Pearson r > 0.3
- [ ] Success rate: AUROC > 0.7
- [ ] Results reproducible

### P2.4 Success
- [ ] All 4 P-factors predicted
- [ ] Average Pearson r > 0.2
- [ ] Individual correlations positive
- [ ] Missing data handled properly

### P2.5 Success
- [ ] Inference latency < 50ms average
- [ ] P95 latency < 75ms
- [ ] Accuracy maintained (< 2% degradation)
- [ ] Optimization documented

---

## Current Focus

**RIGHT NOW**: Start P2.2 (Foundation Model Training)

Rationale:
- We have 2 validated subjects ready for training
- Can start model development immediately
- Data acquisition can continue in parallel
- Early training helps identify issues
- Can retrain on larger dataset later

**Next Command**:
```bash
# Create training config for small dataset
cat > config/foundation_model_small.yaml << 'YAML'
model:
  name: advanced_foundation_model
  n_channels: 128
  sampling_rate: 500
  window_size: 2.0
  n_attention_heads: 8
  n_layers: 6
  hidden_dim: 512
  dropout: 0.1

training:
  batch_size: 8
  max_epochs: 50
  learning_rate: 0.0001
  warmup_steps: 100
  gradient_clip: 1.0
  
data:
  subjects: 2
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
YAML

# Start training
python scripts/train_foundation_model.py --config config/foundation_model_small.yaml
```

---

**Last Updated**: October 14, 2025  
**Next Review**: October 15, 2025 (daily updates)
