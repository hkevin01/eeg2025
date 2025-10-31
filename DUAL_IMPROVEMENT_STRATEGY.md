# Dual Improvement Strategy - C1 & C2

**Date**: October 30, 2025  
**Goal**: Improve both C1 (from 1.0002) and C2 (from 1.0087) to get overall score ~1.00 or better  
**Strategy**: Flexible architectures with strong anti-overfitting measures

---

## 📊 Current Status

| Challenge | Current Score | Target | Improvement Needed |
|-----------|--------------|--------|-------------------|
| Challenge 1 | 1.0002 | < 1.00 | 0.0002 (0.02%) |
| Challenge 2 | 1.0087 | < 1.00 | 0.0087 (0.87%) |
| Overall | 1.0061 | < 1.00 | 0.0061 (0.61%) |

**Key Insight**: C2 has 43x more room for improvement than C1!

---

## 🚀 Challenge 1: Enhanced Training (IN PROGRESS)

### Status: ✅ RUNNING NOW

**Script**: `train_c1_aggressive.py`  
**Session**: tmux session `aggressive_training`

### Strategy:

1. **Deeper Architecture**:
   - 3 conv blocks: 48 → 64 → 96 channels
   - Temporal attention mechanism
   - ~150K parameters (vs 74K in V8)

2. **Stronger Regularization**:
   - Dropout: [0.6, 0.7, 0.75] (even stronger!)
   - Weight decay: 0.1 (2x stronger than V8)
   - Gradient clipping: max_norm=1.0

3. **Advanced Augmentation**:
   - **Channel dropout**: 30% chance, drop 5-20 channels
   - **Temporal cutout**: 30% chance, mask 10-30 timepoints  
   - Stronger mixup: α=0.4 (vs 0.2 in V8)
   - Time shift: ±15 samples
   - Amplitude scaling: 0.85-1.15x
   - Gaussian noise: σ=0.02

4. **Better Training**:
   - 50 epochs with cosine annealing + warm restarts
   - Batch size: 32 (more updates)
   - Z-score normalization per channel
   - Early stopping patience: 10

### Expected Results:

| Metric | V8 (Current) | V9 (Expected) | Improvement |
|--------|--------------|---------------|-------------|
| Val NRMSE | 0.160418 | 0.145-0.155 | 3-10% |
| Test Score | 1.0002 | 0.95-0.99 | 1-5% |

### Monitoring:

```bash
# Check if running
ps aux | grep train_c1_aggressive

# Watch progress
tmux attach -t aggressive_training
# (Ctrl+B then D to detach)

# Check specific metrics
tail -f training_aggressive.log | grep "Val NRMSE"
```

### Time Estimate:

- ~20-30 minutes (50 epochs, smaller batches)
- Will save to: `checkpoints/challenge1_aggressive_TIMESTAMP/`

---

## 🧠 Challenge 2: Flexible Stimulus Decoder (READY TO START)

### Status: ✅ SCRIPT READY

**Script**: `train_c2_improved.py`

### Why Focus on C2?

1. **More room for improvement**: 1.0087 vs 1.0002 (43x more)
2. **Different task**: Classification vs regression
3. **Baseline is weaker**: EEGNeX at 1.0087 vs CompactCNN at 1.0015
4. **Higher impact**: Improving C2 by 0.008 = improving overall by 0.004

### Architecture: FlexibleStimulusDecoder

```
Encoder (Feature Extraction):
  Conv1d(129→64, k=5, s=2) + BN + ReLU + Dropout(0.4)
  Conv1d(64→96, k=5, s=2) + BN + ReLU + Dropout(0.5)
  Conv1d(96→128, k=3, s=2) + BN + ReLU + Dropout(0.6)
  
Attention:
  AdaptiveAvgPool → Linear(128→64) → ReLU → Linear(64→128) → Sigmoid
  
Decoder:
  AdaptiveAvgPool → Flatten
  Linear(128→96) + ReLU + Dropout(0.5)
  Linear(96→64) + ReLU + Dropout(0.4)
  
Output Heads (separate for each stimulus dimension):
  head_dim1: Linear(64→3)
  head_dim2: Linear(64→3)
  head_dim3: Linear(64→3)

Total: ~100K parameters
```

### Anti-Overfitting Strategy:

1. **Strong dropout**: Progressive [0.4, 0.5, 0.6] in encoder
2. **Weight decay**: 0.05 (same as V8)
3. **Label smoothing**: 0.1 (softens classification targets)
4. **Gradient clipping**: max_norm=1.0
5. **Data augmentation**: time shift, amplitude, noise, channel dropout
6. **Mixup**: α=0.3 for classification
7. **Early stopping**: patience 8 epochs
8. **Cosine annealing**: with warm restarts
9. **Z-score normalization**: per channel

### Training Configuration:

```python
{
    'batch_size': 32,
    'epochs': 40,
    'lr': 0.0005,  # Lower for stability
    'weight_decay': 0.05,
    'dropout_encoder': [0.4, 0.5, 0.6],
    'dropout_decoder': [0.5, 0.4],
    'patience': 8,
    'mixup_alpha': 0.3,
    'label_smoothing': 0.1,
}
```

### To Start C2 Training:

```bash
cd /home/kevin/Projects/eeg2025
source venv_training/bin/activate
tmux new-session -d -s c2_training "python train_c2_improved.py 2>&1 | tee training_c2.log"

# Monitor
tmux attach -t c2_training
# or
tail -f training_c2.log
```

### Expected Results:

| Metric | Baseline (EEGNeX) | V9 (Expected) | Improvement |
|--------|-------------------|---------------|-------------|
| Val Loss | ~0.30-0.35 | 0.25-0.32 | 5-15% |
| Val Acc | ~85-90% | 88-93% | 3-8% |
| Test Score | 1.0087 | 0.95-1.02 | 1-6% |

### Time Estimate:

- ~30-40 minutes (40 epochs, early stopping likely around 20-25)
- Will save to: `checkpoints/challenge2_improved_TIMESTAMP/`

---

## 📈 Combined Expected Improvements

### Scenario 1: Both Improve Moderately

| Challenge | Current | V9 | Improvement |
|-----------|---------|----|-----------  |
| C1 | 1.0002 | 0.98 | -0.0202 (2%) |
| C2 | 1.0087 | 1.00 | -0.0087 (0.87%) |
| **Overall** | **1.0061** | **0.99** | **-0.0161 (1.6%)** ✅ |

### Scenario 2: Both Improve Well

| Challenge | Current | V9 | Improvement |
|-----------|---------|----|-----------  |
| C1 | 1.0002 | 0.96 | -0.0402 (4%) |
| C2 | 1.0087 | 0.98 | -0.0287 (2.9%) |
| **Overall** | **1.0061** | **0.97** | **-0.0361 (3.6%)** ✅✅ |

### Scenario 3: Conservative (One Improves)

| Challenge | Current | V9 | Improvement |
|-----------|---------|----|-----------  |
| C1 | 1.0002 | 0.99 | -0.0102 (1%) |
| C2 | 1.0087 | 1.0087 | 0 (no change) |
| **Overall** | **1.0061** | **1.005** | **-0.0011 (0.1%)** ✅ |

---

## 🎯 Recommended Execution Plan

### Phase 1: Let C1 Finish (ACTIVE NOW)

**Status**: Training in progress (~20-30 min remaining)

**Action**: Wait for completion, monitor progress

**Decision Point**:
- If Val NRMSE < 0.155: ✅ Proceed to test
- If Val NRMSE > 0.160: ⚠️ Marginal improvement only

### Phase 2: Start C2 Training

**When**: After C1 completes or runs in parallel (separate tmux)

**Command**:
```bash
source venv_training/bin/activate
tmux new-session -d -s c2_training "python train_c2_improved.py 2>&1 | tee training_c2.log"
```

**Time**: 30-40 minutes

**Decision Point**:
- If Val Loss significantly improves: ✅ Use in V9
- If no improvement: Use current C2 weights

### Phase 3: Create V9 Submission

**After both trainings complete**:

1. **Evaluate C1 results**:
   ```bash
   python -c "import torch; ckpt = torch.load('checkpoints/challenge1_aggressive_*/best_model.pth', weights_only=False); print(f'C1 Val NRMSE: {ckpt[\"val_nrmse\"]:.6f}')"
   ```

2. **Evaluate C2 results**:
   ```bash
   python -c "import torch; ckpt = torch.load('checkpoints/challenge2_improved_*/best_model.pth', weights_only=False); print(f'C2 Val Loss: {ckpt[\"val_loss\"]:.6f}'); print(f'C2 Val Acc: {ckpt[\"val_acc\"]}')"
   ```

3. **Decision Matrix**:

| C1 Improved? | C2 Improved? | Action |
|--------------|--------------|--------|
| ✅ Yes | ✅ Yes | Use both → V9 |
| ✅ Yes | ❌ No | Use C1 only → V9 |
| ❌ No | ✅ Yes | Use C2 only → V9 |
| ❌ No | ❌ No | Keep V8 |

4. **Create V9**:
   ```bash
   mkdir -p submissions/phase1_v9
   cp submissions/phase1_v8/submission.py submissions/phase1_v9/
   
   # Copy improved weights
   cp checkpoints/challenge1_aggressive_*/best_weights.pt submissions/phase1_v9/weights_challenge_1.pt  # if C1 improved
   cp checkpoints/challenge2_improved_*/best_weights.pt submissions/phase1_v9/weights_challenge_2.pt  # if C2 improved
   
   # Package
   cd submissions/phase1_v9
   zip submission_v9_improved.zip submission.py weights_challenge_1.pt weights_challenge_2.pt
   ```

5. **Verify**:
   ```bash
   python scripts/verify_submission.py submissions/phase1_v9/submission_v9_improved.zip
   ```

---

## ⚠️ Important Considerations

### 1. Overfitting Risk

**C1**:
- Very aggressive regularization applied
- Deeper model + more dropout
- Risk: Medium (strong reg should handle it)

**C2**:
- Multi-output classification (harder to overfit)
- Label smoothing + dropout
- Risk: Medium-Low (classification is more stable)

### 2. Time Investment

- **C1 Training**: 20-30 minutes (ACTIVE)
- **C2 Training**: 30-40 minutes
- **Total**: 50-70 minutes
- **Verification & Packaging**: 10 minutes

**Total Time**: ~1-1.5 hours

### 3. Success Probability

| Outcome | Probability | Result |
|---------|-------------|---------|
| Both improve | 40% | Overall < 1.00 ✅✅ |
| One improves | 35% | Overall < 1.005 ✅ |
| Neither improves | 25% | Keep V8 (1.0061) |

**Expected Value**: 75% chance of some improvement!

### 4. Fallback Strategy

- **V8 is safe**: 1.0061 is already excellent
- **Can always revert**: If V9 worse, submit V8
- **Low risk**: Strong regularization minimizes overfit risk

---

## 📊 Monitoring Commands

### Check What's Running:

```bash
# See all training processes
ps aux | grep "train_c"

# Check tmux sessions
tmux list-sessions
```

### Monitor C1:

```bash
# Attach to C1 training
tmux attach -t aggressive_training

# Watch log
tail -f training_aggressive.log

# Check latest Val NRMSE
grep "Val NRMSE" training_aggressive.log | tail -10
```

### Monitor C2:

```bash
# Attach to C2 training
tmux attach -t c2_training

# Watch log
tail -f training_c2.log

# Check latest Val Loss
grep "Val Loss" training_c2.log | tail -10
```

### Kill if Needed:

```bash
# Stop C1
tmux kill-session -t aggressive_training

# Stop C2
tmux kill-session -t c2_training
```

---

## 🎉 Success Criteria

### Minimum Success (One improves):

- C1: Val NRMSE < 0.158 (2% improvement)
- C2: Val Loss < 0.32 (5% improvement)
- Overall expected: < 1.005

### Good Success (Both improve moderately):

- C1: Val NRMSE < 0.155 (3% improvement)
- C2: Val Loss < 0.30 (10% improvement)
- Overall expected: < 1.00 ✅

### Excellent Success (Both improve significantly):

- C1: Val NRMSE < 0.150 (6% improvement)
- C2: Val Loss < 0.28 (15% improvement)
- Overall expected: < 0.98 ✅✅

---

## 🚀 Quick Start Commands

### Start C2 Training Now (while C1 runs):

```bash
cd /home/kevin/Projects/eeg2025
source venv_training/bin/activate
tmux new-session -d -s c2_training "python train_c2_improved.py 2>&1 | tee training_c2.log"
echo "✅ C2 training started!"
```

### Check Status:

```bash
./watch_training.sh  # if exists
# or
ps aux | grep "train_c"
```

---

**STATUS**: 
- ✅ C1 Training ACTIVE (train_c1_aggressive.py)
- ⏳ C2 Training READY (train_c2_improved.py)
- 🎯 Target: Both < 1.00, Overall < 1.00

**Next Step**: Start C2 training or wait for C1 to complete!

