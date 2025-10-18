# Multi-Head Self-Attention CNN for Challenge 1

## 📋 Overview

After discovering that the TCN model overfitted (Val Loss 0.0102 but test NRMSE 1.63), we propose enhancing the working CNN architecture with **Multi-Head Self-Attention** layers.

### Why Attention Instead of TCN?

**TCN Problems:**
- ✅ Good validation performance (0.0102)
- ❌ Terrible test performance (1.63)
- ❌ Overfitted to validation set
- ❌ 196K parameters (2.6x more than simple CNN)
- ❌ Complex architecture, hard to generalize

**Attention Benefits:**
- ✅ Captures long-range temporal dependencies
- ✅ More interpretable (attention weights show what model focuses on)
- ✅ Proven effective for sequential data
- ✅ Can be lightweight (only +4.7K params)
- ✅ Complements CNN's local pattern detection

---

## 🏗️ Architecture Options

### Option 1: LightweightAttentionCNN (Recommended)

**Parameters:** 79,489 (+4,736 from original 74,753)
**Strategy:** Minimal parameter increase, maximum benefit

```
Architecture Flow:
Input (129, 200)
    ↓
Conv Block 1: 129→32 channels  (local pattern extraction)
    ↓ (32, 100)
Conv Block 2: 32→64 channels
    ↓ (64, 50)
Multi-Head Self-Attention (4 heads)  ← NEW
    + Residual Connection
    ↓ (64, 50)
Conv Block 3: 64→96 channels  (feature refinement)
    ↓ (96, 25)
Global Average Pooling
    ↓ (96)
Regressor: 96→48→1
    ↓
Output (1)
```

**Key Features:**
- 4 attention heads for multi-scale temporal modeling
- Residual connection preserves original CNN features
- Lightweight: Only 6.3% parameter increase
- Dropout 0.1 in attention layer for regularization

### Option 2: AttentionCNN_ResponseTime (Full)

**Parameters:** 91,521 (+16,768 from original)
**Strategy:** More capacity, richer representations

```
Architecture Flow:
Input (129, 200)
    ↓
Conv Block 1: 129→32 channels
    ↓ (32, 100)
Conv Block 2: 32→64 channels
    ↓ (64, 50)
Multi-Head Self-Attention (4 heads)
    + Layer Normalization  ← Additional normalization
    + Residual Connection
    ↓ (64, 50)
Conv Block 3: 64→128 channels  ← Larger capacity
    ↓ (128, 25)
Global Average Pooling
    ↓ (128)
Regressor: 128→64→32→1  ← Deeper regressor
    ↓
Output (1)
```

**Key Features:**
- LayerNorm after attention for better training stability
- Larger final conv channels (128 vs 96)
- Deeper regressor (3 layers vs 2)
- 22.4% parameter increase

---

## 🔧 Multi-Head Self-Attention Details

### What It Does

```python
class MultiHeadSelfAttention(nn.Module):
    """
    Learns to attend to different parts of the temporal sequence
    
    For EEG signals, this means:
    - Early time points can influence later predictions
    - Model learns which time windows are most relevant
    - Captures dependencies beyond CNN's receptive field
    """
```

### How It Works

1. **Input:** (batch, 64 channels, 50 time steps)

2. **Transform to Q, K, V:**
   - Query (Q): "What am I looking for?"
   - Key (K): "What information do I have?"
   - Value (V): "What information should I pass forward?"

3. **Compute Attention:**
   ```
   Attention(Q, K, V) = softmax(QK^T / √d) V
   ```
   - Attention scores show which time steps attend to which others
   - Softmax ensures scores sum to 1
   - Scale by √d for numerical stability

4. **Multi-Head:**
   - 4 parallel attention mechanisms
   - Each head learns different temporal patterns
   - Head 1 might focus on early EEG response
   - Head 2 might focus on sustained activity
   - Head 3 might focus on late components
   - Head 4 might learn cross-frequency interactions

5. **Output:** Attended features + residual connection

---

## 📊 Expected Benefits

### 1. Better Temporal Modeling

**CNN Limitation:**
- Receptive field limited by kernel size and depth
- After 3 conv layers: ~15 time step receptive field
- EEG responses can span 100+ time steps

**Attention Solution:**
- Every time step can attend to every other time step
- Full 200-step receptive field
- Learns which time windows matter most

### 2. Reduced Overfitting

**Strategy:**
- Start from working CNN (NRMSE 1.00)
- Add minimal parameters (+6.3%)
- Attention dropout (0.1)
- Residual connections preserve learned features
- If attention doesn't help, residual ensures baseline performance

### 3. Interpretability

**Attention Weights:**
- Can visualize which time points model focuses on
- Helps understand EEG→response time relationship
- Debugging tool for model behavior

---

## 🎯 Training Strategy

### Phase 1: Conservative Training

```python
CONFIG = {
    'model_type': 'lightweight',
    'num_heads': 4,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'num_epochs': 50,
    'patience': 10
}
```

**Why Conservative:**
- Oct 16 CNN already achieves 1.00 NRMSE
- Goal: Match or beat 1.00, not revolutionary improvement
- Small learning rate to prevent divergence
- Early stopping to catch overfitting early

### Phase 2: Data Augmentation

If lightweight model works:
- Time shifting
- Channel dropout
- Gaussian noise
- Mixup/CutMix

### Phase 3: Ensemble (Optional)

- Ensemble attention CNN + original CNN
- Weighted average predictions
- Diversify model types for robustness

---

## 📈 Success Criteria

### Minimum Success
- **Val NRMSE ≤ 0.015** (match original CNN validation)
- **Test NRMSE ≤ 1.00** (match Oct 16 submission)
- No overfitting (val ≈ test performance)

### Good Success
- **Test NRMSE 0.90-0.95** (5-10% better than Oct 16)
- Consistent across validation folds
- Attention weights make sense

### Excellent Success
- **Test NRMSE < 0.90** (>10% improvement)
- Beats leaderboard competitors
- Clear interpretable attention patterns

---

## ⚠️ Risk Mitigation

### Risk 1: Overfitting (Like TCN)

**Mitigation:**
- Start with lightweight model (only +6.3% params)
- Strong dropout in attention (0.1)
- Early stopping (patience=10)
- Regular validation on held-out data
- If overfitting detected: reduce capacity, increase dropout

### Risk 2: Training Instability

**Mitigation:**
- LayerNorm after attention
- Residual connections
- AdamW optimizer (better than Adam)
- Learning rate scheduling
- Gradient clipping if needed

### Risk 3: Attention Doesn't Help

**Mitigation:**
- Residual connections ensure baseline performance
- Can fall back to original CNN if attention hurts
- Have backup: Oct 16 submission (1.32 NRMSE)

---

## 🚀 Implementation Plan

### Step 1: Setup (✅ Complete)
- [x] Create MultiHeadSelfAttention module
- [x] Create LightweightAttentionCNN
- [x] Create AttentionCNN_ResponseTime
- [x] Test architectures work
- [x] Verify parameter counts

### Step 2: Training (Next)
- [ ] Load Challenge 1 dataset
- [ ] Create train/val splits
- [ ] Train LightweightAttentionCNN
- [ ] Monitor validation performance
- [ ] Save best checkpoint

### Step 3: Validation (Next)
- [ ] Evaluate on validation set
- [ ] Compare to original CNN (Val Loss ~0.015)
- [ ] Check for overfitting
- [ ] Visualize attention weights

### Step 4: Decision (Next)
- [ ] If val performance good: create submission
- [ ] If overfitting: tune hyperparameters
- [ ] If worse than CNN: analyze why
- [ ] Document findings

### Step 5: Submission (If successful)
- [ ] Create new submission.py with attention model
- [ ] Package with trained weights
- [ ] Test locally
- [ ] Upload to Codabench
- [ ] Compare test NRMSE to Oct 16 (1.00)

---

## 📝 Code Files Created

### models_with_attention.py
```python
# Multi-Head Self-Attention module
class MultiHeadSelfAttention(nn.Module)

# Full attention model (91K params)
class AttentionCNN_ResponseTime(nn.Module)

# Lightweight model (79K params) - RECOMMENDED
class LightweightAttentionCNN(nn.Module)
```

### train_attention_model.py
```python
# Training configuration
CONFIG = {...}

# Model setup
model = LightweightAttentionCNN(num_heads=4)

# Training loop template
# Includes: AdamW, ReduceLROnPlateau, Early Stopping
```

---

## 💡 Key Insights

### Why This Could Work

1. **Proven Architecture Pattern**
   - Vision: ResNet + Attention = Better image recognition
   - NLP: BERT = Transformers (pure attention)
   - Audio: CNN + Attention = Better speech recognition
   - EEG: Should benefit from temporal attention

2. **EEG Characteristics**
   - Event-Related Potentials span 100-500ms
   - At 100Hz, that's 10-50 time steps
   - Need long-range dependencies
   - Attention naturally models this

3. **Conservative Approach**
   - Small parameter increase (+6.3%)
   - Residual connections as safety net
   - Start from working baseline (1.00 NRMSE)
   - Easy to revert if doesn't work

### Why This Might Not Work

1. **EEG is Noisy**
   - Attention might focus on noise
   - Need strong regularization
   - May need more data

2. **Simple is Sometimes Better**
   - Oct 16 CNN with 75K params works
   - More complexity != better performance
   - Risk of overfitting like TCN

3. **Competition Data Characteristics**
   - Maybe simple patterns dominate
   - CNN's local receptive field sufficient
   - Attention adds complexity without benefit

---

## 🎓 Recommendation

### Start Here: LightweightAttentionCNN

**Rationale:**
1. Minimal risk (+6.3% params)
2. Residual connection preserves baseline
3. If it fails, easy to debug
4. If it works, can try full model next

**Training Steps:**
```bash
# 1. Prepare data
python prepare_challenge1_data.py

# 2. Train lightweight model
python train_attention_model.py

# 3. Evaluate
python evaluate_attention_model.py

# 4. If successful, create submission
python create_attention_submission.py
```

**Decision Tree:**
```
Train LightweightAttentionCNN
    ↓
Val NRMSE ≤ 0.015?
    ├─ Yes → Test on validation set
    │   ↓
    │   Test NRMSE ≤ 1.00?
    │       ├─ Yes → CREATE SUBMISSION ✅
    │       └─ No → Tune hyperparameters or revert
    │
    └─ No → Overfitting detected
        ↓
        Increase dropout, reduce capacity, or revert to Oct 16 CNN
```

---

## 📦 Current Status

✅ **Architecture Created**
- LightweightAttentionCNN: 79,489 params
- AttentionCNN_ResponseTime: 91,521 params
- Both tested and working

✅ **Training Setup Ready**
- AdamW optimizer configured
- ReduceLROnPlateau scheduler
- Early stopping with patience=10
- CUDA enabled

⏳ **Next Steps**
- Load Challenge 1 dataset
- Train LightweightAttentionCNN
- Validate performance
- Compare to Oct 16 baseline (1.00 NRMSE)

🎯 **Goal**
- Test NRMSE ≤ 1.00 (match or beat Oct 16)
- If successful: New submission
- If not: Revert to Oct 16 CNN (1.32 overall)

---

**Created:** October 18, 2025
**Status:** Ready to train
**Priority:** MEDIUM (Oct 16 submission is working, this is optimization)

