# Challenge 1 Improvement Roadmap - Target Score: 0.9 or Below

**Created:** October 27, 2025, 6:30 PM  
**Current Score:** 1.0020 (baseline: 1.0015)  
**Target Score:** ≤ 0.9  
**Required Improvement:** ~11% reduction in NRMSE  
**Status:** 🚀 ACTIVE DEVELOPMENT

---

## Executive Summary

**Goal:** Reduce Challenge 1 NRMSE from 1.0020 to ≤ 0.9 (~11% improvement)

**Strategy:** Multi-pronged approach combining:
1. **New Architectures** (Transformer, GNN, Hybrid)
2. **Fixed Training** (better data, better loss, cross-validation)
3. **Data Analysis** (distribution comparison, domain adaptation)

**Timeline:** 2-3 days intensive development

---

## Phase 1: Data Analysis & Understanding (Priority: CRITICAL)

### 1.1 Dataset Distribution Analysis

**Objective:** Understand why validation ≠ test performance

**Tasks:**
```markdown
- [ ] Load all available datasets (R1-R7)
- [ ] Compare R4 (current validation) vs other R sets
- [ ] Analyze feature statistics:
  - [ ] Channel-wise mean/std
  - [ ] Power spectral density
  - [ ] Temporal patterns
  - [ ] Response time distributions
- [ ] Identify domain shift between R4 and test
- [ ] Visualize differences (plots, histograms)
```

**Code Location:** `src/analysis/dataset_distribution_analysis.py`

**Expected Output:**
- Distribution comparison plots
- Statistical tests (KS test, t-test)
- Recommendations for which R sets to use

### 1.2 Response Time Prediction Analysis

**Objective:** Understand the prediction task better

**Tasks:**
```markdown
- [ ] Analyze response time distribution
- [ ] Check correlation between EEG features and response time
- [ ] Identify most predictive channels
- [ ] Frequency band analysis (delta, theta, alpha, beta, gamma)
- [ ] Temporal window importance (early vs late)
```

**Code Location:** `src/analysis/response_time_analysis.py`

**Expected Output:**
- Feature importance rankings
- Channel selection recommendations
- Optimal preprocessing strategy

---

## Phase 2: Architecture Experiments (Priority: HIGH)

### 2.1 Transformer-Based Models

**Rationale:** Attention mechanisms may capture temporal dependencies better

**Architectures to Try:**

#### A. Pure Transformer
```python
# src/models/transformers/eeg_transformer.py

class EEGTransformer(nn.Module):
    """
    Multi-head self-attention over temporal dimension
    - Input: (batch, 129 channels, 200 timepoints)
    - Positional encoding for temporal order
    - Multi-head attention (8 heads)
    - Feed-forward layers
    """
```

**Tasks:**
```markdown
- [ ] Implement EEGTransformer architecture
- [ ] Add positional encoding
- [ ] Experiment with attention heads (4, 8, 16)
- [ ] Train on R1-R7 combined
- [ ] Evaluate on held-out set
```

#### B. Vision Transformer (ViT) Adaptation
```python
# src/models/transformers/vit_eeg.py

class ViTEEG(nn.Module):
    """
    Treat EEG as image: 129 channels × 200 timepoints
    - Patch embedding (e.g., 10x10 patches)
    - ViT architecture
    - Classification head → regression
    """
```

**Tasks:**
```markdown
- [ ] Adapt ViT for EEG regression
- [ ] Experiment with patch sizes
- [ ] Compare to pure Transformer
```

#### C. Conformer (CNN + Transformer Hybrid)
```python
# src/models/hybrid/conformer_eeg.py

class ConformerEEG(nn.Module):
    """
    Best of both worlds:
    - Conv layers for local patterns
    - Transformer for global dependencies
    - Squeeze-and-excitation for channel attention
    """
```

**Tasks:**
```markdown
- [ ] Implement Conformer architecture
- [ ] Balance conv vs attention layers
- [ ] Add channel attention mechanism
```

### 2.2 Graph Neural Networks

**Rationale:** Model spatial relationships between EEG channels

**Architectures to Try:**

#### A. GCN for EEG
```python
# src/models/gnn/eeg_gcn.py

class EEGGCN(nn.Module):
    """
    Graph Convolutional Network:
    - Nodes: 129 EEG channels
    - Edges: Spatial proximity or correlation
    - Graph convolutions capture channel relationships
    - Temporal pooling → regression
    """
```

**Tasks:**
```markdown
- [ ] Define channel graph (spatial layout)
- [ ] Implement GCN layers
- [ ] Try different graph constructions:
  - [ ] Spatial proximity
  - [ ] Correlation-based
  - [ ] Learnable adjacency
```

#### B. Graph Attention Networks (GAT)
```python
# src/models/gnn/eeg_gat.py

class EEGGAT(nn.Module):
    """
    Attention over channel graph:
    - Learn which channels to attend to
    - More flexible than fixed adjacency
    """
```

**Tasks:**
```markdown
- [ ] Implement GAT for EEG
- [ ] Visualize learned attention weights
- [ ] Compare to fixed GCN
```

### 2.3 State-Space Models (Mamba/S4)

**Rationale:** Efficient long-range temporal modeling

```python
# src/models/ssm/mamba_eeg.py

class MambaEEG(nn.Module):
    """
    State-space model for EEG:
    - Efficient O(N) complexity
    - Long-range dependencies
    - Better than Transformer for long sequences
    """
```

**Tasks:**
```markdown
- [ ] Research Mamba/S4 implementations
- [ ] Adapt for EEG regression
- [ ] Compare efficiency vs Transformer
```

---

## Phase 3: Training Improvements (Priority: HIGH)

### 3.1 Better Dataset Strategy

**Current Issue:** Training on R4 only (16,604 samples)

**Solutions:**

#### A. Use All R1-R7 Data
```markdown
- [ ] Combine all R sets (verify allowed by rules)
- [ ] Estimate total samples: ~100k+
- [ ] Create stratified split
- [ ] Use cross-validation to find test-like validation
```

#### B. Smart Data Selection
```markdown
- [ ] Analyze which R sets are most similar to test
- [ ] Weight samples by similarity
- [ ] Use domain adaptation techniques
```

**Code Location:** `src/dataio/multi_dataset_loader.py`

### 3.2 Better Loss Function

**Current Issue:** Training on Pearson+NRMSE, evaluated on NRMSE only

**Solution:** Train directly on competition metric

```python
# src/training/losses/nrmse_loss.py

class NRMSELoss(nn.Module):
    """
    Normalized Root Mean Square Error (competition metric)
    
    NRMSE = sqrt(MSE(y_pred, y_true)) / std(y_true)
    """
    def forward(self, y_pred, y_true):
        mse = F.mse_loss(y_pred, y_true)
        std = torch.std(y_true)
        return torch.sqrt(mse) / (std + 1e-8)
```

**Tasks:**
```markdown
- [ ] Implement NRMSE loss exactly as competition
- [ ] Remove Pearson correlation from loss
- [ ] Train models with NRMSE only
- [ ] Compare validation metrics
```

### 3.3 Cross-Validation Strategy

**Current Issue:** Single validation set (R4) not representative

**Solution:** K-fold cross-validation to find robust models

```python
# src/training/cross_validation.py

class CrossValidator:
    """
    5-fold cross-validation:
    - Split all data into 5 folds
    - Train on 4, validate on 1
    - Ensemble predictions from 5 models
    - Select fold that best matches test distribution
    """
```

**Tasks:**
```markdown
- [ ] Implement 5-fold CV
- [ ] Train 5 models per architecture
- [ ] Ensemble predictions (average, weighted)
- [ ] Identify which fold is most test-like
```

### 3.4 Training Hyperparameters

**Experiments to Run:**

```markdown
- [ ] Learning rates: [1e-3, 5e-4, 1e-4, 5e-5]
- [ ] Batch sizes: [16, 32, 64, 128]
- [ ] Optimizers: [AdamW, Adam, SGD with momentum]
- [ ] Schedulers: [CosineAnnealing, ReduceLROnPlateau, OneCycleLR]
- [ ] Regularization:
  - [ ] Dropout: [0.0, 0.1, 0.2, 0.3]
  - [ ] Weight decay: [0.0, 1e-4, 1e-3]
  - [ ] Label smoothing
  - [ ] Mixup augmentation
```

**Code Location:** `src/training/hyperparameter_search.py`

---

## Phase 4: Data Augmentation & Preprocessing (Priority: MEDIUM)

### 4.1 Signal Preprocessing

**Current:** Raw 129 channels

**Improvements:**

```markdown
- [ ] Bandpass filtering (0.5-50 Hz)
- [ ] Notch filter (50/60 Hz line noise)
- [ ] ICA for artifact removal
- [ ] Channel standardization (per-channel z-score)
- [ ] Temporal downsampling (if beneficial)
```

**Code Location:** `src/dataio/preprocessing.py`

### 4.2 Data Augmentation

**Techniques to Try:**

```markdown
- [ ] Time shifting (random offset)
- [ ] Time stretching (speed up/slow down)
- [ ] Magnitude scaling (random gain)
- [ ] Adding noise (gaussian, realistic EEG noise)
- [ ] Channel dropout (randomly drop channels)
- [ ] Temporal masking (mask random time segments)
- [ ] Mixup (interpolate between samples)
```

**Code Location:** `src/dataio/augmentation.py`

### 4.3 Feature Engineering

**Manual Features to Extract:**

```markdown
- [ ] Power spectral density (delta, theta, alpha, beta, gamma)
- [ ] Band power ratios
- [ ] Hjorth parameters (activity, mobility, complexity)
- [ ] Sample entropy
- [ ] Fractal dimension
- [ ] Coherence between channels
```

**Code Location:** `src/features/eeg_features.py`

**Usage:** Concatenate with learned features

---

## Phase 5: Ensemble Strategies (Priority: MEDIUM)

### 5.1 Diverse Model Ensemble

**Current Issue:** Ensemble of 3× same architecture didn't help

**Solution:** Ensemble DIVERSE models

```python
# src/models/ensemble/diverse_ensemble.py

class DiverseEnsemble:
    """
    Combine predictions from:
    1. CompactCNN (current best)
    2. Transformer
    3. GNN
    4. Conformer
    5. Mamba
    
    Weight by validation performance or learn weights
    """
```

**Tasks:**
```markdown
- [ ] Train 5 different architectures
- [ ] Ensemble with equal weights
- [ ] Ensemble with performance-weighted
- [ ] Ensemble with learned stacking
- [ ] Test-time augmentation (predict multiple times, average)
```

### 5.2 Stacking & Meta-Learning

```python
# src/models/ensemble/stacking.py

class StackingEnsemble:
    """
    Level 0: Base models (CNN, Transformer, GNN, etc.)
    Level 1: Meta-learner (XGBoost, LightGBM, Linear)
    
    Train meta-learner on base model predictions
    """
```

**Tasks:**
```markdown
- [ ] Collect predictions from all base models
- [ ] Train meta-learner (XGBoost, LightGBM)
- [ ] Use both predictions + features
- [ ] Cross-validation to avoid overfitting
```

---

## Phase 6: Domain Adaptation (Priority: MEDIUM)

### 6.1 Distribution Matching

**Objective:** Make validation distribution match test

**Techniques:**

```markdown
- [ ] Maximum Mean Discrepancy (MMD) loss
- [ ] Correlation alignment (CORAL)
- [ ] Domain adversarial training
```

**Code Location:** `src/training/domain_adaptation.py`

### 6.2 Test-Time Adaptation

```markdown
- [ ] Predict on test samples
- [ ] Update batch norm statistics
- [ ] Fine-tune last layer only
- [ ] Self-supervised adaptation
```

---

## Implementation Plan - Week 1

### Day 1 (Today - Oct 27 Evening):
```markdown
✅ Phase 1.1: Dataset Distribution Analysis
  - [ ] Implement distribution comparison script
  - [ ] Load R1-R7 datasets
  - [ ] Generate comparison plots
  - [ ] Identify domain shift
  
✅ Phase 1.2: Response Time Analysis
  - [ ] Feature correlation analysis
  - [ ] Channel importance
  - [ ] Frequency band analysis
```

**Deliverable:** `docs/C1_DATA_ANALYSIS_REPORT.md`

### Day 2 (Oct 28):
```markdown
✅ Phase 2.1: Transformer Implementation
  - [ ] Implement EEGTransformer
  - [ ] Train on combined R1-R7
  - [ ] Evaluate vs baseline
  
✅ Phase 3.1: Better Dataset
  - [ ] Create multi-dataset loader
  - [ ] Implement cross-validation
```

**Deliverable:** Transformer model checkpoint + CV results

### Day 3 (Oct 29):
```markdown
✅ Phase 2.2: GNN Implementation
  - [ ] Implement EEGGCN
  - [ ] Define channel graph
  - [ ] Train and evaluate
  
✅ Phase 3.2: Better Loss Function
  - [ ] Implement pure NRMSE loss
  - [ ] Retrain top models
```

**Deliverable:** GNN model + NRMSE-trained models

### Day 4 (Oct 30):
```markdown
✅ Phase 2.3: Conformer Implementation
  - [ ] Implement hybrid CNN-Transformer
  - [ ] Train and evaluate
  
✅ Phase 4: Augmentation & Preprocessing
  - [ ] Implement augmentations
  - [ ] Test preprocessing variants
```

**Deliverable:** Conformer model + augmented training

### Day 5 (Oct 31):
```markdown
✅ Phase 5: Ensemble Creation
  - [ ] Combine top 5 models
  - [ ] Test different ensemble strategies
  - [ ] Create final submission
  
✅ Testing & Submission
  - [ ] Local validation
  - [ ] Upload to competition
  - [ ] Analyze results
```

**Deliverable:** Final ensemble submission

---

## Success Metrics

### Intermediate Goals:
- **Day 1:** Understand data distribution, identify issues
- **Day 2:** Transformer beats baseline (< 1.0015)
- **Day 3:** GNN shows promise (< 1.0)
- **Day 4:** Conformer competitive (< 1.0)
- **Day 5:** Ensemble achieves ≤ 0.95

### Final Goal:
- **Challenge 1 NRMSE:** ≤ 0.9
- **Overall Score:** ≤ 0.95 (C1: 0.9, C2: 1.0087)

---

## Risk Mitigation

### Risk 1: Architectures don't improve
**Mitigation:** Focus on data quality and preprocessing

### Risk 2: Training takes too long
**Mitigation:** Use smaller models first, then scale up

### Risk 3: Overfitting to new validation
**Mitigation:** Use cross-validation, track multiple metrics

### Risk 4: Test distribution too different
**Mitigation:** Domain adaptation techniques, robust models

---

## Resource Requirements

### Compute:
- GPU: 1-2 days continuous training
- CPU: Data preprocessing, analysis

### Data:
- All R1-R7 datasets (verify availability)
- ~100k samples estimated

### Time:
- 5 days intensive development
- Parallel experiments where possible

---

## Next Actions (IMMEDIATE)

1. **Create project structure** for new code
2. **Implement dataset analysis** (Phase 1.1)
3. **Start Transformer implementation** (Phase 2.1)
4. **Set up experiment tracking** (MLflow or similar)

---

**Status:** 🚀 READY TO START  
**Target:** Challenge 1 NRMSE ≤ 0.9  
**Timeline:** 5 days intensive development  
**Confidence:** High - multi-pronged approach maximizes success chance

