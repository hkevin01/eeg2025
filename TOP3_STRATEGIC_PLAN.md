# Strategic Plan to Reach Top 3 - EEG Foundation Challenge

**Current Position**: #65 (1.00613 overall)  
**Target Position**: Top 3  
**Time Remaining**: 4 days (until November 3, 2025)  
**Current Leaders**: brain&ai (0.97367), Sigma Nova (0.97841), bluewater (1.01356)

---

## 📊 Leaderboard Analysis

### Top 3 Teams (Target to Beat)

| Rank | Team | Overall | C1 | C2 | Gap from Us |
|------|------|---------|----|----|-------------|
| 2 | **brain&ai** | **0.97367** | **0.91222** | **1.0** | -0.03246 (-3.2%) |
| 4 | **Sigma Nova** | **0.97841** | **0.92245** | **1.00239** | -0.02772 (-2.8%) |
| 3 | **bluewater** | **1.01356** | **0.92215** | **1.05274** | +0.00743 (+0.7%) |

### Our Current Scores

| Challenge | Our Score | Top 3 Avg | Gap | Improvement Needed |
|-----------|-----------|-----------|-----|-------------------|
| **C1** | **1.00019** | **0.91894** | **-0.08125** | **-8.1%** ⚠️ |
| **C2** | **1.00867** | **1.01838** | **+0.00971** | **+1.0%** ✅ |
| **Overall** | **1.00613** | **0.98855** | **-0.01758** | **-1.7%** |

### Critical Insight

**C1 is the bottleneck!** We need to improve C1 from 1.00019 to ~0.92 (8% improvement) to be competitive.

Our C2 (1.00867) is actually **better** than top teams! 

---

## 🎯 Path to Top 3: Two Strategies

### Strategy A: Aggressive C1 Improvement (Primary)

**Goal**: Get C1 from 1.00019 → 0.92 (8% improvement)  
**Time**: 3 days intensive training  
**Risk**: High (big changes needed)  
**Reward**: High (can reach top 3)

### Strategy B: Balanced Improvement (Backup)

**Goal**: C1 from 1.00019 → 0.96 (4%) + C2 from 1.00867 → 0.99 (2%)  
**Time**: 2-3 days  
**Risk**: Medium  
**Reward**: Medium (reach top 10-15)

**Recommendation**: Try Strategy A with hybrid supervised + unsupervised approach

---

## 🔬 Strategy A: Hybrid Supervised + Unsupervised Learning

### Phase 1: Self-Supervised Pre-training (Day 1-2)

**Objective**: Learn robust EEG representations without age labels

#### Approach 1: Contrastive Learning
```
1. SimCLR for EEG
   - Augmentations: time shifts, channel dropout, amplitude scaling, temporal masking
   - Contrastive loss: Pull together different augmentations of same signal
   - Architecture: Same CompactCNN backbone
   - Expected: Better feature representations
   
2. Temporal Prediction
   - Predict future windows from past windows
   - Masked autoencoding: Mask 30% of temporal windows, predict them
   - Forces model to learn temporal structure
```

#### Approach 2: Multi-Task Self-Supervised
```
1. Auto-encoding task: Reconstruct input EEG
2. Temporal order prediction: Shuffle windows, predict correct order
3. Channel prediction: Mask channels, predict from others
4. Rotation prediction: Apply time-domain rotations, predict angle
```

#### Implementation Plan
```python
# ssl_pretrain_c1.py

class EEGContrastiveLearner(nn.Module):
    def __init__(self, base_encoder):
        self.encoder = base_encoder  # Our CompactCNN
        self.projector = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    
    def forward(self, x1, x2):
        # x1, x2 are two augmented views
        z1 = self.projector(self.encoder.features(x1))
        z2 = self.projector(self.encoder.features(x2))
        return z1, z2

# Loss: NT-Xent (normalized temperature-scaled cross entropy)
def contrastive_loss(z1, z2, temperature=0.5):
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    # Maximize similarity between positive pairs
    # Minimize similarity between negative pairs
    ...
```

**Expected Improvement**: 2-4% on C1

### Phase 2: Fine-tuning on Age Prediction (Day 2-3)

**Objective**: Transfer learned representations to age prediction

#### Approach: Progressive Fine-tuning
```
1. Freeze encoder, train only head (5 epochs)
   - Quick adaptation to age task
   - Preserve learned representations
   
2. Unfreeze top layers, fine-tune (10 epochs)
   - Conv block 3 + regressor
   - Low learning rate (1e-4)
   
3. Full fine-tuning (15 epochs)
   - All layers trainable
   - Very low learning rate (5e-5)
   - Strong regularization
```

**Expected Improvement**: Additional 2-3% on C1

### Phase 3: Ensemble with Diversity (Day 3-4)

**Objective**: Combine multiple models for robustness

#### Approach: Diverse Ensemble
```
1. Pre-trained model (from Phase 1-2)
2. V8 baseline (our current best)
3. Attention-based model (lightweight)
4. Frequency-domain model (FFT features)
5. Multi-scale temporal model

Average predictions with learned weights:
  y = w1*model1 + w2*model2 + ... + w5*model5
  
Optimize weights on validation set
```

**Expected Improvement**: Additional 1-2% on C1

### Total Expected C1 Score

| Component | Current | After SSL | After Fine-tune | After Ensemble | Target |
|-----------|---------|-----------|----------------|----------------|--------|
| C1 Score | 1.00019 | 0.980 | 0.950 | 0.930-0.940 | 0.92 |
| Improvement | - | -2% | -3% | -6-7% | **-8%** |

---

## 🧠 C2 Minor Improvements

Since C2 is already competitive (1.00867, better than top 3 average), focus on stability:

### Approach: Light Ensemble
```
1. Current EEGNeX baseline (1.00867)
2. SAM-optimized variant
3. Attention-enhanced variant

Simple averaging
Expected: 0.995-1.005 (maintain or slight improvement)
```

---

## 📋 Detailed 4-Day Implementation Plan

### Day 1 (Oct 30-31): Self-Supervised Pre-training

**Morning (4 hours)**:
```bash
# 1. Create contrastive learning dataset
python create_ssl_dataset_c1.py
  - Load all C1 data (R1-R4)
  - Create paired augmented views
  - Save as H5 files

# 2. Implement contrastive learner
python ssl_pretrain_c1.py
  - SimCLR-style contrastive learning
  - Train for 50 epochs
  - Save best encoder
```

**Afternoon (4 hours)**:
```bash
# 3. Multi-task SSL (parallel)
python ssl_multitask_c1.py
  - Auto-encoding + temporal prediction
  - Train for 30 epochs
  - Save best encoder

# 4. Evaluate representations
python evaluate_ssl_features.py
  - Linear probe on frozen features
  - Check if better than random init
```

**Evening (2 hours)**:
```bash
# 5. Select best SSL model
# 6. Document results
```

**Expected Output**: Pre-trained encoder with Val NRMSE ~0.140-0.150

### Day 2 (Oct 31-Nov 1): Fine-tuning + C1 Variants

**Morning (4 hours)**:
```bash
# 1. Progressive fine-tuning
python finetune_ssl_c1.py --freeze-encoder
  - Train head only (5 epochs)
  
python finetune_ssl_c1.py --freeze-early
  - Unfreeze top layers (10 epochs)
  
python finetune_ssl_c1.py --full
  - Full fine-tuning (15 epochs)
```

**Afternoon (4 hours)**:
```bash
# 2. Train complementary models
python train_c1_attention.py
  - Lightweight attention model
  - 25 epochs
  
python train_c1_frequency.py
  - FFT-based features
  - 20 epochs
```

**Evening (2 hours)**:
```bash
# 3. Evaluate all C1 models
python evaluate_c1_variants.py
  - Compare Val NRMSE
  - Select top 3-5 models
```

**Expected Output**: 3-5 C1 models with Val NRMSE 0.130-0.150

### Day 3 (Nov 1-2): Ensemble Optimization + C2

**Morning (3 hours)**:
```bash
# 1. C1 Ensemble
python optimize_c1_ensemble.py
  - Grid search ensemble weights
  - Validate on R4
  - Target: Val NRMSE < 0.135
```

**Afternoon (3 hours)**:
```bash
# 2. C2 Ensemble (light)
python train_c2_variants.py
  - Train 2-3 variants quickly
  
python optimize_c2_ensemble.py
  - Simple averaging
  - Target: Maintain ~1.00 or slightly better
```

**Evening (4 hours)**:
```bash
# 3. Create V10 submission
python create_v10_submission.py
  - Package C1 ensemble
  - Package C2 ensemble
  - Test locally
  
# 4. Submit V10
# Upload and get test scores
```

**Expected Output**: V10 with Overall ~0.96-0.98

### Day 4 (Nov 2-3): Final Refinement

**Morning (4 hours)**:
```bash
# 1. Analyze V10 results
# 2. If needed, retrain weak component
# 3. Hyperparameter tuning
```

**Afternoon (3 hours)**:
```bash
# 4. Create V11 (final)
# 5. Test thoroughly
# 6. Submit final entry
```

**Evening (2 hours)**:
```bash
# 7. Monitor leaderboard
# 8. Last-minute adjustments if time permits
```

---

## 🛠️ Technical Implementation Details

### Self-Supervised Contrastive Learning

```python
# create_ssl_dataset_c1.py

class ContrastiveEEGDataset(Dataset):
    def __init__(self, h5_paths):
        # Load data
        self.data = load_eeg_data(h5_paths)
        
    def __getitem__(self, idx):
        x = self.data[idx]
        
        # Create two augmented views
        x1 = self.augment(x)
        x2 = self.augment(x)
        
        return x1, x2
    
    def augment(self, x):
        # Strong augmentation for contrastive learning
        aug_list = []
        
        # 1. Time shift: ±20 samples
        if random.random() < 0.8:
            shift = random.randint(-20, 20)
            x = torch.roll(x, shifts=shift, dims=-1)
        
        # 2. Channel dropout: Drop 20-40% channels
        if random.random() < 0.5:
            n_drop = random.randint(25, 50)
            drop_idx = random.sample(range(129), n_drop)
            x[drop_idx, :] = 0
        
        # 3. Temporal masking: Mask 20-40% timepoints
        if random.random() < 0.5:
            mask_len = random.randint(40, 80)
            start = random.randint(0, 200 - mask_len)
            x[:, start:start+mask_len] = 0
        
        # 4. Amplitude scaling: 0.7-1.3x
        if random.random() < 0.7:
            scale = random.uniform(0.7, 1.3)
            x = x * scale
        
        # 5. Gaussian noise: σ=0.03
        if random.random() < 0.6:
            noise = torch.randn_like(x) * 0.03
            x = x + noise
        
        # 6. Frequency filtering (optional)
        if random.random() < 0.3:
            # Apply bandpass filter
            x = apply_bandpass(x, low=1, high=40)
        
        return x


class SimCLR_EEG(nn.Module):
    def __init__(self, base_encoder, projection_dim=64):
        super().__init__()
        self.encoder = base_encoder
        
        # Projection head (throw away after pre-training)
        self.projector = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Linear(128, projection_dim)
        )
    
    def forward(self, x):
        features = self.encoder.features(x)
        features = F.adaptive_avg_pool1d(features, 1).flatten(1)
        projection = self.projector(features)
        return F.normalize(projection, dim=1)


def nt_xent_loss(z1, z2, temperature=0.5):
    """Normalized Temperature-scaled Cross Entropy Loss"""
    batch_size = z1.shape[0]
    
    # Concatenate z1 and z2
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(z, z.t()) / temperature  # (2B, 2B)
    
    # Create positive pair mask
    pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
    pos_mask[range(batch_size), range(batch_size, 2*batch_size)] = True
    pos_mask[range(batch_size, 2*batch_size), range(batch_size)] = True
    
    # Compute loss
    exp_sim = torch.exp(sim_matrix)
    pos_sim = exp_sim[pos_mask].view(2*batch_size, 1)
    neg_sim = exp_sim[~pos_mask].view(2*batch_size, -1).sum(dim=1, keepdim=True)
    
    loss = -torch.log(pos_sim / (pos_sim + neg_sim))
    return loss.mean()


# Training loop
def train_ssl(model, dataloader, optimizer, epochs=50):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for x1, x2 in dataloader:
            x1, x2 = x1.to(device), x2.to(device)
            
            # Forward pass
            z1 = model(x1)
            z2 = model(x2)
            
            # Compute loss
            loss = nt_xent_loss(z1, z2, temperature=0.5)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Validate: Linear probe on frozen features
        if (epoch + 1) % 5 == 0:
            val_nrmse = linear_probe_eval(model.encoder)
            print(f"  Linear Probe Val NRMSE: {val_nrmse:.6f}")
```

### Fine-tuning Strategy

```python
# finetune_ssl_c1.py

def finetune_progressive(pretrained_encoder, train_loader, val_loader):
    # Phase 1: Freeze encoder, train head only
    model = nn.Sequential(
        pretrained_encoder.features,  # Frozen
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(96, 32),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(32, 1)
    )
    
    # Freeze encoder
    for param in model[0].parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        weight_decay=0.05
    )
    
    print("Phase 1: Training head only (5 epochs)")
    train(model, train_loader, val_loader, optimizer, epochs=5)
    
    # Phase 2: Unfreeze top layers
    for param in model[0][-3:].parameters():  # Last conv block
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001,  # Lower LR
        weight_decay=0.05
    )
    
    print("Phase 2: Fine-tuning top layers (10 epochs)")
    train(model, train_loader, val_loader, optimizer, epochs=10)
    
    # Phase 3: Full fine-tuning
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.00005,  # Even lower LR
        weight_decay=0.05
    )
    
    print("Phase 3: Full fine-tuning (15 epochs)")
    train(model, train_loader, val_loader, optimizer, epochs=15)
    
    return model
```

### Ensemble Optimization

```python
# optimize_c1_ensemble.py

def optimize_ensemble_weights(models, val_loader):
    """Find optimal weights for ensemble averaging"""
    
    # Collect all predictions
    all_preds = []
    all_targets = []
    
    for model in models:
        model.eval()
        preds = []
        targets = []
        
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                pred = model(X)
                preds.extend(pred.cpu().numpy())
                targets.extend(y.numpy())
        
        all_preds.append(np.array(preds))
        all_targets.append(np.array(targets))
    
    all_targets = all_targets[0]  # Same for all models
    all_preds = np.array(all_preds)  # (n_models, n_samples)
    
    # Grid search for weights
    best_nrmse = float('inf')
    best_weights = None
    
    # Try different weight combinations
    from itertools import product
    weight_ranges = [np.arange(0, 1.1, 0.1) for _ in range(len(models))]
    
    for weights in product(*weight_ranges):
        weights = np.array(weights)
        if weights.sum() == 0:
            continue
        weights = weights / weights.sum()  # Normalize
        
        # Weighted average
        ensemble_preds = (all_preds.T @ weights).T
        
        # Compute NRMSE
        mse = np.mean((ensemble_preds - all_targets) ** 2)
        rmse = np.sqrt(mse)
        nrmse = rmse  # Absolute NRMSE
        
        if nrmse < best_nrmse:
            best_nrmse = nrmse
            best_weights = weights
    
    print(f"Best ensemble Val NRMSE: {best_nrmse:.6f}")
    print(f"Optimal weights: {best_weights}")
    
    return best_weights
```

---

## 📊 Expected Outcomes

### Conservative Estimate

| Component | Current | After SSL+Fine-tune | After Ensemble | Final |
|-----------|---------|---------------------|----------------|-------|
| C1 | 1.00019 | 0.960 | 0.940 | **0.935** |
| C2 | 1.00867 | 1.00867 | 1.005 | **1.000** |
| Overall | 1.00613 | 0.983 | 0.971 | **0.967** |
| **Rank** | **65** | **~20** | **~10** | **~5-8** |

### Optimistic Estimate

| Component | Current | After SSL+Fine-tune | After Ensemble | Final |
|-----------|---------|---------------------|----------------|-------|
| C1 | 1.00019 | 0.945 | 0.925 | **0.915** |
| C2 | 1.00867 | 1.00867 | 0.995 | **0.985** |
| Overall | 1.00613 | 0.975 | 0.958 | **0.950** |
| **Rank** | **65** | **~15** | **~5** | **~2-3** ✅ |

### Target for Top 3

Need overall score: **< 0.980**  
This requires:
- C1: **< 0.93** (from 1.00019)
- C2: **< 1.02** (already achieved!)

**Confidence**: 60-70% with hybrid SSL approach

---

## 🚨 Risk Mitigation

### High-Risk Items

1. **SSL may not help**: Pre-training might not improve supervised performance
   - **Mitigation**: Test early (Day 1 evening), pivot if needed
   
2. **Time constraint**: 4 days is tight for complex approaches
   - **Mitigation**: Parallelize training, use multiple GPUs/machines
   
3. **Overfitting**: More complex models might overfit
   - **Mitigation**: Strong regularization, cross-validation

### Backup Plans

**If SSL doesn't work (after Day 1)**:
- Fall back to advanced supervised techniques:
  - Multi-scale temporal modeling
  - Frequency-domain features
  - Attention mechanisms
  - Knowledge distillation from larger models

**If running out of time (Day 3)**:
- Focus only on C1 (bigger impact)
- Use simpler ensemble (just 3 models)
- Skip C2 improvements (already competitive)

---

## 💻 Resource Requirements

### Compute
- **GPU**: Need 1-2 GPUs for parallel training
  - SSL pre-training: 4-6 hours on single GPU
  - Fine-tuning: 2-3 hours per model
  - Total: ~20-30 GPU-hours

### Storage
- SSL datasets: ~5-10 GB
- Checkpoints: ~2-3 GB
- Total: ~15 GB

### Coding
- New scripts: ~1000-1500 lines
- Modifications: ~500 lines
- Total: ~2000 lines over 4 days

---

## 📋 Implementation Checklist

### Day 1 Tasks
- [ ] Create SSL augmentation pipeline
- [ ] Implement SimCLR for EEG
- [ ] Train SSL model (50 epochs, ~4 hours)
- [ ] Evaluate SSL representations (linear probe)
- [ ] **Decision point**: Continue SSL or pivot?

### Day 2 Tasks
- [ ] Progressive fine-tuning (3 phases)
- [ ] Train attention variant
- [ ] Train frequency variant
- [ ] Evaluate all C1 models
- [ ] Select top 3-5 for ensemble

### Day 3 Tasks
- [ ] Optimize C1 ensemble weights
- [ ] Train C2 variants (optional)
- [ ] Create V10 submission
- [ ] Submit and get test scores
- [ ] **Decision point**: Need V11?

### Day 4 Tasks
- [ ] Analyze V10 results
- [ ] Retrain if needed
- [ ] Create V11 (final)
- [ ] Submit final entry
- [ ] Monitor leaderboard

---

## 🎯 Success Criteria

### Minimum Success (Top 10)
- Overall: < 0.99
- C1: < 0.96
- C2: < 1.02
- Rank: #8-10

### Target Success (Top 5)
- Overall: < 0.98
- C1: < 0.94
- C2: < 1.01
- Rank: #4-5

### Maximum Success (Top 3) ✅
- Overall: < 0.975
- C1: < 0.92
- C2: < 1.00
- Rank: #2-3

---

## 🚀 NEXT IMMEDIATE ACTION

Start SSL pre-training NOW:

```bash
# Create SSL implementation
python create_ssl_pretrain_script.py

# Start training immediately
python ssl_pretrain_c1.py --epochs 50 --batch-size 256

# Monitor progress
tail -f training_ssl.log
```

**Time is critical! Begin implementation immediately!**

---

**Created**: October 30, 2025, 12:40 PM  
**Deadline**: November 3, 2025, 7:00 AM EST  
**Time Remaining**: 90 hours (3.75 days)  
**Strategy**: Hybrid supervised + self-supervised learning  
**Target**: Top 3 (Overall < 0.975)

