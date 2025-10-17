# 🎯 Integrated Improvement Plan
**Combining Simple Roadmap + Advanced Algorithms**

**Current Status:** Position #47, Overall: 2.013  
**Target:** Top 15 (Overall < 1.3)  
**Timeline:** Tonight → Tomorrow → Weekend

---

## 📊 Strategy Comparison & Integration

### Original Roadmap (Simple)
✅ Multi-release training (R1+R2+R3)
✅ Stronger regularization (dropout, weight decay)
✅ 3-fold ensemble

### Advanced Algorithmic Methods
✅ Release-grouped cross-validation (GroupKFold)
✅ Robust loss functions (Huber, Quantile)
✅ CORAL alignment (distribution matching)
✅ Multi-scale CNN with attention
✅ Domain-adversarial training (optional)
✅ Per-sample reweighting (residual-based)

### **INTEGRATED APPROACH** ⭐
Combine the best of both:
1. **Phase 1 (Tonight):** Simple multi-release + Huber loss
2. **Phase 2 (Tomorrow):** Release-grouped CV + CORAL alignment
3. **Phase 3 (Weekend):** Multi-scale architecture + ensemble

---

## 🚀 Phase 1: Quick Win (Tonight, 3 hours)
**Goal:** Simple multi-release training + robust loss  
**Expected:** 2.01 → 1.5-1.7 (top 30)

### ✅ Approved Methods from Advanced Guide:
1. **Huber Loss** (instead of MSE)
   - More robust to outliers
   - Delta = 1.0
   - Drop-in replacement, no architecture change

2. **Per-sample Reweighting** (after warmup)
   - Downweight high-residual samples
   - Start after epoch 5
   - Prevents overfitting to noisy labels

3. **Multi-release Training**
   - R1+R2+R3 combined (80/20 split)
   - Random split with seed=42

### Implementation Checklist:

```markdown
Phase 1 Todo:
- [ ] 1. Create train_challenge1_robust.py (15 min)
      - Copy from train_challenge1_multi_release.py
      - Change R1+R2 → R1+R2+R3 with 80/20 split
      - Replace MSE with Huber loss
      - Add residual reweighting after epoch 5
      
- [ ] 2. Create train_challenge2_robust.py (15 min)
      - Same modifications for Challenge 2
      
- [ ] 3. Train Challenge 1 (1 hour)
      python scripts/train_challenge1_robust.py
      
- [ ] 4. Train Challenge 2 (1 hour)
      python scripts/train_challenge2_robust.py
      
- [ ] 5. Create submission v2 (15 min)
      - Package weights
      - Update submission.py
      - Upload to Codabench
```

### Code Changes for Phase 1:

**File: `scripts/train_challenge1_robust.py`**

```python
#!/usr/bin/env python3
"""
ROBUST TRAINING v1: Multi-release + Huber loss + Residual reweighting

Changes from baseline:
1. Train on R1+R2+R3 (80/20 split) instead of R1+R2 train, R3 val
2. Huber loss (delta=1.0) instead of MSE - robust to outliers
3. Per-sample reweighting after warmup (epoch 5+) - downweight noisy samples
"""

import torch
import torch.nn as nn

# ============= HUBER LOSS (Robust) =============
def huber_loss(pred, target, delta=1.0):
    """
    Huber loss: quadratic for small errors, linear for large errors.
    More robust than MSE to outliers.
    """
    err = pred - target
    abs_err = err.abs()
    quad = torch.clamp(abs_err, max=delta)
    lin = abs_err - quad
    return (0.5 * quad**2 + delta * lin).mean()


# ============= DATASET LOADING (Multi-release) =============
def load_datasets():
    """Load R1+R2+R3 combined, split 80/20"""
    
    print("\n📦 Loading ALL releases (R1+R2+R3)...")
    print("⚠️  Will split 80% train / 20% validation")
    
    # Load all data
    all_dataset = MultiReleaseDataset(
        releases=['R1', 'R2', 'R3'],
        mini=False,
        cache_dir='data/raw'
    )
    
    print(f"Total samples: {len(all_dataset)}")
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(all_dataset))
    val_size = len(all_dataset) - train_size
    
    print(f"Train samples: {train_size}")
    print(f"Val samples: {val_size}")
    
    # Reproducible split
    train_dataset, val_dataset = torch.utils.data.random_split(
        all_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, val_dataset


# ============= TRAINING LOOP WITH REWEIGHTING =============
def train_epoch(model, train_loader, optimizer, device, epoch, warmup_epochs=5):
    """
    Training with residual-based reweighting after warmup.
    
    After warmup, samples with large residuals (noisy labels) are downweighted.
    This prevents overfitting to outliers.
    """
    model.train()
    total_loss = 0.0
    n_samples = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device).float()
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs).squeeze()
        
        # Compute loss
        if epoch < warmup_epochs:
            # Warmup: standard Huber loss
            loss = huber_loss(outputs, targets, delta=1.0)
        else:
            # After warmup: residual-based reweighting
            with torch.no_grad():
                residuals = (outputs - targets).abs()
                # Compute robust weights (downweight large residuals)
                residual_std = residuals.std().clamp(min=1e-6)
                z_scores = (residuals / residual_std).clamp(max=3.0)
                weights = torch.exp(-z_scores / 2.0)  # Gaussian-like weighting
                weights = weights.detach()
            
            # Weighted Huber loss
            err = outputs - targets
            abs_err = err.abs()
            delta = 1.0
            quad = torch.clamp(abs_err, max=delta)
            lin = abs_err - quad
            loss = (weights * (0.5 * quad**2 + delta * lin)).mean()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(targets)
        n_samples += len(targets)
    
    return total_loss / n_samples


# ============= MAIN TRAINING =============
def main():
    # Load datasets
    train_dataset, val_dataset = load_datasets()
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CompactResponseTimeCNN().to(device)
    
    # Optimizer (already using weight_decay=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Training loop
    best_val_nrmse = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(50):
        # Train with reweighting after epoch 5
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, warmup_epochs=5)
        
        # Validate
        val_nrmse = validate(model, val_loader, device)
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val NRMSE: {val_nrmse:.4f}")
        
        # Save best model
        if val_nrmse < best_val_nrmse:
            best_val_nrmse = val_nrmse
            torch.save(model.state_dict(), 'weights/weights_challenge_1_robust.pt')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"\n✅ Best Val NRMSE: {best_val_nrmse:.4f}")
    print(f"Expected test improvement: 4.05 → 2.0-2.5")

if __name__ == "__main__":
    main()
```

---

## 🎯 Phase 2: Advanced Methods (Tomorrow, 6 hours)
**Goal:** Release-grouped CV + CORAL alignment  
**Expected:** 1.5 → 1.2-1.3 (top 15-20)

### ✅ Approved Methods:

1. **Release-Grouped Cross-Validation**
   - Fold 1: Train R1+R2, Val R3
   - Fold 2: Train R1+R3, Val R2
   - Fold 3: Train R2+R3, Val R1
   - Ensemble predictions (median)

2. **CORAL Loss** (Distribution Alignment)
   - Align feature covariances across releases
   - Lambda = 1e-3
   - Helps generalization to R4+R5

3. **Stronger Regularization**
   - Already have dropout 0.3-0.5 ✓
   - Already have weight_decay 1e-4 ✓
   - Add: Mixup augmentation (alpha=0.2)

### Implementation Checklist:

```markdown
Phase 2 Todo:
- [ ] 1. Create train_challenge1_cv_coral.py (1 hour)
      - Implement 3-fold release-grouped CV
      - Add CORAL loss between releases
      - Add Mixup augmentation
      
- [ ] 2. Create train_challenge2_cv_coral.py (1 hour)
      - Same for Challenge 2
      
- [ ] 3. Train 3 folds for Challenge 1 (3 hours)
      - Fold 1: R1+R2 train, R3 val
      - Fold 2: R1+R3 train, R2 val
      - Fold 3: R2+R3 train, R1 val
      
- [ ] 4. Train 3 folds for Challenge 2 (3 hours)
      - Same fold structure
      
- [ ] 5. Create ensemble submission v3 (30 min)
      - Load all 6 models (3 per challenge)
      - Median ensemble
      - Upload to Codabench
```

### Key Code Snippets for Phase 2:

**CORAL Loss (Distribution Alignment):**
```python
def coral_loss(h_source, h_target):
    """
    CORAL: Align feature covariance between releases.
    Helps model generalize across R1/R2/R3 → R4/R5.
    """
    bs = h_source.size(0)
    bt = h_target.size(0)
    
    # Center features
    hs = h_source - h_source.mean(0)
    ht = h_target - h_target.mean(0)
    
    # Covariance matrices
    cs = (hs.T @ hs) / (bs - 1)
    ct = (ht.T @ ht) / (bt - 1)
    
    # Frobenius norm of difference
    return ((cs - ct) ** 2).sum()


# In training loop (if batch contains multiple releases):
loss_main = huber_loss(pred, target)
loss_coral = coral_loss(features_r1, features_r2) + coral_loss(features_r2, features_r3)
loss_total = loss_main + 1e-3 * loss_coral
```

**Mixup Augmentation:**
```python
def mixup_batch(x, y, alpha=0.2):
    """
    Mixup: linear interpolation between samples.
    Improves regularization.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y


# In training loop:
if use_mixup:
    inputs, targets = mixup_batch(inputs, targets, alpha=0.2)
```

**3-Fold Cross-Validation:**
```python
def train_3_fold_cv():
    """Train 3 models with release-grouped folds"""
    
    folds = [
        (['R1', 'R2'], ['R3']),
        (['R1', 'R3'], ['R2']),
        (['R2', 'R3'], ['R1'])
    ]
    
    for fold_id, (train_releases, val_releases) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"Fold {fold_id+1}/3: Train {train_releases}, Val {val_releases}")
        print(f"{'='*60}")
        
        # Load datasets
        train_dataset = MultiReleaseDataset(releases=train_releases, ...)
        val_dataset = MultiReleaseDataset(releases=val_releases, ...)
        
        # Train model
        model = CompactResponseTimeCNN()
        train_model(model, train_dataset, val_dataset)
        
        # Save model
        torch.save(model.state_dict(), f'weights/model_fold{fold_id}.pt')


def ensemble_predict(x, models):
    """Median ensemble from 3 fold models"""
    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(x)
            preds.append(pred)
    
    # Median (more robust than mean)
    preds = torch.stack(preds, dim=0)
    median_pred = torch.median(preds, dim=0)[0]
    return median_pred
```

---

## 🏗️ Phase 3: Architecture Improvements (Weekend, 8 hours)
**Goal:** Multi-scale CNN + SE attention  
**Expected:** 1.2 → 1.0-1.1 (top 10-15)

### ✅ Approved Methods:

1. **Multi-Scale Temporal CNN**
   - Parallel branches: kernels [5, 15, 45, 125]
   - Concatenate multi-scale features
   - Captures both fine and coarse patterns

2. **Squeeze-and-Excitation (SE) Attention**
   - Channel-wise attention
   - Learn importance of different frequency bands
   - Lightweight (few parameters)

3. **CBAM (Convolutional Block Attention)**
   - Channel + Spatial attention
   - More powerful than SE alone

### Architecture Code:

```python
class SEBlock(nn.Module):
    """Squeeze-and-Excitation: Channel attention"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [B, C, T]
        w = self.fc(x)[:, :, None]  # [B, C, 1]
        return x * w


class MultiScaleCNN(nn.Module):
    """Multi-scale temporal CNN with SE attention"""
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        
        # Multi-scale branches (different kernel sizes)
        kernel_sizes = [5, 15, 45, 125]
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, hidden, k, padding=k//2, bias=False),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(0.5)
            ) for k in kernel_sizes
        ])
        
        # SE attention on concatenated features
        total_channels = hidden * len(kernel_sizes)
        self.se = SEBlock(total_channels, reduction=8)
        
        # Final classification head
        self.head = nn.Sequential(
            nn.Conv1d(total_channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        # Multi-scale feature extraction
        features = [branch(x) for branch in self.branches]
        features = torch.cat(features, dim=1)  # [B, 4*hidden, T]
        
        # SE attention
        features = self.se(features)
        
        # Classification
        output = self.head(features)
        return output
```

---

## ❌ EXCLUDED Methods (Why)

### From Advanced Guide - Not Applicable:

1. **P300 Features**
   - ❌ Correlation = 0.007 (useless)
   - Already confirmed in analysis

2. **Domain-Adversarial (DANN)**
   - ❌ Too complex for first iterations
   - ⏳ Consider if CORAL doesn't help enough

3. **Quantile Regression**
   - ❌ NRMSE metric expects point estimates, not quantiles
   - Could use median (0.5 quantile) but same as Huber

4. **Sequence Context (GRU over trials)**
   - ❌ Competition evaluates per-trial independently
   - No temporal context across trials in evaluation

5. **Subject Normalization with FiLM**
   - ❌ Test set doesn't provide subject IDs
   - Can't personalize at inference

6. **Heavy Transformers**
   - ❌ Already overfitting with CNN
   - Need better generalization, not bigger models

---

## 📊 Expected Score Progression

```
Current (Position #47):
├─ Overall: 2.013
├─ Challenge 1: 4.047
└─ Challenge 2: 1.141

After Phase 1 (Tonight):
├─ Overall: 1.5-1.7  ← 25-30% improvement
├─ Challenge 1: 2.0-2.5
└─ Challenge 2: 0.7-0.9
Expected Rank: #25-30

After Phase 2 (Tomorrow):
├─ Overall: 1.2-1.4  ← Additional 20% improvement
├─ Challenge 1: 1.7-2.0
└─ Challenge 2: 0.6-0.8
Expected Rank: #15-20

After Phase 3 (Weekend):
├─ Overall: 1.0-1.2  ← Additional 15% improvement
├─ Challenge 1: 1.4-1.7
└─ Challenge 2: 0.5-0.7
Expected Rank: #10-15
```

---

## 🎯 Recommended Execution Order

### Tonight (3 hours) - Phase 1:
```bash
# 1. Create robust training scripts (30 min)
cp scripts/train_challenge1_multi_release.py scripts/train_challenge1_robust.py
cp scripts/train_challenge2_multi_release.py scripts/train_challenge2_robust.py

# Edit both files:
#   - Change R1+R2 → R1+R2+R3 (80/20 split)
#   - Replace MSE with Huber loss
#   - Add residual reweighting after epoch 5

# 2. Train models (2 hours)
python scripts/train_challenge1_robust.py > logs/train_c1_robust.log 2>&1 &
sleep 10
python scripts/train_challenge2_robust.py > logs/train_c2_robust.log 2>&1 &

# Monitor
tail -f logs/train_c1_robust.log

# 3. Create submission v2 (30 min)
# ... package and upload ...
```

### Tomorrow (6 hours) - Phase 2:
```bash
# 1. Create CV+CORAL training (2 hours)
# Implement 3-fold CV with CORAL loss

# 2. Train 6 models (3 per challenge, 4 hours)
for fold in 0 1 2; do
    python scripts/train_challenge1_cv_coral.py --fold $fold &
    python scripts/train_challenge2_cv_coral.py --fold $fold &
    wait
done

# 3. Create ensemble submission v3 (30 min)
# Update submission.py to load 3 models per challenge and ensemble
```

### Weekend (8 hours) - Phase 3:
```bash
# 1. Implement multi-scale architecture (2 hours)
# Create new model class with SE attention

# 2. Train from scratch (4 hours)
# Use Phase 2 CV setup with new architecture

# 3. Final ensemble (2 hours)
# Combine Phase 2 + Phase 3 models (6 models per challenge!)
```

---

## 💡 Key Insights

**What Makes This Plan Better:**

1. **Incremental Validation**
   - Each phase builds on previous
   - Can stop if Phase 1 reaches top 20

2. **Risk Management**
   - Phase 1 is low-risk, high-reward
   - Phase 2/3 only if needed

3. **Competition-Aware**
   - Excluded methods that violate rules
   - Excluded methods proven ineffective (P300)
   - Focus on what generalizes to R4+R5

4. **Time-Efficient**
   - 3 hours → significant improvement
   - 9 hours total → top 15 possible

**Success Metrics:**

- **Phase 1 Success:** Overall < 1.7 (rank < 30)
- **Phase 2 Success:** Overall < 1.3 (rank < 20)
- **Phase 3 Success:** Overall < 1.1 (rank < 15)

---

## 📋 Final Checklist

Before starting:
- [ ] Read this entire document
- [ ] Understand Phase 1 code changes
- [ ] Backup current weights
- [ ] Clear logs directory
- [ ] Check GPU availability

Phase 1 (Tonight):
- [ ] Create robust training scripts
- [ ] Start training (2 hours)
- [ ] Validate results
- [ ] Create submission v2
- [ ] Upload and check score

Phase 2 (Tomorrow):
- [ ] Implement CV+CORAL
- [ ] Train 6 models
- [ ] Create ensemble
- [ ] Submit v3

Phase 3 (Weekend):
- [ ] Implement multi-scale architecture
- [ ] Train with new architecture
- [ ] Final ensemble
- [ ] Submit v4

---

**Ready to start Phase 1? Let me know and I'll help create the robust training scripts!** 🚀
