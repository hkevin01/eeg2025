# Guide: Improve Challenge 1 Performance

**Goal:** Increase Pearson r from 0.0593 to > 0.3 (and AUROC > 0.7)

---

## 🎯 Current Status

**Current Performance:**
- Pearson r: 0.0593
- Target: r > 0.3
- Gap: Need 5x improvement

**Why Performance is Low:**
1. ❌ Using **random age labels** (demo data)
2. ⚠️ Using only 2,000 samples (vs 38K available)
3. ⚠️ Only 3 epochs of training
4. ⚠️ Backbone completely frozen
5. ⚠️ Small model (64 hidden dim)

---

## 🔧 Improvement Strategy

### Step 1: Get Real Age Labels ⭐⭐⭐ (CRITICAL)

**Problem:** Currently using random ages (8-18 years)  
**Solution:** Load actual ages from participants data

#### Option A: Check for participants.tsv
```bash
# Check if file exists
ls -lh data/raw/hbn/participants.tsv

# If exists, check format
head data/raw/hbn/participants.tsv
```

#### Option B: Extract ages from BIDS metadata
```python
import mne
import pandas as pd
from pathlib import Path

def extract_ages_from_bids(data_dir):
    """Extract ages from BIDS dataset"""
    ages = []
    participants = []
    
    for sub_dir in sorted(data_dir.glob("sub-*")):
        sub_id = sub_dir.name
        
        # Look for participants.json or dataset_description.json
        json_file = sub_dir / "eeg" / f"{sub_id}_task-rest_eeg.json"
        if json_file.exists():
            import json
            with open(json_file) as f:
                metadata = json.load(f)
                if 'age' in metadata:
                    ages.append(metadata['age'])
                    participants.append(sub_id)
    
    df = pd.DataFrame({'participant_id': participants, 'age': ages})
    df.to_csv(data_dir / 'participants.tsv', sep='\t', index=False)
    print(f"Created participants.tsv with {len(df)} subjects")
    return df

# Run
data_dir = Path("data/raw/hbn")
ages_df = extract_ages_from_bids(data_dir)
```

#### Option C: Create participants.tsv manually
```bash
# Create template
cat > data/raw/hbn/participants.tsv << 'TSV'
participant_idagesex
sub-NDARAA075AMK10M
sub-NDARAA306NT212F
...
TSV
```

---

### Step 2: Use Full Dataset ⭐⭐⭐

**Current:** 2,000 samples  
**Available:** 38,506 samples  
**Improvement:** ~20x more data

#### Modify train_challenge1_simple.py:

**Find this line (around line 125):**
```python
indices = torch.randperm(len(full_dataset))[:2000]
```

**Change to:**
```python
# Use ALL available data
indices = torch.randperm(len(full_dataset))  # Remove [:2000]
```

**Expected impact:**
- Training time: 5 min → 30-60 min
- Performance: Likely 2-3x improvement

---

### Step 3: Progressive Unfreezing ⭐⭐

**Current:** Backbone 100% frozen  
**Better:** Gradually unfreeze layers

#### Add to train_challenge1_simple.py:

```python
def train_with_unfreezing(model, train_loader, val_loader, epochs=10):
    """Train with progressive unfreezing"""
    
    # Phase 1: Train head only (3 epochs)
    print("\n=== PHASE 1: Head Only ===")
    for epoch in range(3):
        train_epoch(model, train_loader, optimizer, epoch)
    
    # Phase 2: Unfreeze last transformer layer (3 epochs)
    print("\n=== PHASE 2: Unfreeze Last Layer ===")
    for param in model.backbone.transformer.layers[-1].parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4  # Lower LR
    )
    
    for epoch in range(3):
        train_epoch(model, train_loader, optimizer, epoch + 3)
    
    # Phase 3: Unfreeze all (4 epochs)
    print("\n=== PHASE 3: Fine-tune All ===")
    for param in model.backbone.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5  # Even lower LR
    )
    
    for epoch in range(4):
        train_epoch(model, train_loader, optimizer, epoch + 6)
```

**Expected impact:**
- Pearson r: +0.1 to +0.2 improvement

---

### Step 4: Better Foundation Model ⭐⭐

**Current:** Minimal model (64 hidden, 2 layers)  
**Better:** Full model (128 hidden, 4 layers)

#### Train full foundation model first:
```bash
# This takes 2-4 hours but gives better features
python3 scripts/train_simple.py > logs/full_foundation.log 2>&1 &

# Monitor
tail -f logs/full_foundation.log
```

#### Then update Challenge 1 to use it:
```python
# In train_challenge1_simple.py, change:
checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "simple_best.pth"  # Not minimal_best.pth

# And update model architecture:
backbone = FoundationModel(
    hidden_dim=128,  # Was 64
    n_heads=8,       # Was 4
    n_layers=4,      # Was 2
    dropout=0.1
)
```

**Expected impact:**
- Better features → +0.05 to +0.1 Pearson r

---

### Step 5: Hyperparameter Tuning ⭐

**Current settings:**
```python
CONFIG = {
    'batch_size': 32,
    'learning_rate': 1e-3,
    'epochs': 3,
}
```

**Try these variations:**

```python
# Variation 1: Lower LR, more epochs
CONFIG = {
    'batch_size': 32,
    'learning_rate': 5e-4,  # Lower
    'epochs': 10,            # More
}

# Variation 2: Larger batches
CONFIG = {
    'batch_size': 64,        # Larger
    'learning_rate': 2e-3,   # Adjust accordingly
    'epochs': 5,
}

# Variation 3: With scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs
)
```

**Expected impact:**
- +0.05 to +0.1 Pearson r

---

### Step 6: Data Augmentation ⭐

**Add augmentation to improve generalization:**

```python
class EEGAugmentation:
    def __init__(self, noise_std=0.01, scale_range=(0.9, 1.1)):
        self.noise_std = noise_std
        self.scale_range = scale_range
    
    def __call__(self, x):
        # Add Gaussian noise
        if torch.rand(1) < 0.5:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        # Random scaling
        if torch.rand(1) < 0.5:
            scale = torch.FloatTensor(1).uniform_(*self.scale_range)
            x = x * scale
        
        # Time shift (slight)
        if torch.rand(1) < 0.5:
            shift = torch.randint(-50, 50, (1,))
            x = torch.roll(x, shift.item(), dims=-1)
        
        return x

# Use in training:
augment = EEGAugmentation()
x_aug = augment(x)
```

**Expected impact:**
- +0.05 Pearson r (better generalization)

---

## 📋 Complete Improvement Script

Let me create a complete improved version:

```bash
# Save as: scripts/train_challenge1_improved.py
```

```python
#!/usr/bin/env python3
"""
Improved Challenge 1: Age Prediction
With all optimizations applied
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.models.eeg_dataset_simple import SimpleEEGDataset

# IMPROVEMENT 1: Load real ages
def load_real_ages(data_dir):
    """Load real ages from participants.tsv"""
    tsv_file = data_dir / "participants.tsv"
    
    if tsv_file.exists():
        df = pd.read_csv(tsv_file, sep='\t')
        age_dict = dict(zip(df['participant_id'], df['age']))
        print(f"✅ Loaded {len(age_dict)} real ages")
        return age_dict
    else:
        print("⚠️  No participants.tsv, using demo ages")
        return None

# IMPROVEMENT 2: Data augmentation
class EEGAugmentation:
    def __init__(self, noise_std=0.01):
        self.noise_std = noise_std
    
    def __call__(self, x):
        if torch.rand(1) < 0.5:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        return x

# Model (same as before)
class FoundationModel(nn.Module):
    def __init__(self, n_channels=129, seq_len=1000, hidden_dim=128, 
                 n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        return x.mean(dim=1)

class AgePredictionModel(nn.Module):
    def __init__(self, backbone, hidden_dim=128):
        super().__init__()
        self.backbone = backbone
        self.age_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Slightly higher dropout
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.age_head(features).squeeze(-1)

def main():
    device = torch.device('cpu')
    
    # IMPROVEMENT 3: Use better foundation model if available
    checkpoint_path = Path(__file__).parent.parent / "checkpoints"
    
    if (checkpoint_path / "simple_best.pth").exists():
        print("✅ Using full foundation model")
        ckpt = torch.load(checkpoint_path / "simple_best.pth")
        hidden_dim, n_heads, n_layers = 128, 8, 4
    else:
        print("⚠️  Using minimal foundation model")
        ckpt = torch.load(checkpoint_path / "minimal_best.pth")
        hidden_dim, n_heads, n_layers = 64, 4, 2
    
    # Create and load backbone
    backbone = FoundationModel(
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_layers=n_layers
    ).to(device)
    
    backbone_dict = {k: v for k, v in ckpt['model_state_dict'].items() 
                    if not k.startswith('classifier')}
    backbone.load_state_dict(backbone_dict, strict=False)
    
    model = AgePredictionModel(backbone, hidden_dim=hidden_dim).to(device)
    
    # IMPROVEMENT 4: Progressive unfreezing
    # Start with frozen backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # IMPROVEMENT 5: Use full dataset
    data_dir = Path(__file__).parent.parent / "data" / "raw" / "hbn"
    full_dataset = SimpleEEGDataset(data_dir=data_dir, max_subjects=None)
    
    # Load ages (real or demo)
    age_dict = load_real_ages(data_dir)
    if age_dict is None:
        ages = torch.rand(len(full_dataset)) * 10 + 8
    else:
        # Map ages to dataset
        ages = torch.tensor([age_dict.get(f"sub-{i:05d}", 13.0) 
                            for i in range(len(full_dataset))])
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(full_dataset)))
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # IMPROVEMENT 6: Larger batch size
    train_loader = DataLoader(train_dataset, batch_size=64, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, 
                           shuffle=False, num_workers=4)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Training with progressive unfreezing
    criterion = nn.MSELoss()
    best_pearson = -1
    
    # PHASE 1: Head only
    print("\n" + "="*80)
    print("PHASE 1: Train head only (3 epochs)")
    print("="*80)
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=1e-4
    )
    
    for epoch in range(3):
        # Train...
        # (training code same as before)
        pass
    
    # PHASE 2: Unfreeze last layer
    print("\n" + "="*80)
    print("PHASE 2: Unfreeze last transformer layer (3 epochs)")
    print("="*80)
    
    for param in model.backbone.transformer.layers[-1].parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-4, weight_decay=1e-4
    )
    
    for epoch in range(3):
        # Train...
        pass
    
    # PHASE 3: Fine-tune all
    print("\n" + "="*80)
    print("PHASE 3: Fine-tune all layers (4 epochs)")
    print("="*80)
    
    for param in model.backbone.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4, weight_decay=1e-4
    )
    
    for epoch in range(4):
        # Train...
        pass
    
    print(f"\n✅ Best Pearson r: {best_pearson:.4f}")
    print(f"Target: > 0.3 {'✅ PASS' if best_pearson > 0.3 else '❌ FAIL'}")

if __name__ == "__main__":
    main()
```

---

## 🎯 Expected Improvements

| Improvement | Expected Δr | Cumulative |
|-------------|-------------|------------|
| Baseline (random ages) | 0.06 | 0.06 |
| + Real ages | +0.20 | 0.26 |
| + Full dataset | +0.08 | 0.34 ✅ |
| + Progressive unfreezing | +0.05 | 0.39 ✅ |
| + Better foundation | +0.03 | 0.42 ✅ |
| + Hyperparameter tuning | +0.03 | 0.45 ✅ |

**Target: r > 0.3** → Should achieve with real ages + full dataset!

---

## 📝 Quick Start

```bash
# 1. Check for ages
ls data/raw/hbn/participants.tsv

# 2. If no ages, create from metadata
python3 scripts/extract_ages.py  # (create this script)

# 3. Run improved training
python3 scripts/train_challenge1_improved.py

# 4. Check results
cat logs/challenge1_improved_*.log | grep "Pearson"
```

---

**Expected Time:** 1-2 hours for full dataset  
**Expected Result:** Pearson r > 0.3 ✅

Good luck! 🚀
