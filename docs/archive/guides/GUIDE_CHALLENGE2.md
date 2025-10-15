# Guide: Implement Challenge 2 (Sex Classification)

**Goal:** Binary classification of sex from EEG (AUROC > 0.7, Accuracy > 65%)

---

## 🎯 Challenge 2 Overview

**Task:** Predict biological sex (M/F) from resting-state EEG  
**Metric:** AUROC (Area Under ROC Curve) > 0.7  
**Secondary:** Accuracy > 65%, Balanced Accuracy  
**Type:** Binary classification

**Why it's interesting:**
- Sex differences in brain activity patterns
- Validates EEG features capture biological information
- Often easier than age prediction (binary vs continuous)

---

## 📋 Implementation Steps

### Step 1: Get Sex Labels

#### Option A: From participants.tsv
```bash
# Check file
cat data/raw/hbn/participants.tsv | head

# Expected format:
# participant_id    age    sex
# sub-NDARAA075AMK  10     M
# sub-NDARAA306NT2  12     F
```

#### Option B: Extract from BIDS metadata
```python
import json
from pathlib import Path
import pandas as pd

def extract_sex_labels(data_dir):
    """Extract sex labels from BIDS dataset"""
    data = []
    
    for sub_dir in sorted(data_dir.glob("sub-*")):
        sub_id = sub_dir.name
        
        # Check participants.json
        json_file = sub_dir / "eeg" / f"{sub_id}_task-rest_eeg.json"
        if json_file.exists():
            with open(json_file) as f:
                metadata = json.load(f)
                sex = metadata.get('sex', metadata.get('gender', 'U'))
                data.append({'participant_id': sub_id, 'sex': sex})
    
    df = pd.DataFrame(data)
    df.to_csv(data_dir / 'participants.tsv', sep='\t', index=False)
    print(f"Created participants.tsv with {len(df)} subjects")
    print(f"Sex distribution:\n{df['sex'].value_counts()}")
    
    return df

# Run
data_dir = Path("data/raw/hbn")
sex_df = extract_sex_labels(data_dir)
```

#### Option C: Demo data (for testing)
```python
# Random 50/50 split
sexes = torch.randint(0, 2, (len(dataset),))  # 0=F, 1=M
```

---

### Step 2: Create Challenge 2 Script

```bash
# Copy Challenge 1 as template
cp scripts/train_challenge1_simple.py scripts/train_challenge2.py

# Edit to change:
# 1. Regression → Classification
# 2. MSE loss → Binary Cross-Entropy
# 3. Pearson r → AUROC
# 4. Age labels → Sex labels
```

---

### Step 3: Complete Implementation

```python
#!/usr/bin/env python3
"""
Challenge 2: Sex Classification (BSD)
Binary classification from resting-state EEG
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.models.eeg_dataset_simple import SimpleEEGDataset

# === MODELS ===

class FoundationModel(nn.Module):
    """Foundation model (same as before)"""
    def __init__(self, n_channels=129, seq_len=1000, hidden_dim=64, 
                 n_heads=4, n_layers=2, dropout=0.1):
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

class SexClassificationModel(nn.Module):
    """Binary classification head for sex prediction"""
    def __init__(self, backbone, hidden_dim=64):
        super().__init__()
        self.backbone = backbone
        
        # Classification head
        self.sex_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),  # Single output for binary
            nn.Sigmoid()  # Output probability
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.sex_head(features).squeeze(-1)

# === TRAINING ===

def train_epoch(model, loader, optimizer, criterion, device, sexes, indices):
    """Train one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, x in enumerate(loader):
        x = x.to(device)
        
        # Get sex labels for this batch
        batch_indices = indices[batch_idx * loader.batch_size:(batch_idx + 1) * loader.batch_size]
        y = sexes[batch_indices].float().to(device)
        
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    auroc = roc_auc_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    balanced_acc = balanced_accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    
    return total_loss / len(loader), auroc, accuracy, balanced_acc

def validate(model, loader, criterion, device, sexes, indices):
    """Validate"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, x in enumerate(loader):
            x = x.to(device)
            
            batch_indices = indices[batch_idx * loader.batch_size:(batch_idx + 1) * loader.batch_size]
            y = sexes[batch_indices].float().to(device)
            
            preds = model(x)
            loss = criterion(preds, y)
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    auroc = roc_auc_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    balanced_acc = balanced_accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    
    return total_loss / len(loader), auroc, accuracy, balanced_acc

def main():
    print("🚀 CHALLENGE 2: SEX CLASSIFICATION (BSD)")
    
    device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Load foundation model
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "minimal_best.pth"
    if not checkpoint_path.exists():
        print(f"❌ Foundation model not found: {checkpoint_path}")
        return
    
    print("📂 Loading foundation model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create backbone
    backbone = FoundationModel(
        hidden_dim=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    ).to(device)
    
    # Load weights (excluding classifier)
    backbone_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                    if not k.startswith('classifier')}
    backbone.load_state_dict(backbone_dict, strict=False)
    print("✅ Loaded foundation model weights")
    
    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    print("🔒 Backbone frozen, training head only")
    
    # Create sex classification model
    model = SexClassificationModel(backbone, hidden_dim=64).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} params")
    
    # Load dataset
    data_dir = Path(__file__).parent.parent / "data" / "raw" / "hbn"
    full_dataset = SimpleEEGDataset(data_dir=data_dir, max_subjects=None)
    
    # Load sex labels
    participants_file = data_dir / "participants.tsv"
    if participants_file.exists():
        print("📋 Loading sex labels from participants.tsv")
        df = pd.read_csv(participants_file, sep='\t')
        sex_dict = dict(zip(df['participant_id'], df['sex']))
        
        # Convert to binary: M=1, F=0
        sexes = []
        for i in range(len(full_dataset)):
            sub_id = f"sub-{i:05d}"  # Adjust based on your naming
            sex = sex_dict.get(sub_id, 'U')
            sexes.append(1 if sex == 'M' else 0)
        sexes = torch.tensor(sexes)
        
        print(f"Sex distribution: M={sexes.sum().item()}, F={(1-sexes).sum().item()}")
    else:
        print("⚠️  No participants.tsv, using random labels (demo)")
        sexes = torch.randint(0, 2, (len(full_dataset),))
    
    # Use subset for faster training (or full dataset)
    indices = torch.randperm(len(full_dataset))[:2000]  # Remove [:2000] for full
    dataset = Subset(full_dataset, indices.tolist())
    sexes_subset = sexes[indices]
    
    print(f"Using {len(dataset)} samples")
    
    # Split
    train_size = int(0.8 * len(dataset))
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Training setup
    criterion = nn.BCELoss()  # Binary Cross-Entropy
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=1e-4
    )
    
    best_auroc = 0
    best_checkpoint = None
    
    # Train
    epochs = 5
    print(f"\n{'='*80}")
    print("TRAINING")
    print('='*80)
    
    for epoch in range(epochs):
        train_loss, train_auroc, train_acc, train_bacc = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            sexes_subset, train_indices
        )
        
        val_loss, val_auroc, val_acc, val_bacc = validate(
            model, val_loader, criterion, device, 
            sexes_subset, val_indices
        )
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train: loss={train_loss:.4f}, AUROC={train_auroc:.4f}, "
              f"Acc={train_acc:.1%}, Balanced={train_bacc:.1%}")
        print(f"  Val:   loss={val_loss:.4f}, AUROC={val_auroc:.4f}, "
              f"Acc={val_acc:.1%}, Balanced={val_bacc:.1%}")
        
        # Save best
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auroc': val_auroc,
                'val_accuracy': val_acc,
                'val_balanced_accuracy': val_bacc,
            }
            checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            save_path = checkpoint_dir / "challenge2_best.pth"
            torch.save(best_checkpoint, save_path)
            print(f"  💾 Saved checkpoint (best AUROC: {best_auroc:.4f})")
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print('='*80)
    print(f"Best AUROC: {best_auroc:.4f}")
    print(f"Target: > 0.7 {'✅ PASS' if best_auroc > 0.7 else '❌ FAIL'}")
    
    # Generate submission
    print("\n📝 Generating submission...")
    
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for x in DataLoader(full_dataset, batch_size=32):
            x = x.to(device)
            preds = model(x)
            all_preds.extend(preds.cpu().numpy())
    
    # Create submission
    submission = pd.DataFrame({
        'participant_id': [f'sub-{i:05d}' for i in range(len(all_preds))],
        'sex_prediction': all_preds  # Probability of Male
    })
    
    submission_dir = Path(__file__).parent.parent / "submissions"
    submission_dir.mkdir(exist_ok=True)
    submission_path = submission_dir / "challenge2_predictions.csv"
    submission.to_csv(submission_path, index=False)
    
    print(f"✅ Submission saved: {submission_path}")
    print(f"   Rows: {len(submission)}")
    print(f"   Columns: {list(submission.columns)}")
    print(f"   Mean prediction: {submission['sex_prediction'].mean():.3f}")
    print(f"   (0.5 = balanced, <0.5 = more F, >0.5 = more M)")

if __name__ == "__main__":
    main()
```

---

## 📊 Expected Performance

**With random labels (demo):**
- AUROC: ~0.5 (random chance)
- Accuracy: ~50%

**With real labels + transfer learning:**
- AUROC: 0.7-0.8 ✅
- Accuracy: 65-75% ✅
- Balanced Accuracy: 65-75%

**With full dataset + fine-tuning:**
- AUROC: 0.8-0.9 ✅✅
- Accuracy: 75-85%
- Balanced Accuracy: 75-85%

---

## 🚀 Quick Start

```bash
# 1. Create the script
cat > scripts/train_challenge2.py << 'EOF'
# (paste code above)
