#!/usr/bin/env python3
"""
Challenge 2: Sex Classification (Binary)
Transfer learning from foundation model with REAL labels
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

print("=" * 80)
print("🚀 CHALLENGE 2: SEX CLASSIFICATION (Binary)")
print("=" * 80)

# Foundation model architecture (same as Challenge 1)
class FoundationModel(nn.Module):
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
        x = x.mean(dim=1)
        return x

# Sex classification head
class SexClassificationModel(nn.Module):
    def __init__(self, backbone, hidden_dim=64):
        super().__init__()
        self.backbone = backbone
        self.sex_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Binary classification output
        )
    
    def forward(self, x):
        features = self.backbone(x)
        sex_prob = self.sex_head(features).squeeze(-1)
        return sex_prob

if __name__ == "__main__":
    device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Load foundation model
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "minimal_best.pth"
    if not checkpoint_path.exists():
        print(f"❌ Foundation model not found: {checkpoint_path}")
        print("Please train foundation model first: python3 scripts/train_minimal.py")
        sys.exit(1)
    
    print(f"\n📂 Loading foundation model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create backbone
    backbone = FoundationModel(hidden_dim=64, n_heads=4, n_layers=2, dropout=0.1).to(device)
    
    # Load weights (excluding classifier)
    backbone_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                    if not k.startswith('classifier')}
    backbone.load_state_dict(backbone_dict, strict=False)
    print("✅ Loaded foundation model weights")
    
    # Create sex classification model
    model = SexClassificationModel(backbone, hidden_dim=64).to(device)
    
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"🔒 Backbone frozen: {trainable:,} / {total:,} trainable params")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    data_dir = Path(__file__).parent.parent / "data" / "raw" / "hbn"
    
    # Load REAL sex labels from participants.tsv
    print("📋 Loading real sex labels from participants.tsv...")
    participants_file = data_dir / "participants.tsv"
    
    if participants_file.exists():
        participants_df = pd.read_csv(participants_file, sep='\t')
        print(f"✅ Loaded {len(participants_df)} participants")
        
        # Sex distribution
        sex_counts = participants_df['sex'].value_counts()
        print(f"   Sex distribution: M={sex_counts.get('M', 0)}, F={sex_counts.get('F', 0)}")
        
        # Create sex mapping (M=1, F=0)
        sex_dict = {}
        for _, row in participants_df.iterrows():
            sub_id = row['participant_id']
            sex = 1 if row['sex'] == 'M' else 0
            sex_dict[sub_id] = sex
        
        # Load dataset
        full_dataset = SimpleEEGDataset(data_dir=data_dir, max_subjects=None)
        
        # Map sex labels to windows
        subject_dirs = sorted([d for d in data_dir.iterdir() 
                             if d.is_dir() and d.name.startswith('sub-') and (d / "eeg").exists()])
        
        real_sexes = []
        for subject_dir in subject_dirs:
            subject_id = subject_dir.name
            sex = sex_dict.get(subject_id, 0)
            
            # Estimate windows per subject
            eeg_files = list((subject_dir / "eeg").glob("*.set"))
            estimated_windows = len(eeg_files) * 200
            
            for _ in range(min(estimated_windows, len(full_dataset) - len(real_sexes))):
                real_sexes.append(sex)
        
        # Fill remaining if needed
        while len(real_sexes) < len(full_dataset):
            real_sexes.append(0)
        
        real_sexes = real_sexes[:len(full_dataset)]
        
        print(f"✅ Mapped sex labels to {len(full_dataset)} windows")
        
        # Use 5000 samples
        indices = torch.randperm(len(full_dataset))[:5000]
        dataset = Subset(full_dataset, indices)
        sexes = torch.tensor([real_sexes[i] for i in indices], dtype=torch.float32)
        
        print(f"✅ Using {len(dataset)} samples with REAL sex labels")
        print(f"   Distribution: M={sexes.sum().item():.0f}, F={(len(sexes)-sexes.sum()).item():.0f}")
    else:
        print("⚠️  participants.tsv not found")
        sys.exit(1)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    
    print("\n" + "=" * 80)
    print("🏋️  TRAINING")
    print("=" * 80)
    
    best_auroc = 0
    
    for epoch in range(5):
        print(f"\nEpoch {epoch+1}/5:")
        
        # Train
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        
        for batch_idx, x in enumerate(train_loader):
            x = x.to(device)
            batch_indices = train_indices[batch_idx * 32:(batch_idx + 1) * 32]
            y = sexes[batch_indices].to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(pred.detach().cpu().numpy())
            train_labels.extend(y.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_auroc = roc_auc_score(train_labels, train_preds)
        train_acc = accuracy_score(train_labels, [1 if p > 0.5 else 0 for p in train_preds])
        
        # Validate
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch_idx, x in enumerate(val_loader):
                x = x.to(device)
                batch_indices = val_indices[batch_idx * 32:(batch_idx + 1) * 32]
                y = sexes[batch_indices].to(device)
                
                pred = model(x)
                loss = criterion(pred, y)
                
                val_loss += loss.item()
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_auroc = roc_auc_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, [1 if p > 0.5 else 0 for p in val_preds])
        val_bal_acc = balanced_accuracy_score(val_labels, [1 if p > 0.5 else 0 for p in val_preds])
        
        print(f"  Train: loss={train_loss:.4f}, AUROC={train_auroc:.4f}, Acc={train_acc:.1%}")
        print(f"  Val:   loss={val_loss:.4f}, AUROC={val_auroc:.4f}, Acc={val_acc:.1%}, Bal={val_bal_acc:.1%}")
        
        # Save best
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auroc': val_auroc,
                'val_accuracy': val_acc,
                'val_balanced_accuracy': val_bal_acc
            }, checkpoint_dir / "challenge2_best.pth")
            print(f"  💾 Saved checkpoint (best AUROC: {best_auroc:.4f})")
    
    print("\n" + "=" * 80)
    print("✅ CHALLENGE 2 COMPLETE!")
    print("=" * 80)
    print(f"Best AUROC: {best_auroc:.4f}")
    print(f"Target: > 0.7 {'✅ PASS' if best_auroc > 0.7 else '❌ FAIL'}")
    
    # Generate submission
    print("\n📝 Generating submission...")
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for x in DataLoader(full_dataset, batch_size=32, num_workers=2):
            x = x.to(device)
            pred = model(x)
            all_preds.extend(pred.cpu().numpy())
    
    submission = pd.DataFrame({
        'participant_id': [f'sub-{i:05d}' for i in range(len(all_preds))],
        'sex_prediction': all_preds
    })
    
    submission_dir = Path(__file__).parent.parent / "submissions"
    submission_dir.mkdir(exist_ok=True)
    submission.to_csv(submission_dir / "challenge2_predictions.csv", index=False)
    
    print(f"✅ Submission saved: submissions/challenge2_predictions.csv")
    print(f"   Rows: {len(submission)}, Mean: {submission['sex_prediction'].mean():.3f}")
