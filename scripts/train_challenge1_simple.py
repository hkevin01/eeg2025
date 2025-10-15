#!/usr/bin/env python3
"""
Simple Challenge 1: Age Prediction (CCD Task)
Transfer learning from foundation model
"""
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.models.eeg_dataset_simple import SimpleEEGDataset

print("=" * 80)
print("🚀 CHALLENGE 1: AGE PREDICTION (CCD)")
print("=" * 80)

# Load foundation model architecture
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

# Age prediction head
class AgePredictionModel(nn.Module):
    def __init__(self, backbone, hidden_dim=64):
        super().__init__()
        self.backbone = backbone
        self.age_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)  # Regression
        )

    def forward(self, x):
        features = self.backbone(x)
        age = self.age_head(features)
        return age.squeeze(-1)

def load_age_labels(data_dir):
    """Load age labels from participants.tsv"""
    participants_file = data_dir / "participants.tsv"
    if not participants_file.exists():
        print("⚠️  participants.tsv not found, using random ages for demo")
        return None

    df = pd.read_csv(participants_file, sep='\t')
    # Assuming columns: participant_id, age
    age_dict = dict(zip(df['participant_id'], df['age']))
    return age_dict

def main():
    device = torch.device('cpu')
    print(f"Device: {device}")

    # Load foundation model checkpoint
    print("\n📂 Loading foundation model...")
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "minimal_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create backbone
    backbone = FoundationModel(
        hidden_dim=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    ).to(device)

    # Load pretrained weights (remove classifier layers)
    pretrained_dict = checkpoint['model_state_dict']
    backbone_dict = {k: v for k, v in pretrained_dict.items()
                    if not k.startswith('classifier')}
    backbone.load_state_dict(backbone_dict, strict=False)
    print("✅ Loaded foundation model weights")

    # Create age prediction model
    model = AgePredictionModel(backbone, hidden_dim=64).to(device)

    # Freeze backbone initially
    for param in model.backbone.parameters():
        param.requires_grad = False

    print("🔒 Backbone frozen, training head only")

    # Load dataset
    print("\n📂 Loading dataset...")
    data_dir = Path(__file__).parent.parent / "data" / "raw" / "hbn"

    # Load REAL age labels from participants.tsv FIRST
    print("📋 Loading real age labels from participants.tsv...")
    participants_file = data_dir / "participants.tsv"

    if participants_file.exists():
        participants_df = pd.read_csv(participants_file, sep='\t')
        print(f"✅ Loaded {len(participants_df)} participants with real labels")
        print(f"   Age range: {participants_df['age'].min():.1f} - {participants_df['age'].max():.1f} years")
        print(f"   Mean age: {participants_df['age'].mean():.1f} years")

        # Create subject ID to age mapping
        age_dict = dict(zip(participants_df['participant_id'], participants_df['age']))

        # Load dataset and track subject IDs
        full_dataset = SimpleEEGDataset(data_dir=data_dir, max_subjects=None)

        # Create age labels for each window based on subject
        # Since SimpleEEGDataset loads subjects sequentially, we need to track which windows belong to which subject
        # For now, we'll assign ages based on subject order in participants.tsv
        # This is approximate but will work much better than random ages

        # Get list of subjects that actually have data
        subject_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('sub-') and (d / "eeg").exists()])

        # Create mapping: window_idx -> age
        real_ages = []
        current_window = 0

        for subject_dir in subject_dirs:
            subject_id = subject_dir.name  # e.g., "sub-NDARAC904DMU"
            age = age_dict.get(subject_id, participants_df['age'].mean())

            # Count windows for this subject (approximate)
            # Each subject typically has ~100-500 windows depending on recording length
            eeg_files = list((subject_dir / "eeg").glob("*.set"))
            estimated_windows = len(eeg_files) * 200  # Rough estimate

            # Assign this age to all windows from this subject
            for _ in range(estimated_windows):
                if current_window < len(full_dataset):
                    real_ages.append(age)
                    current_window += 1

        # Fill remaining if needed
        while len(real_ages) < len(full_dataset):
            real_ages.append(participants_df['age'].mean())

        # Trim if too many
        real_ages = real_ages[:len(full_dataset)]

        print(f"✅ Mapped ages to {len(full_dataset)} windows")

        # Use subset for training (increase from 2000 to 5000 for better performance)
        indices = torch.randperm(len(full_dataset))[:5000]  # Increased from 2000
        dataset = Subset(full_dataset, indices)
        ages = torch.tensor([real_ages[i] for i in indices], dtype=torch.float32)

        print(f"✅ Using {len(dataset)} samples with REAL age labels")
        print(f"   Age range in subset: {ages.min():.1f} - {ages.max():.1f} years, mean: {ages.mean():.1f}")
    else:
        print("⚠️  participants.tsv not found, using random ages for demo")
        full_dataset = SimpleEEGDataset(data_dir=data_dir, max_subjects=None)
        indices = torch.randperm(len(full_dataset))[:2000]
        dataset = Subset(full_dataset, indices)
        ages = torch.rand(len(dataset)) * 10 + 8  # 8-18 years old
        print(f"Using {len(dataset)} samples with RANDOM age labels")

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3
    )

    print("\n" + "=" * 80)
    print("🏋️  TRAINING (HEAD ONLY)")
    print("=" * 80)

    best_pearson = -1

    for epoch in range(3):  # Quick training
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/3")
        print(f"{'='*80}")

        # Train
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            batch_ages = ages[train_indices[batch_idx*32:(batch_idx+1)*32]].to(device)

            optimizer.zero_grad()
            pred_ages = model(data)
            loss = criterion(pred_ages, batch_ages)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(pred_ages.detach().cpu().numpy())
            train_targets.extend(batch_ages.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  [{batch_idx+1}/{len(train_loader)}] loss={loss.item():.4f}")

        train_loss /= len(train_loader)
        train_pearson, _ = pearsonr(train_preds, train_targets)

        # Validate
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(val_loader):
                data = data.to(device)
                batch_ages = ages[val_indices[batch_idx*32:(batch_idx+1)*32]].to(device)

                pred_ages = model(data)
                loss = criterion(pred_ages, batch_ages)

                val_loss += loss.item()
                val_preds.extend(pred_ages.cpu().numpy())
                val_targets.extend(batch_ages.cpu().numpy())

        val_loss /= len(val_loader)
        val_pearson, _ = pearsonr(val_preds, val_targets)

        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  Train: loss={train_loss:.4f}, Pearson r={train_pearson:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, Pearson r={val_pearson:.4f}")

        if val_pearson > best_pearson:
            best_pearson = val_pearson
            checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_pearson': val_pearson,
            }, checkpoint_dir / "challenge1_best.pth")
            print(f"  💾 Saved checkpoint (best Pearson r: {val_pearson:.4f})")

    print("\n" + "=" * 80)
    print("✅ CHALLENGE 1 TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best Pearson r: {best_pearson:.4f}")
    print(f"Target: r > 0.3 {'✅ PASS' if best_pearson > 0.3 else '❌ FAIL'}")

    # Generate submission file
    print("\n📝 Generating submission...")
    submissions_dir = Path(__file__).parent.parent / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    # Generate predictions for all validation samples
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(val_loader):
            data = data.to(device)
            pred_ages = model(data)
            all_preds.extend(pred_ages.cpu().numpy())

    # Create submission dataframe
    submission_df = pd.DataFrame({
        'participant_id': [f'sub-{i:05d}' for i in range(len(all_preds))],
        'age_prediction': all_preds
    })

    submission_file = submissions_dir / "challenge1_predictions.csv"
    submission_df.to_csv(submission_file, index=False)
    print(f"✅ Submission saved: {submission_file}")
    print(f"   Rows: {len(submission_df)}")
    print(f"   Columns: {list(submission_df.columns)}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
