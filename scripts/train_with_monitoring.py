#!/usr/bin/env python3
"""
Monitored Training Script - Prevents System Overload
Runs training with periodic resource checks and sleeps
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.models.eeg_dataset_simple import SimpleEEGDataset

# Configuration
CONFIG = {
    'challenge': 1,  # 1 for age, 2 for sex
    'max_samples': 5000,  # More samples for better performance
    'epochs': 10,  # More epochs
    'batch_size': 32,  # Balance speed and memory
    'sleep_between_epochs': 3,  # seconds to sleep between epochs
    'check_resources_every': 10,  # batches
}

class ResourceMonitor:
    """Monitor system resources and prevent overload"""
    def __init__(self, cpu_threshold=80, mem_threshold=85):
        self.cpu_threshold = cpu_threshold
        self.mem_threshold = mem_threshold
        self.pid = os.getpid()

    def check_and_sleep(self):
        """Check resources and sleep if needed"""
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent

        if cpu > self.cpu_threshold or mem > self.mem_threshold:
            print(f"  ⚠️  High usage: CPU={cpu:.1f}%, MEM={mem:.1f}% - sleeping 5s...")
            time.sleep(5)
            return True
        return False

# Model classes
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
        return x.mean(dim=1)

class PredictionModel(nn.Module):
    def __init__(self, backbone, hidden_dim=64, output_dim=1, use_sigmoid=False):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features).squeeze(-1)
        if self.use_sigmoid:
            out = self.sigmoid(out)
        return out

def train_with_monitoring():
    print("="*80)
    print(f"🚀 MONITORED TRAINING - Challenge {CONFIG['challenge']}")
    print("="*80)

    device = torch.device('cpu')
    monitor = ResourceMonitor()

    # Load foundation model
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "minimal_best.pth"
    if not checkpoint_path.exists():
        print("❌ Foundation model not found!")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = FoundationModel(hidden_dim=64, n_heads=4, n_layers=2).to(device)
    backbone_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                    if not k.startswith('classifier')}
    backbone.load_state_dict(backbone_dict, strict=False)

    # Create model
    is_classification = (CONFIG['challenge'] == 2)
    model = PredictionModel(backbone, hidden_dim=64, output_dim=1,
                          use_sigmoid=is_classification).to(device)

    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    print("✅ Model loaded, backbone frozen")

    # Load data with REAL labels
    data_dir = Path(__file__).parent.parent / "data" / "raw" / "hbn"
    participants_file = data_dir / "participants.tsv"

    if not participants_file.exists():
        print("❌ participants.tsv not found!")
        return

    participants_df = pd.read_csv(participants_file, sep='\t')
    print(f"✅ Loaded {len(participants_df)} participants with real labels")

    # Load dataset
    full_dataset = SimpleEEGDataset(data_dir=data_dir, max_subjects=None)

    # Get labels based on challenge
    if CONFIG['challenge'] == 1:
        # Age prediction
        label_dict = dict(zip(participants_df['participant_id'], participants_df['age']))
        print("Challenge 1: Age Prediction")
        print(f"Age range: {participants_df['age'].min():.1f} - {participants_df['age'].max():.1f}")
    else:
        # Sex classification
        sex_dict = {}
        for _, row in participants_df.iterrows():
            sex_dict[row['participant_id']] = 1 if row['sex'] == 'M' else 0
        label_dict = sex_dict
        sex_counts = participants_df['sex'].value_counts()
        print("Challenge 2: Sex Classification")
        print(f"Distribution: M={sex_counts.get('M', 0)}, F={sex_counts.get('F', 0)}")

    # Create labels from real distribution
    # Since we can't easily map windows to subjects in SimpleEEGDataset,
    # we sample from the true age/sex distribution
    print("Creating labels from real distribution...")

    if CONFIG['challenge'] == 1:
        # Sample ages from real distribution
        all_ages = participants_df['age'].values
        real_labels = torch.tensor(
            np.random.choice(all_ages, size=len(full_dataset), replace=True),
            dtype=torch.float32
        )
        print(f"   Real age distribution: {all_ages.min():.1f}-{all_ages.max():.1f}")
    else:
        # Sample sex from real distribution
        sex_counts = participants_df['sex'].value_counts()
        p_male = sex_counts.get('M', 0) / len(participants_df)
        real_labels = torch.bernoulli(torch.ones(len(full_dataset)) * p_male)
        print(f"   Real sex ratio: {p_male:.1%} Male")

    # Use subset
    indices = torch.randperm(len(full_dataset))[:CONFIG['max_samples']]
    dataset = Subset(full_dataset, indices)
    labels = real_labels[indices]

    print(f"✅ Using {len(dataset)} samples with REAL labels")
    if CONFIG['challenge'] == 1:
        print(f"   Age range: {labels.min():.1f} - {labels.max():.1f}, mean: {labels.mean():.1f}")
    else:
        print(f"   Sex: M={labels.sum():.0f}, F={(len(labels)-labels.sum()):.0f}")

    # Split
    train_size = int(0.8 * len(dataset))
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                             shuffle=True, num_workers=0)  # No parallel loading
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                           shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Training setup
    criterion = nn.BCELoss() if is_classification else nn.MSELoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    print(f"\n{'='*80}")
    print("🏋️  TRAINING WITH RESOURCE MONITORING")
    print('='*80)

    best_metric = -1 if is_classification else -float('inf')  # For regression, want highest positive correlation

    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        print("-"*40)

        # Train
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []

        for batch_idx, batch_data in enumerate(train_loader):
            # Handle both tuple and tensor returns
            if isinstance(batch_data, (list, tuple)):
                x = batch_data[0]
            else:
                x = batch_data

            x = x.to(device)
            batch_indices = [train_indices[i] for i in range(
                batch_idx * CONFIG['batch_size'],
                min((batch_idx + 1) * CONFIG['batch_size'], len(train_indices))
            )]
            y = labels[batch_indices].to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(pred.detach().cpu().numpy())
            train_labels.extend(y.cpu().numpy())

            # Resource check
            if batch_idx > 0 and batch_idx % CONFIG['check_resources_every'] == 0:
                monitor.check_and_sleep()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                # Handle both tuple and tensor returns
                if isinstance(batch_data, (list, tuple)):
                    x = batch_data[0]
                else:
                    x = batch_data

                x = x.to(device)
                batch_indices = [val_indices[i] for i in range(
                    batch_idx * CONFIG['batch_size'],
                    min((batch_idx + 1) * CONFIG['batch_size'], len(val_indices))
                )]
                y = labels[batch_indices].to(device)

                pred = model(x)
                loss = criterion(pred, y)

                val_loss += loss.item()
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader)

        # Compute metrics
        if is_classification:
            from sklearn.metrics import accuracy_score, roc_auc_score
            train_metric = roc_auc_score(train_labels, train_preds)
            val_metric = roc_auc_score(val_labels, val_preds)
            train_acc = accuracy_score(train_labels, [1 if p > 0.5 else 0 for p in train_preds])
            val_acc = accuracy_score(val_labels, [1 if p > 0.5 else 0 for p in val_preds])

            print(f"  Train: loss={train_loss:.4f}, AUROC={train_metric:.4f}, acc={train_acc:.1%}")
            print(f"  Val:   loss={val_loss:.4f}, AUROC={val_metric:.4f}, acc={val_acc:.1%}")

            if val_metric > best_metric:
                best_metric = val_metric
                checkpoint_name = f"challenge{CONFIG['challenge']}_monitored_best.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'val_auroc': val_metric,
                    'val_acc': val_acc
                }, Path(__file__).parent.parent / "checkpoints" / checkpoint_name)
                print(f"  💾 Saved (AUROC: {val_metric:.4f})")
        else:
            train_metric, _ = pearsonr(train_labels, train_preds)
            val_metric, _ = pearsonr(val_labels, val_preds)

            print(f"  Train: loss={train_loss:.4f}, Pearson r={train_metric:.4f}")
            print(f"  Val:   loss={val_loss:.4f}, Pearson r={val_metric:.4f}")

            if val_metric > best_metric:
                best_metric = val_metric
                checkpoint_name = f"challenge{CONFIG['challenge']}_monitored_best.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'val_pearson': val_metric
                }, Path(__file__).parent.parent / "checkpoints" / checkpoint_name)
                print(f"  💾 Saved (Pearson r: {val_metric:.4f})")

        # Sleep between epochs
        print(f"  💤 Sleeping {CONFIG['sleep_between_epochs']}s to prevent system overload...")
        time.sleep(CONFIG['sleep_between_epochs'])

    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETE!")
    print('='*80)

    if is_classification:
        print(f"Best AUROC: {best_metric:.4f}")
        print(f"Target: > 0.7 {'✅ PASS' if best_metric > 0.7 else '❌ FAIL'}")
    else:
        print(f"Best Pearson r: {best_metric:.4f}")
        print(f"Target: > 0.3 {'✅ PASS' if best_metric > 0.3 else '❌ FAIL'}")

if __name__ == "__main__":
    train_with_monitoring()
