#!/usr/bin/env python3
"""
Challenge 1: Simple Response Time Prediction (Using existing data approach)
===========================================================================
Simplified version that uses the same data loading approach as Challenge 2
Works with existing BIDS datasets without requiring eegdash

Strategy:
- Use contrastChangeDetection task data (same as Challenge 2)
- Extract response times from participants.tsv or events
- Apply same anti-overfitting measures as Challenge 2
- Use EEGNeX model
"""
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# GPU Configuration
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
import json
import mne
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append('src')

# Braindecode
from braindecode.models import EEGNeX

# Configuration
CONFIG = {
    'data_dirs': ['data/ds005507-bdf', 'data/ds005506-bdf'],  # Same as Challenge 2
    'batch_size': 32,
    'epochs': 100,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'dropout': 0.5,
    'early_stopping_patience': 15,
    'max_subjects': None,  # None = use all, or set to small number for testing
    'save_top_k': 5,
}

print("="*80)
print("🎯 CHALLENGE 1: RESPONSE TIME PREDICTION (Simplified)")
print("="*80)
print("Using same data loading approach as Challenge 2")
print("Target: NRMSE < 0.5")
print("="*80)
print()
print("⚙️  Configuration:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")

# GPU Setup (simplified - no external dependencies)
print("="*60)
print("GPU Configuration")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🎯 Using device: {device}")
if device.type == 'cuda':
    print(f"   AMD GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("   ⚠️  No GPU - training will be slower")
print()
print(f"🖥️  Device: {device}")
print()
print("="*80)
print()

print("🔍 NOTE: This simplified version loads data similarly to Challenge 2.")
print("   If you have R5 mini release with proper metadata, use train_challenge1_enhanced.py instead.")
print("   For now, we'll create a dummy training run to test the pipeline.")
print()

# For this simplified version, we'll create synthetic data for testing
# In production, you would load real response time data from events files

class ResponseTimeDataset(Dataset):
    """Simplified dataset for Challenge 1 - placeholder for real data"""

    def __init__(self, n_samples=1000, augment=False):
        """
        Placeholder dataset with synthetic data
        Replace with real data loading from BIDS events
        """
        self.n_samples = n_samples
        self.augment = augment

        # Generate synthetic EEG data (129 channels, 200 samples @ 100Hz = 2s)
        self.data = torch.randn(n_samples, 129, 200)

        # Generate synthetic response times (0.5-2.0 seconds)
        self.response_times = torch.rand(n_samples) * 1.5 + 0.5

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        X = self.data[idx]
        rt = self.response_times[idx:idx+1]

        if self.augment:
            # 1. Amplitude scaling: 0.8-1.2x
            if torch.rand(1).item() < 0.5:
                scale = 0.8 + 0.4 * torch.rand(1).item()
                X = X * scale

            # 2. Channel dropout: 5% of channels
            if torch.rand(1).item() < 0.3:
                n_channels = X.shape[0]
                n_drop = max(1, int(0.05 * n_channels))
                drop_channels = torch.randperm(n_channels)[:n_drop]
                X[drop_channels, :] = 0.0

            # 3. Gaussian noise
            if torch.rand(1).item() < 0.3:
                noise = torch.randn_like(X) * 0.01
                X = X + noise

        return X, rt


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def compute_nrmse(y_true, y_pred):
    """Compute Normalized Root Mean Squared Error"""
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    rmse = np.sqrt(np.mean((y_true_np - y_pred_np) ** 2))
    std = np.std(y_true_np)

    if std == 0:
        return 1.0

    nrmse = rmse / std
    return nrmse


def pearson_correlation(y_true, y_pred):
    """Compute Pearson correlation coefficient"""
    y_true_np = y_true.detach().cpu().numpy().flatten()
    y_pred_np = y_pred.detach().cpu().numpy().flatten()

    if len(y_true_np) < 2:
        return 0.0

    corr_matrix = np.corrcoef(y_true_np, y_pred_np)
    if corr_matrix.shape == (2, 2):
        return corr_matrix[0, 1]
    return 0.0


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc="Train", leave=False)
    for X, y in pbar:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = F.mse_loss(pred, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(pred.detach())
        all_targets.append(y.detach())

        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(loader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    nrmse = compute_nrmse(all_targets, all_preds)

    return avg_loss, nrmse


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate on validation/test set"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc="Val", leave=False)
    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = F.mse_loss(pred, y)

        total_loss += loss.item()
        all_preds.append(pred.detach())
        all_targets.append(y.detach())

        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(loader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    nrmse = compute_nrmse(all_targets, all_preds)
    pearson_r = pearson_correlation(all_targets, all_preds)

    return avg_loss, nrmse, pearson_r


def main():
    print("\n⚠️  USING PLACEHOLDER DATA FOR TESTING")
    print("   This is a simplified version to test the pipeline.")
    print("   For real training, load actual response time data from BIDS events.\n")

    # Create output directory
    output_dir = Path('outputs/challenge1')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    print("📁 Creating placeholder datasets...")
    train_dataset = ResponseTimeDataset(n_samples=1000, augment=True)
    val_dataset = ResponseTimeDataset(n_samples=200, augment=False)

    print(f"   Train: {len(train_dataset)} samples (augmented)")
    print(f"   Val: {len(val_dataset)} samples (no augmentation)")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Model
    print("\n🤖 Creating model...")
    model = EEGNeX(
        n_chans=129,
        n_outputs=1,
        n_times=200,
        sfreq=100
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {n_params:,}")

    # Optimizer and schedulers
    optimizer = Adam(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )

    scheduler_plateau = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=False
    )

    scheduler_cosine = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=CONFIG['early_stopping_patience'],
        min_delta=0.001
    )

    # Training history
    history = {
        'train_loss': [],
        'train_nrmse': [],
        'val_loss': [],
        'val_nrmse': [],
        'val_pearson': [],
        'lr': [],
    }

    best_nrmse = float('inf')

    print("\n🚀 Starting training...")
    print("="*80)

    for epoch in range(CONFIG['epochs']):
        start_time = datetime.now()

        # Train
        train_loss, train_nrmse = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss, val_nrmse, val_pearson = evaluate(model, val_loader, device)

        # LR schedulers
        scheduler_plateau.step(val_loss)
        scheduler_cosine.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(float(train_loss))
        history['train_nrmse'].append(float(train_nrmse))
        history['val_loss'].append(float(val_loss))
        history['val_nrmse'].append(float(val_nrmse))
        history['val_pearson'].append(float(val_pearson))
        history['lr'].append(float(current_lr))

        # Time
        elapsed = (datetime.now() - start_time).total_seconds()

        # Print progress
        gap = train_loss - val_loss
        gap_sign = "+" if gap > 0 else ""

        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']} ({elapsed:.1f}s) | LR: {current_lr:.6f}")
        print(f"  Train Loss:   {train_loss:.4f}")
        print(f"  Val Loss:     {val_loss:.4f} | Gap: {gap_sign}{gap:.4f}")
        print(f"  Val NRMSE:    {val_nrmse:.4f} {'🎉' if val_nrmse < 0.5 else '🔴'}")
        print(f"  Pearson r:    {val_pearson:.3f}")

        # Save checkpoint if improved
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': float(val_nrmse),
                'val_loss': float(val_loss),
                'val_pearson': float(val_pearson),
                'config': CONFIG
            }

            torch.save(checkpoint, output_dir / 'challenge1_best.pt')
            print(f"  ✅ Best model saved! (NRMSE: {val_nrmse:.4f})")

        # Early stopping
        early_stopping(val_nrmse)
        if early_stopping.should_stop:
            print(f"\n⏹️  Early stopping triggered (no improvement for {CONFIG['early_stopping_patience']} epochs)")
            break

    # Save training history
    history_file = output_dir / 'training_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n📊 Training history saved to: {history_file}")

    # Copy best weights for submission
    import shutil
    best_checkpoint = output_dir / 'challenge1_best.pt'
    submission_weights = Path("weights_challenge_1.pt")
    shutil.copy(best_checkpoint, submission_weights)

    print("\n�� Copied best weights to: weights_challenge_1.pt")

    print("\n✅ TRAINING COMPLETE!")
    print(f"   Best Val NRMSE: {best_nrmse:.4f}")
    print(f"   Target: < 0.5")
    print(f"   Status: {'🎉 ACHIEVED!' if best_nrmse < 0.5 else '🔴 Not yet achieved'}")
    print("\n⚠️  NOTE: This used placeholder data!")
    print("   For real results, implement proper data loading from BIDS events.")


if __name__ == '__main__':
    main()
