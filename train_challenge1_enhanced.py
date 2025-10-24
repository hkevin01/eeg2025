#!/usr/bin/env python3
"""
Challenge 1: Enhanced Response Time Prediction with Anti-Overfitting
===================================================================
Following Challenge 2's success (NRMSE: 0.0918), applying similar strategy:
- Standard braindecode EEGNeX model (small, proven architecture)
- Comprehensive data augmentation
- Strong regularization
- Early stopping with patience
- Top-5 ensemble capability

Target: NRMSE < 0.5 (competitive submission)
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
from pathlib import Path

# Add src to path
sys.path.append('src')

# Braindecode and EEGDash
from braindecode.models import EEGNeX
from braindecode.preprocessing import Preprocessor, preprocess, create_windows_from_events
from eegdash.dataset import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)

# Configuration
CONFIG = {
    'data_dir': 'data',
    'releases': ['R5'],  # Start with R5 mini, can expand to R1-R5 later
    'mini': True,  # Set to False for full dataset
    'batch_size': 32,
    'epochs': 100,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'dropout': 0.5,
    'early_stopping_patience': 15,
    'save_top_k': 5,
    'output_dir': 'outputs/challenge1',
    'checkpoint_prefix': 'challenge1_eegnex',
}

print("="*80)
print("🎯 CHALLENGE 1: ENHANCED RESPONSE TIME PREDICTION (Anti-Overfitting)")
print("="*80)
print("Target: NRMSE < 0.5 | Competition: Code Submission")
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


class AugmentedResponseTimeDataset(Dataset):
    """Dataset with augmentation for Challenge 1"""

    def __init__(self, windows_dataset, augment=True):
        """
        Args:
            windows_dataset: EEG windows with response times
            augment: Apply data augmentation (for training)
        """
        self.windows_dataset = windows_dataset
        self.augment = augment

    def __len__(self):
        return len(self.windows_dataset)

    def __getitem__(self, idx):
        # Get window and metadata
        X, y, _ = self.windows_dataset[idx]

        # Convert to torch tensor
        X = torch.FloatTensor(X)

        # Get response time from metadata
        metadata = self.windows_dataset.get_metadata()
        rt = metadata.iloc[idx]['rt_from_stimulus']
        rt = torch.FloatTensor([rt])

        if self.augment:
            # 1. Amplitude scaling: 0.8-1.2x
            if torch.rand(1).item() < 0.5:
                scale = 0.8 + 0.4 * torch.rand(1).item()
                X = X * scale

            # 2. Channel dropout: 5% of channels (30% of samples)
            if torch.rand(1).item() < 0.3:
                n_channels = X.shape[0]
                n_drop = max(1, int(0.05 * n_channels))
                drop_channels = torch.randperm(n_channels)[:n_drop]
                X[drop_channels, :] = 0.0

            # 3. Gaussian noise: small amount
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

    return np.corrcoef(y_true_np, y_pred_np)[0, 1]


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

        # Gradient clipping
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
    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("📁 Loading data...")
    datasets = []
    for release in CONFIG['releases']:
        print(f"   Loading {release}...")
        dataset = EEGChallengeDataset(
            task="contrastChangeDetection",
            release=release,
            cache_dir=CONFIG['data_dir'],
            mini=CONFIG['mini']
        )
        datasets.append(dataset)
        print(f"   ✅ {release}: {len(dataset.datasets)} subjects")

    # Combine datasets
    from braindecode.datasets import BaseConcatDataset
    combined_dataset = BaseConcatDataset(datasets)

    # Preprocess: Add stimulus-aligned metadata
    print("\n📊 Preprocessing...")
    EPOCH_LEN_S = 2.0
    SFREQ = 100

    transformation_offline = [
        Preprocessor(
            annotate_trials_with_target,
            target_field="rt_from_stimulus",
            epoch_length=EPOCH_LEN_S,
            require_stimulus=True,
            require_response=True,
            apply_on_array=False,
        ),
        Preprocessor(add_aux_anchors, apply_on_array=False),
    ]

    preprocess(combined_dataset, transformation_offline, n_jobs=1)

    # Filter to only recordings with stimulus anchors
    ANCHOR = "stimulus_anchor"
    SHIFT_AFTER_STIM = 0.5
    WINDOW_LEN = 2.0

    dataset_filtered = keep_only_recordings_with(ANCHOR, combined_dataset)
    print(f"   ✅ Filtered to {len(dataset_filtered.datasets)} recordings with stimulus anchors")

    # Create windows
    print("\n🪟 Creating windows...")
    windows_dataset = create_windows_from_events(
        dataset_filtered,
        mapping={ANCHOR: 0},
        trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),
        trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),
        window_size_samples=int(EPOCH_LEN_S * SFREQ),
        window_stride_samples=SFREQ,
        drop_bad_windows=True,  # Drop windows that fall outside recording boundaries
        preload=True,
    )

    print(f"   ✅ Created {len(windows_dataset)} windows")

    # CRITICAL: Inject response time metadata into windows
    print("\n📝 Injecting response time metadata...")
    try:
        windows_dataset = add_extras_columns(
            windows_dataset,
            dataset_filtered,
            desc=ANCHOR,
            keys=("rt_from_stimulus",)
        )
        print("   ✅ Metadata injected")
    except AttributeError as e:
        print(f"   ⚠️ Using alternative metadata extraction: {e}")
        # Fallback: Extract metadata directly
        windows_dataset.get_metadata()  # Trigger metadata creation
        windows_dataset = add_extras_columns(
            windows_dataset,
            dataset_filtered,
            desc=ANCHOR,
            keys=("rt_from_stimulus",)
        )
        print("   ✅ Metadata injected (alternative method)")

    # Create datasets with/without augmentation
    print("\n📦 Creating train/val datasets...")
    train_dataset = AugmentedResponseTimeDataset(windows_dataset, augment=True)
    val_dataset = AugmentedResponseTimeDataset(windows_dataset, augment=False)

    # Split
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, _ = random_split(train_dataset, [train_size, val_size])

    val_split_size = int(0.2 * len(val_dataset))
    _, val_dataset = random_split(val_dataset, [len(val_dataset) - val_split_size, val_split_size])

    print(f"   Train: {len(train_dataset)} windows (augmented)")
    print(f"   Val: {len(val_dataset)} windows (no augmentation)")

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
        verbose=True
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
    top_k_checkpoints = []

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
        history['train_loss'].append(train_loss)
        history['train_nrmse'].append(train_nrmse)
        history['val_loss'].append(val_loss)
        history['val_nrmse'].append(val_nrmse)
        history['val_pearson'].append(val_pearson)
        history['lr'].append(current_lr)

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

        # Save checkpoint if in top-k
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': val_nrmse,
                'val_loss': val_loss,
                'val_pearson': val_pearson,
                'config': CONFIG
            }

            # Add to top-k
            top_k_checkpoints.append((val_nrmse, checkpoint))
            top_k_checkpoints.sort(key=lambda x: x[0])
            top_k_checkpoints = top_k_checkpoints[:CONFIG['save_top_k']]

            # Save current best
            torch.save(
                checkpoint,
                output_dir / f"{CONFIG['checkpoint_prefix']}_best.pt"
            )

            print(f"  ✅ Best model saved! (NRMSE: {val_nrmse:.4f})")

        # Early stopping
        early_stopping(val_nrmse)
        if early_stopping.should_stop:
            print(f"\n⏹️  Early stopping triggered (no improvement for {CONFIG['early_stopping_patience']} epochs)")
            break

    # Save top-k checkpoints
    print("\n💾 Saving top-5 checkpoints...")
    for i, (nrmse, checkpoint) in enumerate(top_k_checkpoints):
        torch.save(
            checkpoint,
            output_dir / f"{CONFIG['checkpoint_prefix']}_top{i+1}_nrmse{nrmse:.4f}.pt"
        )
        print(f"   #{i+1}: NRMSE {nrmse:.4f}")

    # Save training history
    history_file = output_dir / 'training_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n�� Training history saved to: {history_file}")

    # Copy best weights for submission
    import shutil
    best_checkpoint = output_dir / f"{CONFIG['checkpoint_prefix']}_best.pt"
    submission_weights = Path("weights_challenge_1.pt")
    shutil.copy(best_checkpoint, submission_weights)

    print("\n📦 Copied best weights to: weights_challenge_1.pt")

    print("\n✅ TRAINING COMPLETE!")
    print(f"   Best Val NRMSE: {best_nrmse:.4f}")
    print(f"   Target: < 0.5")
    print(f"   Status: {'🎉 ACHIEVED!' if best_nrmse < 0.5 else '🔴 Not yet achieved'}")


if __name__ == '__main__':
    main()
