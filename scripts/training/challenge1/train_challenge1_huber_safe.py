#!/usr/bin/env python3
"""
Challenge 1: Memory-Safe Training with Huber Loss + Residual Reweighting
==========================================================================

CRITICAL FIXES:
1. Memory-safe: Load releases ONE AT A TIME (prevents PC crash)
2. Huber Loss: Robust to outliers (better than MSE for noisy EEG)
3. Residual Reweighting: Downweight noisy samples dynamically
4. Gradient accumulation: Simulate large batches without memory

Target: < 0.85 NRMSE (15% improvement from 1.00)
"""
import os
import sys
import gc
from pathlib import Path
import time
import warnings
import logging
import traceback
from datetime import datetime

warnings.filterwarnings('ignore')

# Setup logging
log_dir = Path("logs/training_comparison")
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = log_dir / f"challenge1_huber_safe_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%m/%d/%y %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Force CPU (more stable)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HIP_VISIBLE_DEVICES'] = ''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from braindecode.preprocessing import (
    Preprocessor,
    preprocess,
)
from eegdash import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)
from braindecode.preprocessing import create_windows_from_events

print("="*80)
print("🎯 CHALLENGE 1: HUBER LOSS + RESIDUAL REWEIGHTING (MEMORY-SAFE)")
print("="*80)
print("Improvements:")
print("  ✅ Memory-safe: ONE release at a time (no PC crashes)")
print("  ✅ Huber Loss: Robust to outliers (better than MSE)")
print("  ✅ Residual Reweighting: Auto-downweight noisy trials")
print("  ✅ Gradient Accumulation: Simulate large batches")
print("  ✅ Stimulus-aligned windows")
print("  ✅ L1 + L2 + Dropout regularization")
print("Target: < 0.85 NRMSE (15% improvement)")
print("="*80)

def compute_nrmse(y_true, y_pred):
    """Normalized RMSE"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    nrmse = rmse / (np.max(y_true) - np.min(y_true) + 1e-8)
    return nrmse

class ResidualReweightingLoss(nn.Module):
    """
    Dynamically reweight samples based on residuals.
    High-error samples (likely noisy) get lower weight.
    """
    def __init__(self, percentile=75, base_criterion=None):
        super().__init__()
        self.percentile = percentile
        self.base_criterion = base_criterion or nn.HuberLoss(delta=1.0, reduction='none')

    def forward(self, pred, target):
        # Compute per-sample losses
        losses = self.base_criterion(pred, target)

        # Find threshold (e.g., 75th percentile)
        threshold = torch.quantile(losses, self.percentile / 100.0)

        # Samples above threshold get 0.5 weight, others get 1.0
        weights = torch.where(losses > threshold, 0.5, 1.0)

        # Weighted average
        return torch.mean(weights * losses)

class CompactCNN(nn.Module):
    """Compact CNN with dropout"""
    def __init__(self, dropout_p=0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_p * 0.6),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_p * 0.8),

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_p * 0.8),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        return x

def load_single_release_safe(release, mini=False):
    """
    Load ONE release safely with memory cleanup
    """
    logger.info(f"Loading {release} (mini={mini})...")
    print(f"\n📂 Loading {release}...")

    try:
        # Load dataset
        dataset = EEGChallengeDataset(
            release=release,
            mini=mini,
            query=dict(task="contrastChangeDetection"),  # CHALLENGE 1 task
            description_fields=["target", "correct", "response_type"],
            cache_dir=Path("data/raw")
        )
        logger.info(f"  {release}: Loaded {len(dataset.datasets)} datasets")

        # Preprocessing
        print(f"  Preprocessing...")
        preprocessors = [
            Preprocessor("pick", ch_names=dataset.datasets[0].raw.ch_names, ordered=True),
            Preprocessor(annotate_trials_with_target, apply_on_array=False),
            Preprocessor(add_aux_anchors, apply_on_array=False),
        ]
        dataset = preprocess(dataset, preprocessors)

        # Filter for stimulus_anchor
        print(f"  Filtering for stimulus_anchor...")
        dataset = keep_only_recordings_with("stimulus_anchor", dataset)
        print(f"  Datasets with stimulus_anchor: {len(dataset.datasets)}")

        if len(dataset.datasets) == 0:
            logger.warning(f"  {release}: No datasets with stimulus_anchor!")
            return None

        # Create windows
        print(f"  Creating stimulus-aligned windows...")
        SFREQ = 100
        windows_dataset = create_windows_from_events(
            dataset,
            mapping={"stimulus_anchor": 0},
            trial_start_offset_samples=int(0.5 * SFREQ),
            trial_stop_offset_samples=int(2.5 * SFREQ),
            window_size_samples=int(2.0 * SFREQ),
            window_stride_samples=SFREQ,
            preload=True,
        )

        # Add metadata
        print(f"  Adding metadata...")
        windows_dataset = add_extras_columns(
            windows_dataset,
            dataset,
            desc="stimulus_anchor",
            keys=("rt_from_stimulus",)
        )

        # Extract data
        metadata_df = windows_dataset.get_metadata()
        rt_values = metadata_df['rt_from_stimulus'].values

        # Remove invalid RT values
        valid_mask = ~np.isnan(rt_values) & (rt_values >= 0) & (rt_values <= 10)
        valid_indices = np.where(valid_mask)[0]

        print(f"  Valid windows: {len(valid_indices)}/{len(rt_values)}")
        logger.info(f"  {release}: {len(valid_indices)} valid windows")

        if len(valid_indices) == 0:
            return None

        # Get data and labels
        X_list = []
        y_list = []
        for idx in valid_indices:
            X_list.append(windows_dataset[idx][0])
            y_list.append(rt_values[idx])

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"  Data shape: {X.shape}, Labels shape: {y.shape}")
        print(f"  RT stats: mean={y.mean():.3f}, std={y.std():.3f}, range=[{y.min():.3f}, {y.max():.3f}]")

        # Cleanup
        del dataset, windows_dataset, metadata_df
        gc.collect()

        return X, y

    except Exception as e:
        logger.error(f"  {release}: Failed to load - {e}")
        logger.error(traceback.format_exc())
        return None

class RTDataset(Dataset):
    """Simple RT dataset"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_with_gradient_accumulation(
    model, train_loader, val_loader,
    criterion, optimizer, scheduler,
    epochs=50, accumulation_steps=4,
    l1_lambda=1e-5, patience=15
):
    """
    Train with gradient accumulation and memory efficiency
    """
    best_nrmse = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*80}")

        # Train
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        epoch_l1_penalty = 0

        optimizer.zero_grad()

        for i, (data, labels) in enumerate(train_loader):
            # Forward
            outputs = model(data)

            # Loss (Huber + Residual Reweighting)
            loss = criterion(outputs, labels)

            # L1 penalty
            l1_penalty = sum(torch.abs(p).sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_penalty

            # Scale loss for accumulation
            loss = loss / accumulation_steps
            loss.backward()

            # Step every accumulation_steps
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps
            epoch_l1_penalty += l1_penalty.item()
            train_preds.extend(outputs.detach().numpy().flatten())
            train_labels.extend(labels.numpy().flatten())

        # Validate
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for data, labels in val_loader:
                outputs = model(data)
                val_preds.extend(outputs.numpy().flatten())
                val_labels.extend(labels.numpy().flatten())

        # Metrics
        train_nrmse = compute_nrmse(np.array(train_labels), np.array(train_preds))
        val_nrmse = compute_nrmse(np.array(val_labels), np.array(val_preds))
        avg_l1 = epoch_l1_penalty / len(train_loader)

        print(f"Train NRMSE: {train_nrmse:.4f}  |  L1: {avg_l1:.2e}")
        print(f"Val NRMSE:   {val_nrmse:.4f}")

        scheduler.step()

        # Early stopping
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            patience_counter = 0
            torch.save(model.state_dict(), 'weights_challenge_1_huber_safe.pt')
            print(f"✅ Best model saved (NRMSE: {best_nrmse:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping (no improvement for {patience} epochs)")
                break

    return best_nrmse

def main():
    """Main training function"""
    print("\n📦 Loading data release-by-release (memory-safe)...")

    # Load releases ONE AT A TIME
    all_X = []
    all_y = []

    for release in ['R1', 'R2', 'R3', 'R4']:
        result = load_single_release_safe(release, mini=False)
        if result is not None:
            X, y = result
            all_X.append(X)
            all_y.append(y)
            print(f"  ✅ {release}: {X.shape[0]} windows loaded")

            # Force cleanup
            gc.collect()
        else:
            print(f"  ❌ {release}: Failed to load")

    if len(all_X) == 0:
        print("❌ No data loaded!")
        return

    # Combine
    print("\n📊 Combining releases...")
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"  Total: {X.shape[0]} windows")
    print(f"  Data shape: {X.shape}")
    print(f"  RT stats: mean={y.mean():.3f}, std={y.std():.3f}")

    # Clean up
    del all_X, all_y
    gc.collect()

    # Split
    train_size = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    print(f"  Train: {len(X_train)} windows")
    print(f"  Val:   {len(X_val)} windows")

    # Datasets
    train_dataset = RTDataset(X_train, y_train)
    val_dataset = RTDataset(X_val, y_val)

    # DataLoaders (small batch size for memory)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Model
    print("\n🏗️  Building model...")
    model = CompactCNN(dropout_p=0.5)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Loss and optimizer
    print("\n⚙️  Configuring training...")
    criterion = ResidualReweightingLoss(percentile=75)  # Huber + Reweighting
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    print("  Loss: Huber + Residual Reweighting (robust to outliers)")
    print("  Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)")
    print("  Regularization: L1 (1e-5) + L2 (1e-4) + Dropout (0.5)")
    print("  Gradient Accumulation: 4 steps (simulate batch_size=128)")

    # Train
    print("\n🚀 Starting training...")
    logger.info("Training started")

    best_nrmse = train_with_gradient_accumulation(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        epochs=50,
        accumulation_steps=4,
        l1_lambda=1e-5,
        patience=15
    )

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation NRMSE: {best_nrmse:.4f}")
    print(f"Target: < 0.85 (15% improvement from 1.00 baseline)")
    print(f"Model saved: weights_challenge_1_huber_safe.pt")
    print("="*80)

    logger.info(f"✅ Training completed successfully! Best NRMSE: {best_nrmse:.4f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error(traceback.format_exc())
        raise

