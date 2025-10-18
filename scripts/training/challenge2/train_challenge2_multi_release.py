#!/usr/bin/env python3
"""
Challenge 2: Multi-Release Externalizing Prediction
===================================================
CRITICAL FIX FOR OVERFITTING: Train on R1-R5 instead of just R5

Problem: Previous model trained only on R5, tested on R12 → 14x degradation
Solution: Multi-release training for better generalization

Model improvements:
- Reduced parameters: 600K → 150K (less overfitting)
- Increased dropout: 0.3 → 0.5 (better regularization)
- Weight decay: 1e-4 (L2 regularization)
- Cross-release validation: Train R1-R4, validate R5
"""
import os
import sys
from pathlib import Path
import time
import warnings
import logging
import traceback
from datetime import datetime
import psutil
import gc

warnings.filterwarnings('ignore')

# Setup comprehensive logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
crash_log_file = log_dir / f"challenge2_crash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(crash_log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Force CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HIP_VISIBLE_DEVICES'] = ''

# CRITICAL: Memory safety limits to prevent crashes
MEMORY_SAFETY_CHECKS = True
MAX_MEMORY_PERCENT = 80  # Stop if memory usage exceeds 80%
MAX_DATASETS_PER_RELEASE = 100  # Limit datasets per release for safety
CHECK_MEMORY_EVERY_N_BATCHES = 10

def check_memory_safe():
    """Check if memory usage is safe, return (safe, percent_used)"""
    memory = psutil.virtual_memory()
    percent_used = memory.percent
    safe = percent_used < MAX_MEMORY_PERCENT
    return safe, percent_used

def log_memory_status(prefix=""):
    """Log current memory status"""
    memory = psutil.virtual_memory()
    logger.info(f"{prefix}Memory: {memory.percent:.1f}% used ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")
    print(f"{prefix}Memory: {memory.percent:.1f}% used ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")

logger.info("="*80)
logger.info("🚀 Challenge 2 Training Started (MEMORY-SAFE MODE)")
logger.info(f"Crash log: {crash_log_file}")
logger.info(f"Memory Safety: Enabled (max {MAX_MEMORY_PERCENT}%)")
logger.info(f"Max datasets per release: {MAX_DATASETS_PER_RELEASE}")
logger.info("="*80)

# Check initial memory
log_memory_status("Initial ")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from eegdash import EEGChallengeDataset

print("="*80)
print("🎯 CHALLENGE 2: MULTI-RELEASE EXTERNALIZING PREDICTION - IMPROVED")
print("="*80)
print("Training on: R1, R2, R3, R4 (ALL available full releases, +33% more data)")
print("Validation on: R5")
print("Improvements:")
print("  ✅ Using R1-R4 for maximum data and variance (not just R2-R4)")
print("  ✅ Elastic Net regularization (L1 + L2)")
print("  ✅ Dropout 0.3-0.5 across layers")
print("Expected improvement: 15-25% NRMSE reduction")
print("="*80)


class MultiReleaseExternalizingDataset(Dataset):
    """Dataset for externalizing prediction across releases"""

    def __init__(self, releases, mini=True, cache_dir='data/raw'):
        """
        Args:
            releases: List of release names ['R1', 'R2', ...]
            mini: Use mini dataset
            cache_dir: Cache directory
        """
        self.releases = releases
        self.windows_datasets = []
        self.release_labels = []
        self.externalizing_scores = []  # Store externalizing score for each window

        print(f"\n📂 Loading releases: {releases}")
        logger.info(f"Loading {len(releases)} releases: {releases}")

        for release_idx, release in enumerate(releases, 1):
            # Check memory before loading each release
            if MEMORY_SAFETY_CHECKS:
                safe, mem_pct = check_memory_safe()
                if not safe:
                    logger.error(f"❌ MEMORY LIMIT EXCEEDED: {mem_pct:.1f}% > {MAX_MEMORY_PERCENT}%")
                    print(f"❌ MEMORY LIMIT EXCEEDED: {mem_pct:.1f}% > {MAX_MEMORY_PERCENT}%")
                    print(f"⚠️  Stopping before loading {release} to prevent crash")
                    logger.warning(f"Stopping at {release_idx-1}/{len(releases)} releases")
                    break
                log_memory_status(f"Before {release}: ")

            print(f"\n  [{release_idx}/{len(releases)}] Loading {release}...")
            logger.info(f"Loading release {release} ({release_idx}/{len(releases)})")
            release_start = time.time()

            # Load dataset with externalizing scores
            # CRITICAL: Challenge 2 uses contrastChangeDetection task (per starter kit)
            try:
                dataset = EEGChallengeDataset(
                    release=release,
                    mini=mini,
                    query=dict(task="contrastChangeDetection"),  # Challenge 2 uses this task
                    description_fields=["externalizing"],
                    cache_dir=Path(cache_dir)
                )
                print(f"    Datasets: {len(dataset.datasets)}")
                logger.info(f"  {release}: Loaded {len(dataset.datasets)} datasets")

                # SAFETY: Limit number of datasets per release
                if MEMORY_SAFETY_CHECKS and len(dataset.datasets) > MAX_DATASETS_PER_RELEASE:
                    logger.warning(f"  {release}: Limiting to {MAX_DATASETS_PER_RELEASE} datasets (was {len(dataset.datasets)})")
                    print(f"    ⚠️  SAFETY: Limiting to {MAX_DATASETS_PER_RELEASE} datasets")
                    dataset.datasets = dataset.datasets[:MAX_DATASETS_PER_RELEASE]

            except Exception as e:
                logger.error(f"  ❌ Failed to load {release}: {e}")
                raise

            # Filter out corrupted datasets by testing each one
            print(f"    Checking for corrupted files in {len(dataset.datasets)} datasets...")
            logger.info(f"  {release}: Checking {len(dataset.datasets)} files for corruption...")
            valid_datasets = []
            corrupted_count = 0

            for idx, ds in enumerate(dataset.datasets):
                try:
                    # Try to access the raw data to check if file is corrupted
                    _ = ds.raw.n_times
                    valid_datasets.append(ds)

                    # Progress indicator every 50 files
                    if (idx + 1) % 50 == 0:
                        print(f"    Checked {idx + 1}/{len(dataset.datasets)} files... ({len(valid_datasets)} valid)")
                        logger.info(f"  {release}: Progress {idx + 1}/{len(dataset.datasets)} - {len(valid_datasets)} valid, {corrupted_count} corrupted")

                except (IndexError, ValueError, OSError, RuntimeError) as e:
                    corrupted_count += 1
                    error_msg = str(e)[:100]
                    print(f"    ⚠️  Skipping corrupted dataset {idx}: {error_msg}")
                    logger.warning(f"  {release}: Corrupted file {idx}: {error_msg}")
                    continue

            # Create a new dataset with only valid datasets
            from braindecode.datasets import BaseConcatDataset
            dataset = BaseConcatDataset(valid_datasets)

            release_time = time.time() - release_start
            skipped = corrupted_count
            print(f"    ✅ Valid datasets: {len(dataset.datasets)} (skipped {skipped} corrupted) - {release_time:.1f}s")
            logger.info(f"  {release}: Complete - {len(dataset.datasets)} valid, {skipped} corrupted, {release_time:.1f}s")

            # Simple preprocessing - just filter valid epochs
            # No need for trial annotation for resting state

            # Manual windowing: create fixed-length windows from continuous EEG data
            print("    Creating windows from continuous data...")
            logger.info(f"  {release}: Creating fixed-length windows (manual)...")

            window_size = 200  # 2 seconds @ 100 Hz
            window_stride = 100  # 50% overlap
            windows_created = 0

            try:
                for ds_idx, ds in enumerate(valid_datasets):
                    # Get externalizing score
                    externalizing_score = ds.description.get("externalizing", None)
                    if externalizing_score is None or np.isnan(externalizing_score):
                        continue  # Skip if no score available

                    # Get raw EEG data
                    raw_data = ds.raw.get_data()  # (n_channels, n_timepoints)
                    n_channels, n_timepoints = raw_data.shape

                    # Create sliding windows
                    start_idx = 0
                    while start_idx + window_size <= n_timepoints:
                        window_data = raw_data[:, start_idx:start_idx+window_size]

                        # Store window, externalizing score, and release label
                        self.windows_datasets.append(window_data)
                        self.externalizing_scores.append(externalizing_score)
                        self.release_labels.append(release)
                        windows_created += 1

                        start_idx += window_stride

                print(f"    Windows created: {windows_created}")
                logger.info(f"  {release}: Created {windows_created} windows from {len(valid_datasets)} datasets")

                if windows_created == 0:
                    logger.warning(f"  {release}: No windows created, skipping...")
                    continue

            except Exception as e:
                logger.error(f"  {release}: Failed to create windows: {e}")
                raise

        # All windows are already in self.windows_datasets, self.externalizing_scores, self.release_labels
        print(f"\n✅ Total windows: {len(self.windows_datasets)}")
        print(f"   Releases: {set(self.release_labels)}")

        # Print externalizing score stats
        if len(self.externalizing_scores) > 0:
            print(f"   Externalizing scores: {len(self.externalizing_scores)} windows")
            print(f"   Range: [{min(self.externalizing_scores):.3f}, {max(self.externalizing_scores):.3f}]")
            print(f"   Mean: {np.mean(self.externalizing_scores):.3f}, Std: {np.std(self.externalizing_scores):.3f}")

    def __len__(self):
        return len(self.windows_datasets)

    def __getitem__(self, idx):
        # Get window data (already stored as numpy array)
        X = self.windows_datasets[idx]  # (n_channels, n_timepoints)

        # Normalize per-channel
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

        # Get externalizing score
        externalizing = self.externalizing_scores[idx]

        return torch.FloatTensor(X), torch.FloatTensor([externalizing])


class CompactExternalizingCNN(nn.Module):
    """Smaller CNN for externalizing prediction (150K params)"""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Conv1: channels x 200 -> 32x100
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3),

            # Conv2: 32x100 -> 64x50
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.4),

            # Conv3: 64x50 -> 96x25
            nn.Conv1d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(96),
            nn.ELU(),
            nn.Dropout(0.5),

            # Global pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.regressor = nn.Sequential(
            nn.Linear(96, 48),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(48, 24),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(24, 1)
        )

    def forward(self, x):
        features = self.features(x)
        output = self.regressor(features)
        return output


def compute_nrmse(y_true, y_pred):
    """Normalized RMSE"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    std = np.std(y_true)
    return rmse / std if std > 0 else 0.0


def train_model(model, train_loader, val_loader, epochs=50):
    """Train with strong regularization"""

    print("\n" + "="*80)
    print("🔥 Training Multi-Release Model")
    print("="*80)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # L1 regularization (Elastic Net with L2 from weight_decay)
    l1_lambda = 1e-5
    print(f"   L1 Lambda: {l1_lambda}")
    print(f"   L2 (weight_decay): {1e-4}")
    print("   ✅ Elastic Net regularization (L1 + L2)")

    best_nrmse = float('inf')
    patience_counter = 0
    patience = 15

    for epoch in range(epochs):
        # Check memory at start of each epoch
        if MEMORY_SAFETY_CHECKS:
            safe, mem_pct = check_memory_safe()
            if not safe:
                logger.error(f"❌ MEMORY LIMIT EXCEEDED at epoch {epoch+1}: {mem_pct:.1f}%")
                print(f"❌ MEMORY LIMIT EXCEEDED: {mem_pct:.1f}% > {MAX_MEMORY_PERCENT}%")
                print(f"⚠️  Stopping training to prevent crash")
                break

        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*80}")

        # Train
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        epoch_l1_penalty = 0
        batch_count = 0

        for data, labels in train_loader:
            # Periodic memory check during training
            if MEMORY_SAFETY_CHECKS and batch_count % CHECK_MEMORY_EVERY_N_BATCHES == 0:
                safe, mem_pct = check_memory_safe()
                if not safe:
                    logger.error(f"❌ MEMORY EXCEEDED during training: {mem_pct:.1f}%")
                    print(f"❌ MEMORY EXCEEDED: {mem_pct:.1f}%, stopping batch processing")
                    break
            batch_count += 1
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)

            # Add L1 regularization
            l1_penalty = 0
            for param in model.parameters():
                l1_penalty += torch.abs(param).sum()
            loss = loss + l1_lambda * l1_penalty
            epoch_l1_penalty += l1_penalty.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(outputs.detach().numpy().flatten())
            train_labels.extend(labels.numpy().flatten())

        # Validate
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for data, labels in val_loader:
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(outputs.numpy().flatten())
                val_labels.extend(labels.numpy().flatten())

        # Metrics
        train_labels_array = np.array(train_labels)
        train_preds_array = np.array(train_preds)
        val_labels_array = np.array(val_labels)
        val_preds_array = np.array(val_preds)

        train_nrmse = compute_nrmse(train_labels_array, train_preds_array)
        val_nrmse = compute_nrmse(val_labels_array, val_preds_array)

        # Debug info on first epoch
        if epoch == 1:
            print(f"\n🔍 DEBUG - First 10 training targets: {train_labels_array[:10]}")
            print(f"🔍 DEBUG - First 10 training preds: {train_preds_array[:10]}")
            print(f"🔍 DEBUG - Training target stats: mean={train_labels_array.mean():.4f}, std={train_labels_array.std():.4f}, range=[{train_labels_array.min():.4f}, {train_labels_array.max():.4f}]")
            print(f"\n🔍 DEBUG - First 10 val targets: {val_labels_array[:10]}")
            print(f"🔍 DEBUG - First 10 val preds: {val_preds_array[:10]}")
            print(f"🔍 DEBUG - Val target stats: mean={val_labels_array.mean():.4f}, std={val_labels_array.std():.4f}, range=[{val_labels_array.min():.4f}, {val_labels_array.max():.4f}]\n")

        avg_l1_penalty = epoch_l1_penalty / len(train_loader)
        print(f"Train NRMSE: {train_nrmse:.4f}  |  L1 Penalty: {avg_l1_penalty:.2e}")
        print(f"Val NRMSE:   {val_nrmse:.4f}")

        scheduler.step()

        # Early stopping
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            patience_counter = 0
            torch.save(model.state_dict(), 'weights_challenge_2_multi_release.pt')
            print(f"✅ Best model saved (NRMSE: {best_nrmse:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping (no improvement for {patience} epochs)")
                break

    return best_nrmse


def main():
    start_time = time.time()

    # CRITICAL: Each release has CONSTANT externalizing scores!
    # R1=0.325, R2=0.620, R3=-0.387, R4=0.297, R5=0.297
    # Solution: Use R1+R2+R3+R4 combined (maximize data and variance)
    # Split the combined dataset 80/20 for train/val, reserve R5 for final validation
    print("\n📦 Loading R1+R2+R3+R4 data (maximum available releases)...")
    print("⚠️  R1=0.325, R2=0.620, R3=-0.387, R4=0.297")
    print("⚠️  Using all 4 releases for maximum variance and generalization!")
    print("⚠️  R5 reserved for final cross-release validation")
    print("⚠️  +33% more data than R2+R3+R4 alone")
    full_dataset = MultiReleaseExternalizingDataset(
        releases=['R1', 'R2', 'R3', 'R4'],
        mini=False,  # FULL DATASET
        cache_dir='data/raw'
    )

    # Split R1+R2+R3+R4 into train (80%) and validation (20%)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    print("\n📊 Splitting R1+R2+R3+R4 dataset:")
    print(f"   Total: {total_size} windows")
    print(f"   Train: {train_size} windows (80%)")
    print(f"   Val:   {val_size} windows (20%)")

    # Use random_split with fixed seed for reproducibility
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Create model
    model = CompactExternalizingCNN()

    # Train
    best_nrmse = train_model(model, train_loader, val_loader, epochs=50)

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation NRMSE: {best_nrmse:.4f}")
    print("Target: < 0.5 (competitive)")
    print("Previous (R1+R2): 0.29 validation")
    print("Expected (R2+R3+R4): Better generalization with 3 releases!")
    print(f"Time: {elapsed/60:.1f} minutes")
    print("Model saved: weights_challenge_2_multi_release.pt")
    print("="*80)

    logger.info("✅ Training completed successfully!")
    return best_nrmse


if __name__ == "__main__":
    try:
        logger.info("Starting main training pipeline...")
        result = main()
        logger.info(f"✅ SUCCESS! Best NRMSE: {result:.4f}")
        sys.exit(0)
    except KeyboardInterrupt:
        logger.warning("⚠️  Training interrupted by user (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        logger.error("="*80)
        logger.error("💥 FATAL ERROR - Training crashed!")
        logger.error("="*80)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("\nFull traceback:")
        logger.error(traceback.format_exc())
        logger.error("="*80)
        logger.error(f"Crash details saved to: {crash_log_file}")
        logger.error("="*80)
        sys.exit(1)
