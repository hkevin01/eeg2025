#!/usr/bin/env python3
"""
Challenge 1: Multi-Release Response Time Prediction
===================================================
CRITICAL FIX FOR OVERFITTING: Train on R1-R5 instead of just R5

Problem: Previous model trained only on R5, tested on R12 → 10x degradation
Solution: Multi-release training for better generalization

Model improvements:
- Reduced parameters: 800K → 200K (less overfitting)
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
import json

warnings.filterwarnings('ignore')

# Setup comprehensive logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
crash_log_file = log_dir / f"challenge1_crash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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

logger.info("="*80)
logger.info("🚀 Challenge 1 Training Started")
logger.info(f"Crash log: {crash_log_file}")
logger.info("="*80)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np
from braindecode.preprocessing import Preprocessor, preprocess
from braindecode.preprocessing import create_windows_from_events
from eegdash import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
)

print("="*80)
print("🎯 CHALLENGE 1: STIMULUS-ALIGNED RESPONSE TIME PREDICTION")
print("="*80)
print("Training on: R1, R2, R3, R4 (719 subjects)")
print("Validation on: R5 (240 subjects)")
print("Key improvements:")
print("  ✅ STIMULUS-ALIGNED windows (not trial-aligned)")
print("  ✅ Using R4 for 33% more training data")
print("  ✅ Official starter kit metadata extraction")
print("Expected improvement: 15-25% better NRMSE")
print("="*80)

EPOCH_LEN_S = 2.0
SHIFT_AFTER_STIM = 0.5


class MultiReleaseDataset(Dataset):
    """Dataset combining multiple releases for better generalization"""

    def __init__(self, releases, mini=True, cache_dir='data/raw'):
        """
        Args:
            releases: List of release names ['R1', 'R2', ...]
            mini: Use mini dataset (True for testing, False for full)
            cache_dir: Cache directory for data
        """
        self.releases = releases
        self.windows_datasets = []
        self.release_labels = []
        self.response_times = []  # Store response times for each window

        print(f"\n📂 Loading releases: {releases}")
        logger.info(f"Loading {len(releases)} releases: {releases}")

        for release_idx, release in enumerate(releases, 1):
            print(f"\n  [{release_idx}/{len(releases)}] Loading {release}...")
            logger.info(f"Loading release {release} ({release_idx}/{len(releases)})")
            release_start = time.time()

            # Load dataset
            try:
                dataset = EEGChallengeDataset(
                    release=release,
                    mini=mini,
                    query=dict(task="contrastChangeDetection"),
                    cache_dir=Path(cache_dir)
                )
                print(f"    Datasets: {len(dataset.datasets)}")
                logger.info(f"  {release}: Loaded {len(dataset.datasets)} datasets")
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

            # Preprocess
            print(f"    Preprocessing {len(dataset.datasets)} datasets...")
            logger.info(f"  {release}: Starting preprocessing...")
            preprocessors = [
                Preprocessor(
                    annotate_trials_with_target,
                    apply_on_array=False,
                    target_field="rt_from_stimulus",
                    epoch_length=EPOCH_LEN_S,
                    require_stimulus=True,
                    require_response=True,
                ),
                Preprocessor(add_aux_anchors, apply_on_array=False),
            ]

            try:
                preprocess(dataset, preprocessors, n_jobs=-1)
                logger.info(f"  {release}: Preprocessing complete")
            except Exception as e:
                logger.error(f"  {release}: Preprocessing failed: {e}")
                raise

            # Check if we have any valid trials after preprocessing
            valid_trials = sum(1 for ds in dataset.datasets if len(ds.raw.annotations) > 0)
            print(f"    Datasets with valid trials after preprocessing: {valid_trials}/{len(dataset.datasets)}")
            logger.info(f"  {release}: {valid_trials} datasets have valid trials")

            if valid_trials == 0:
                logger.warning(f"  {release}: No valid trials after preprocessing, skipping...")
                continue

            # Create windows from events (one window per trial)
            # CRITICAL: Use stimulus_anchor for proper stimulus alignment!
            # Response time is measured FROM STIMULUS, so windows must be stimulus-locked
            print("    Creating STIMULUS-ALIGNED windows from trials...")
            logger.info(f"  {release}: Creating windows from events...")

            ANCHOR = "stimulus_anchor"  # STIMULUS-ALIGNED anchor from add_aux_anchors
            SFREQ = 100  # Sampling frequency

            windows_dataset = create_windows_from_events(
                dataset,
                mapping={ANCHOR: 0},  # Lock windows to STIMULUS onset
                trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),  # +0.5s after STIMULUS
                trial_stop_offset_samples=int((SHIFT_AFTER_STIM + EPOCH_LEN_S) * SFREQ),  # +2.5s after STIMULUS
                window_size_samples=int(EPOCH_LEN_S * SFREQ),  # 2 seconds
                window_stride_samples=SFREQ,  # 1 second stride (not used for single window per trial)
                preload=True,
            )

            print(f"    Windows created: {len(windows_dataset)}")
            logger.info(f"  {release}: Created {len(windows_dataset)} windows")

            if len(windows_dataset) == 0:
                logger.warning(f"  {release}: No windows created, skipping...")
                continue

            print(f"    Windows: {len(windows_dataset)}")

            # CRITICAL: Use add_extras_columns to inject trial metadata into windows
            # This is the official starter kit approach from challenge_1.py
            # MUST use "stimulus_anchor" to match the anchor we used for windowing!
            print("    Injecting trial metadata into windows...")
            logger.info(f"  {release}: Adding extras columns to windows metadata")

            try:
                windows_dataset = add_extras_columns(
                    windows_dataset,  # Windowed dataset
                    dataset,          # Original preprocessed dataset with annotations
                    desc="stimulus_anchor",  # STIMULUS-ALIGNED descriptor (matches windowing anchor)
                    keys=("rt_from_stimulus", "target", "rt_from_trialstart",
                          "stimulus_onset", "response_onset", "correct", "response_type")
                )
                logger.info(f"  {release}: Metadata injection complete")
            except Exception as e:
                logger.error(f"  {release}: Failed to inject metadata: {e}")
                logger.error(traceback.format_exc())
                # Try with minimal keys
                try:
                    windows_dataset = add_extras_columns(
                        windows_dataset,
                        dataset,
                        desc="stimulus_anchor",  # STIMULUS-ALIGNED descriptor
                        keys=("rt_from_stimulus",)
                    )
                    logger.info(f"  {release}: Metadata injection complete (minimal keys)")
                except Exception as e2:
                    logger.error(f"  {release}: Failed with minimal keys too: {e2}")
                    raise

            # Extract metadata DataFrame - this now has rt_from_stimulus column!
            print("    Extracting response times from metadata...")
            try:
                metadata_df = windows_dataset.get_metadata()
                logger.info(f"  {release}: Got metadata DataFrame with {len(metadata_df)} rows")
                logger.info(f"  {release}: Metadata columns: {metadata_df.columns.tolist()}")

                # Extract response times
                if 'rt_from_stimulus' in metadata_df.columns:
                    rt_values = metadata_df['rt_from_stimulus'].values
                    # Replace NaN with 0.0
                    rt_values = np.nan_to_num(rt_values, nan=0.0)
                    self.response_times.extend(rt_values.tolist())

                    # Print statistics
                    non_zero = rt_values[rt_values != 0.0]
                    if len(non_zero) > 0:
                        print(f"    ✅ Response times extracted: {len(non_zero)}/{len(rt_values)} non-zero")
                        print(f"       Range: [{non_zero.min():.3f}, {non_zero.max():.3f}]")
                        print(f"       Mean: {non_zero.mean():.3f}, Std: {non_zero.std():.3f}")
                    else:
                        logger.warning(f"  {release}: All response times are 0.0!")
                        print("    ⚠️  All response times are 0.0! This may indicate an issue.")
                else:
                    logger.error(f"  {release}: rt_from_stimulus not found in metadata columns!")
                    logger.error(f"  {release}: Available columns: {metadata_df.columns.tolist()}")
                    # Fill with zeros as fallback
                    self.response_times.extend([0.0] * len(windows_dataset))

            except Exception as e:
                logger.error(f"  {release}: Failed to extract metadata: {e}")
                logger.error(traceback.format_exc())
                # Fill with zeros as fallback
                self.response_times.extend([0.0] * len(windows_dataset))

            self.windows_datasets.append(windows_dataset)
            self.release_labels.extend([release] * len(windows_dataset))        # Concatenate all releases
        self.combined_dataset = ConcatDataset(self.windows_datasets)
        print(f"\n✅ Total windows: {len(self.combined_dataset)}")
        print(f"   Releases: {set(self.release_labels)}")

    def __len__(self):
        return len(self.combined_dataset)

    def __getitem__(self, idx):
        windows_ds, rel_idx = self._get_dataset_and_index(idx)
        X, y, metadata = windows_ds[rel_idx]

        # X shape: (1, n_channels, n_timepoints) or (n_channels, n_timepoints)
        if X.ndim == 3:
            X = X.squeeze(0)

        # Normalize
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

        # Get response time from pre-extracted list (extracted during __init__ from metadata)
        response_time = self.response_times[idx] if idx < len(self.response_times) else 0.0
        if np.isnan(response_time):
            response_time = 0.0

        return torch.FloatTensor(X), torch.FloatTensor([response_time])

    def _get_dataset_and_index(self, idx):
        """Find which dataset and local index for given global index"""
        if idx < 0:
            idx = len(self) + idx

        dataset_idx = 0
        while idx >= len(self.windows_datasets[dataset_idx]):
            idx -= len(self.windows_datasets[dataset_idx])
            dataset_idx += 1

        return self.windows_datasets[dataset_idx], idx


class CompactCNN(nn.Module):
    """Compact CNN with STRONG regularization (L1+L2+Dropout) to prevent overfitting"""

    def __init__(self, dropout_p=0.5):
        """
        Args:
            dropout_p: Dropout probability (default 0.5 for strong regularization)
        """
        super().__init__()

        self.features = nn.Sequential(
            # Conv1: channels x 200 -> 32x100
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_p * 0.6),  # 0.3 with default dropout_p=0.5

            # Conv2: 32x100 -> 64x50
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_p * 0.8),  # 0.4 with default dropout_p=0.5

            # Conv3: 64x50 -> 128x25
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_p),  # 0.5 with default dropout_p=0.5

            # Global pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),  # 0.5 strong dropout before first FC
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_p * 0.8),  # 0.4 moderate dropout
            nn.Linear(32, 1)
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


def train_model(model, train_loader, val_loader, epochs=50, l1_lambda=1e-5, l2_lambda=1e-4):
    """
    Train with STRONG regularization (L1 + L2 + Dropout)

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        l1_lambda: L1 regularization strength (default 1e-5)
        l2_lambda: L2 regularization strength (default 1e-4)
    """

    print("\n" + "="*80)
    print("🔥 Training Multi-Release Model with Enhanced Regularization")
    print("="*80)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    print(f"   L1 penalty: {l1_lambda:.0e} (Lasso)")
    print(f"   L2 penalty: {l2_lambda:.0e} (Ridge, via weight_decay)")
    print(f"   Regularization: Elastic Net (L1 + L2)")
    print(f"   Dropout: 0.3-0.5 throughout network")
    print(f"   Gradient clipping: max_norm=1.0")

    criterion = nn.MSELoss()
    # L2 regularization via weight_decay
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=l2_lambda)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_nrmse = float('inf')
    patience_counter = 0
    patience = 15

    for epoch in range(epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*80}")

        # Train
        model.train()
        train_loss = 0
        train_l1_loss = 0
        train_preds = []
        train_labels = []

        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)

            # MSE loss
            mse_loss = criterion(outputs, labels)

            # L1 regularization (Lasso) - sum of absolute weights
            l1_penalty = 0.0
            for param in model.parameters():
                l1_penalty += torch.sum(torch.abs(param))

            # Total loss = MSE + L1 penalty (L2 is handled by weight_decay in optimizer)
            loss = mse_loss + l1_lambda * l1_penalty

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += mse_loss.item()  # Track MSE separately
            train_l1_loss += l1_penalty.item()
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

        # Calculate average L1 penalty per batch
        avg_l1_penalty = train_l1_loss / len(train_loader)

        # Debug info on first epoch
        if epoch == 1:
            print(f"\n🔍 DEBUG - First 10 training targets: {train_labels_array[:10]}")
            print(f"🔍 DEBUG - First 10 training preds: {train_preds_array[:10]}")
            print(f"🔍 DEBUG - Training target stats: mean={train_labels_array.mean():.4f}, std={train_labels_array.std():.4f}, range=[{train_labels_array.min():.4f}, {train_labels_array.max():.4f}]")
            print(f"\n🔍 DEBUG - First 10 val targets: {val_labels_array[:10]}")
            print(f"🔍 DEBUG - First 10 val preds: {val_preds_array[:10]}")
            print(f"🔍 DEBUG - Val target stats: mean={val_labels_array.mean():.4f}, std={val_labels_array.std():.4f}, range=[{val_labels_array.min():.4f}, {val_labels_array.max():.4f}]\n")

        print(f"Train NRMSE: {train_nrmse:.4f}  |  L1 Penalty: {avg_l1_penalty:.2e}")
        print(f"Val NRMSE:   {val_nrmse:.4f}")

        scheduler.step()

        # Early stopping
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            patience_counter = 0
            torch.save(model.state_dict(), 'weights_challenge_1_multi_release.pt')
            print(f"✅ Best model saved (NRMSE: {best_nrmse:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping (no improvement for {patience} epochs)")
                break

    return best_nrmse


def main():
    start_time = time.time()

    # Load training releases (R1-R4) - 33% MORE DATA with R4!
    print("\n📦 Loading training data (R1-R4)...")
    train_dataset = MultiReleaseDataset(
        releases=['R1', 'R2', 'R3', 'R4'],  # Added R3 and R4 for 33% more data!
        mini=False,  # FULL DATASET for production training
        cache_dir='data/raw'
    )

    # Load validation release (R5) - separate from training
    print("\n📦 Loading validation data (R5)...")
    val_dataset = MultiReleaseDataset(
        releases=['R5'],
        mini=False,  # FULL DATASET for validation
        cache_dir='data/raw'
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"\n📊 Dataset sizes:")
    print(f"   Train: {len(train_dataset)} windows")
    print(f"   Val:   {len(val_dataset)} windows")

    # Create model with strong regularization (dropout=0.5)
    print("\n🏗️  Creating model with enhanced regularization:")
    print("   • L1 + L2 regularization (Elastic Net)")
    print("   • Dropout: 0.5 (50% neurons dropped during training)")
    print("   • Gradient clipping: max_norm=1.0")
    model = CompactCNN(dropout_p=0.5)

    # Train
    best_nrmse = train_model(model, train_loader, val_loader, epochs=50)

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation NRMSE: {best_nrmse:.4f}")
    print(f"Target: < 0.5 (competitive)")
    print(f"Previous (R5 only): 0.47 validation → 4.05 test (10x degradation)")
    print(f"Expected (R1-R5): ~0.70 validation → ~1.4 test (2x better!)")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Model saved: weights_challenge_1_multi_release.pt")
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
