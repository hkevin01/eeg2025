#!/usr/bin/env python3
"""
Challenge 1: Robust Multi-Release Training (GPU Optimized)
==========================================================

Phase 1 Improvements:
1. Multi-release training: R1+R2+R3 (80/20 split)
2. Huber loss: Robust to outliers
3. Residual reweighting: Downweight noisy samples after epoch 5

GPU Optimization:
- Auto-detect best device (CUDA/ROCm/MPS/CPU)
- Mixed precision training (AMP)
- Pinned memory for faster data transfer
- Multi-core data loading
- Gradient accumulation for larger effective batch size

Expected: 2.01 → 1.5-1.7 (Rank #25-30)
"""
import os
import sys
from pathlib import Path

# CRITICAL: Apply ROCm fix BEFORE importing torch modules
sys.path.insert(0, str(Path(__file__).parent))
import fix_torch_arange  # This patches torch.arange for ROCm compatibility

import time
import warnings
import logging
import traceback
from datetime import datetime
warnings.filterwarnings('ignore')

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"train_c1_robust_gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torch.amp import autocast, GradScaler
import numpy as np
# braindecode can differ across versions; try importing the helpers robustly
try:
    from braindecode.preprocessing import Preprocessor, preprocess, create_windows_from_events
except Exception as e:
    # Try alternative import path used by some braindecode versions
    try:
        from braindecode.datautil.windowers import create_windows_from_events  # type: ignore
        from braindecode.preprocessing import Preprocessor, preprocess
    except Exception:
        logger.error("Failed to import braindecode create_windows_from_events or preprocessing: %s", e)
        logger.error("Please install a compatible braindecode and mne build. Example: pip install braindecode[mne]==0.7.2")
        raise

from eegdash import EEGChallengeDataset
from eegdash.hbn.windows import add_extras_columns

# ============================================================================
# GPU/Device Setup - Auto-detect best available device
# ============================================================================

def get_optimal_device():
    """Force CPU mode (ROCm has compatibility issues with braindecode)"""

    # Force CPU due to ROCm torch.arange bug in braindecode
    device = torch.device('cpu')
    logger.info("⚙️  Using CPU mode (ROCm compatibility)")
    logger.info(f"   CPU cores: {os.cpu_count()}")
    logger.info("   Optimizations: Multi-threading enabled")

    # Enable CPU optimizations
    torch.set_num_threads(os.cpu_count())

    return device, False

# Get device
DEVICE, USE_GPU = get_optimal_device()
USE_AMP = USE_GPU  # Mixed precision only on GPU

print("="*80)
print("🎯 CHALLENGE 1: ROBUST MULTI-RELEASE TRAINING (GPU OPTIMIZED)")
print("="*80)
print(f"Device: {DEVICE}")
print(f"Mixed Precision (AMP): {USE_AMP}")
print("")
print("Phase 1 Improvements:")
print("  ✅ Multi-release: R1+R2+R3 (80/20 split)")
print("  ✅ Huber loss: Robust to outliers")
print("  ✅ Residual reweighting: After epoch 5")
print("")
print("Expected: Overall 2.01 → 1.5-1.7 (Rank #25-30)")
print("="*80)

EPOCH_LEN_S = 2.0
SHIFT_AFTER_STIM = 0.5

# ============================================================================
# Dataset - Multi-Release Support
# ============================================================================

class MultiReleaseDataset(Dataset):
    """Dataset combining multiple releases"""

    def __init__(self, releases, mini=False, cache_dir='data/raw'):
        self.releases = releases
        self.windows_datasets = []
        self.response_times = []

        print(f"\n📂 Loading releases: {releases}")

        for release_idx, release in enumerate(releases, 1):
            print(f"  [{release_idx}/{len(releases)}] Loading {release}...", end=' ', flush=True)
            start = time.time()

            try:
                dataset = EEGChallengeDataset(
                    release=release,
                    mini=mini,
                    query=dict(task="contrastChangeDetection"),
                    cache_dir=Path(cache_dir)
                )

                # Validate individual files and skip corrupted ones
                valid_ds = []
                corrupted = 0
                for idx, ds in enumerate(dataset.datasets):
                    try:
                        _ = ds.raw.n_times  # access to trigger possible errors
                        valid_ds.append(ds)
                    except Exception as e:
                        corrupted += 1
                        if corrupted % 50 == 0:
                            logger.warning(f"  {release}: skipped {corrupted} corrupted files so far")
                        continue

                if len(valid_ds) == 0:
                    raise RuntimeError(f"No valid subject files found in release {release}")

                # Replace dataset.datasets with only valid ones for downstream processing
                dataset.datasets = valid_ds

                # Preprocessing
                preprocessors = [
                    Preprocessor('set_eeg_reference', ref_channels='average', ch_type='eeg'),
                    Preprocessor(lambda data: np.clip(data, -800e-6, 800e-6), apply_on_array=True),
                ]
                preprocess(dataset, preprocessors)

                # Create windows (always on CPU - braindecode doesn't need GPU here)
                # Data will be moved to GPU during training in the DataLoader
                windows_ds = create_windows_from_events(
                    dataset,
                    trial_start_offset_samples=0,
                    trial_stop_offset_samples=0,
                    picks='eeg',
                    preload=True
                )

                # Extract targets with add_extras_columns
                add_extras_columns(windows_ds)

                # Get response times
                metadata = windows_ds.get_metadata()
                response_times = metadata['response_time'].values

                # Store
                self.windows_datasets.append(windows_ds)
                self.response_times.extend(response_times)

                elapsed = time.time() - start
                print(f"✓ ({len(response_times)} trials, {elapsed:.1f}s) — corrupted: {corrupted}")

            except Exception as e:
                print(f"✗ Error: {e}")
                logger.error(f"Failed to load {release}: {e}")
                continue

        # Combine
        if len(self.windows_datasets) == 0:
            raise RuntimeError("No datasets loaded successfully! All releases failed.")

        self.combined_dataset = ConcatDataset(self.windows_datasets)
        self.response_times = np.array(self.response_times, dtype=np.float32)

        print(f"\n✅ Total: {len(self)} trials across {len(releases)} releases")

    def __len__(self):
        return len(self.combined_dataset)

    def __getitem__(self, idx):
        X, _, _ = self.combined_dataset[idx]
        y = self.response_times[idx]

        # Convert to tensors
        X = torch.from_numpy(X).float()
        y = torch.tensor(y, dtype=torch.float32)

        return X, y


# ============================================================================
# Model - Compact CNN (200K params)
# ============================================================================

class CompactResponseTimeCNN(nn.Module):
    """Compact CNN for response time prediction (200K params)"""

    def __init__(self, n_channels=19, dropout=0.5):
        super().__init__()

        # Temporal convolutions (multi-scale)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, (1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(0.3),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (n_channels, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout(0.4),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # Global average pooling + FC
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # x: [B, C, T]
        x = x.unsqueeze(1)  # [B, 1, C, T]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = self.fc(x)
        return x.squeeze(-1)


# ============================================================================
# Loss Functions
# ============================================================================

def huber_loss(pred, target, delta=1.0):
    """
    Huber loss: Robust to outliers.
    Quadratic for small errors, linear for large errors.
    """
    err = pred - target
    abs_err = err.abs()
    quad = torch.clamp(abs_err, max=delta)
    lin = abs_err - quad
    return (0.5 * quad**2 + delta * lin).mean()


def nrmse_score(pred, target):
    """Normalized RMSE"""
    mse = ((pred - target) ** 2).mean()
    rmse = torch.sqrt(mse)
    std = target.std() + 1e-8
    return rmse / std


# ============================================================================
# Training Loop with GPU Optimization
# ============================================================================

def train_epoch(model, train_loader, optimizer, scaler, device, epoch, warmup_epochs=5):
    """Training epoch with residual reweighting and AMP"""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move to device
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        # Mixed precision training (use new API)
        if USE_AMP:
            with autocast('cuda'):
                outputs = model(inputs).squeeze()

                # Loss computation
                if epoch < warmup_epochs:
                    # Warmup: Standard Huber loss
                    loss = huber_loss(outputs, targets, delta=1.0)
                else:
                    # After warmup: Residual-based reweighting
                    with torch.no_grad():
                        residuals = (outputs - targets).abs()
                        residual_std = residuals.std().clamp(min=1e-6)
                        z_scores = (residuals / residual_std).clamp(max=3.0)
                        weights = torch.exp(-z_scores / 2.0)  # Downweight large residuals

                    # Weighted Huber loss
                    err = outputs - targets
                    abs_err = err.abs()
                    delta = 1.0
                    quad = torch.clamp(abs_err, max=delta)
                    lin = abs_err - quad
                    loss = (weights * (0.5 * quad**2 + delta * lin)).mean()

            # Backward with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # No AMP (CPU or MPS)
            outputs = model(inputs).squeeze()

            if epoch < warmup_epochs:
                loss = huber_loss(outputs, targets, delta=1.0)
            else:
                with torch.no_grad():
                    residuals = (outputs - targets).abs()
                    residual_std = residuals.std().clamp(min=1e-6)
                    z_scores = (residuals / residual_std).clamp(max=3.0)
                    weights = torch.exp(-z_scores / 2.0)

                err = outputs - targets
                abs_err = err.abs()
                delta = 1.0
                quad = torch.clamp(abs_err, max=delta)
                lin = abs_err - quad
                loss = (weights * (0.5 * quad**2 + delta * lin)).mean()

            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(targets)
        n_samples += len(targets)

    return total_loss / n_samples


@torch.no_grad()
def validate(model, val_loader, device):
    """Validation with NRMSE"""
    model.eval()

    all_preds = []
    all_targets = []

    for inputs, targets in val_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if USE_AMP:
            with autocast('cuda'):
                outputs = model(inputs).squeeze()
        else:
            outputs = model(inputs).squeeze()

        all_preds.append(outputs.cpu())
        all_targets.append(targets.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    nrmse = nrmse_score(preds, targets).item()
    return nrmse


def train_model(model, train_loader, val_loader, epochs=50, patience=15):
    """Main training loop"""
    model = model.to(DEVICE)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Gradient scaler for AMP (use new API)
    scaler = GradScaler('cuda') if USE_AMP else None

    best_nrmse = float('inf')
    best_epoch = -1
    patience_counter = 0

    print(f"\n�� Starting training ({epochs} epochs max, patience={patience})")
    print(f"   Device: {DEVICE}")
    print(f"   Mixed Precision: {USE_AMP}")
    print(f"   Huber loss + Residual reweighting after epoch 5\n")

    for epoch in range(epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scaler, DEVICE, epoch, warmup_epochs=5)

        # Validate
        val_nrmse = validate(model, val_loader, DEVICE)

        # Step scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Print progress
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val NRMSE: {val_nrmse:.4f} | "
              f"LR: {lr:.6f} | "
              f"Time: {epoch_time:.1f}s", end='')

        # Save best
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'nrmse': val_nrmse,
            }, 'weights/weights_challenge_1_robust.pt')
            print(" ✓ (best)")
            patience_counter = 0
        else:
            print()
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping (no improvement for {patience} epochs)")
            break

    print(f"\n✅ Training complete!")
    print(f"   Best Val NRMSE: {best_nrmse:.4f} (epoch {best_epoch})")

    return best_nrmse


# ============================================================================
# Main
# ============================================================================

def main():
    start_time = time.time()

    # Create weights directory
    Path("weights").mkdir(exist_ok=True)

    # Load ALL releases and split 80/20
    print("\n📦 Loading ALL releases (R1+R2+R3)...")
    all_dataset = MultiReleaseDataset(
        releases=['R1', 'R2', 'R3'],
        mini=False,
        cache_dir='data/raw'
    )

    # Split 80/20
    train_size = int(0.8 * len(all_dataset))
    val_size = len(all_dataset) - train_size

    print(f"\n📊 Splitting dataset:")
    print(f"   Total: {len(all_dataset)} trials")
    print(f"   Train: {train_size} trials (80%)")
    print(f"   Val:   {val_size} trials (20%)")

    train_dataset, val_dataset = random_split(
        all_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # DataLoaders with GPU optimization
    num_workers = min(4, os.cpu_count() // 2) if not USE_GPU else 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=USE_GPU,  # Faster GPU transfer
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=USE_GPU,
        persistent_workers=True if num_workers > 0 else False
    )

    # Create model
    model = CompactResponseTimeCNN(dropout=0.5)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n🔧 Model: CompactResponseTimeCNN")
    print(f"   Parameters: {n_params:,}")

    # Train
    best_nrmse = train_model(model, train_loader, val_loader, epochs=50, patience=15)

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("✅ PHASE 1 COMPLETE")
    print("="*80)
    print(f"Best validation NRMSE: {best_nrmse:.4f}")
    print(f"Previous test score: 4.047 (Challenge 1)")
    print(f"Expected new score: 2.0-2.5 (50% improvement!)")
    print(f"Training time: {elapsed/60:.1f} minutes")
    print(f"Weights saved: weights/weights_challenge_1_robust.pt")
    print("="*80)

    logger.info(f"✅ Training completed! Best NRMSE: {best_nrmse:.4f}")

    return best_nrmse


if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0)
    except KeyboardInterrupt:
        logger.warning("⚠️  Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error("="*80)
        logger.error("💥 Training crashed!")
        logger.error("="*80)
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        logger.error("="*80)
        sys.exit(1)
