#!/usr/bin/env python3
"""
Challenge 1 Enhanced Training with:
- Temporal Attention Mechanism
- Advanced Augmentation (Mixup + Temporal Masking)
- Multi-Scale Temporal Features
- GPU Training (ALWAYS)
"""

import sys
import json
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import mne
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from src.models.enhanced_components import EnhancedEEGNeX, mixup_data

# Configure MNE to use CPU only (prevent GPU memory issues during data loading)
mne.set_config('MNE_USE_CUDA', 'false', set_env=True)
mne.set_log_level('WARNING')
os.environ['MNE_USE_CUDA'] = 'false'


def _print_banner() -> None:
    print("=" * 80)
    print("🚀 CHALLENGE 1: ENHANCED TRAINING V2")
    print("=" * 80)
    print("NEW: Temporal Attention + Mixup + Multi-Scale Features")
    print("RULE: GPU TRAINING ONLY")
    print("=" * 80)

# ============================================================================
# GPU Check - MANDATORY
# ============================================================================

def setup_device():
    """Setup device - GPU REQUIRED"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n❌ NO GPU AVAILABLE!")
        print("   This script requires GPU for training.")
        print("   Please ensure CUDA is available or use ROCm SDK.")
        sys.exit(1)
    return device

# ============================================================================
# SAM Optimizer
# ============================================================================

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


# ============================================================================
# Advanced Augmentation
# ============================================================================


def temporal_masking(x, mask_ratio=0.15):
    """Mask random temporal segments"""
    batch, channels, time = x.shape
    mask_length = int(time * mask_ratio)

    for i in range(batch):
        if torch.rand(1).item() < 0.5:  # 50% chance
            start = torch.randint(0, time - mask_length, (1,)).item()
            x[i, :, start:start+mask_length] = 0

    return x


def magnitude_warping(x, sigma=0.2):
    """Warp magnitude by smooth curve"""
    batch, channels, time = x.shape

    for i in range(batch):
        if torch.rand(1).item() < 0.3:  # 30% chance
            # Generate smooth warping curve
            knots = torch.randn(5) * sigma + 1.0
            warp_curve = F.interpolate(
                knots.unsqueeze(0).unsqueeze(0),
                size=time,
                mode='linear',
                align_corners=True
            ).squeeze()
            x[i] = x[i] * warp_curve.unsqueeze(0).to(x.device)

    return x


# ============================================================================
# Data Loading
# ============================================================================

class ResponseTimeDataset(Dataset):
    """Load EEG windows with response times from BIDS events"""

    def __init__(self, data_dirs, max_subjects=None, augment=False):
        self.segments = []
        self.response_times = []
        self.subject_ids = []
        self.augment = augment

        print(f"\n📁 Loading data (augment={augment})...")

        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            participants_file = data_dir / "participants.tsv"

            if not participants_file.exists():
                continue

            df = pd.read_csv(participants_file, sep='\t')

            if max_subjects:
                df = df.head(max_subjects)

            print(f"   📊 {data_dir.name}: {len(df)} subjects")

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"   {data_dir.name}", leave=False):
                subject_id = row['participant_id']
                subject_dir = data_dir / subject_id / "eeg"

                if not subject_dir.exists():
                    continue

                eeg_files = list(subject_dir.glob("*contrastChangeDetection*.bdf"))
                if not eeg_files:
                    continue

                for eeg_file in eeg_files:
                    try:
                        raw = mne.io.read_raw_bdf(eeg_file, preload=True, verbose=False)

                        if raw.info['sfreq'] != 100:
                            raw.resample(100, verbose=False)

                        data = raw.get_data()

                        if data.shape[0] != 129:
                            continue

                        data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)

                        events_file = eeg_file.with_name(eeg_file.name.replace('_eeg.bdf', '_events.tsv'))
                        if not events_file.exists():
                            continue

                        events_df = pd.read_csv(events_file, sep='\t')

                        trial_start_events = events_df[events_df['value'].str.contains('contrastTrial_start', case=False, na=False)]
                        button_press_events = events_df[events_df['value'].str.contains('buttonPress', case=False, na=False)]

                        if len(trial_start_events) == 0 or len(button_press_events) == 0:
                            continue

                        for _, trial_event in trial_start_events.iterrows():
                            trial_time = trial_event['onset']

                            later_presses = button_press_events[button_press_events['onset'] > trial_time]
                            if len(later_presses) == 0:
                                continue

                            press_event = later_presses.iloc[0]
                            response_time = press_event['onset'] - trial_time

                            if response_time < 0.1 or response_time > 5.0:
                                continue

                            start_sample = int(trial_time * 100)
                            end_sample = start_sample + 200

                            if end_sample > data.shape[1]:
                                continue

                            segment = data[:, start_sample:end_sample]

                            self.segments.append(segment)
                            self.response_times.append(response_time)
                            self.subject_ids.append(subject_id)

                    except Exception:
                        continue

        self.segments = np.array(self.segments, dtype=np.float32)
        self.response_times = np.array(self.response_times, dtype=np.float32)
        self.subject_ids = np.array(self.subject_ids)

        print(f"\n   ✅ Loaded {len(self)} windows with response times")
        if len(self) > 0:
            print(f"   RT range: {self.response_times.min():.3f} - {self.response_times.max():.3f} seconds")
            print(f"   Unique subjects: {len(np.unique(self.subject_ids))}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        X = torch.FloatTensor(self.segments[idx])
        y = torch.FloatTensor([self.response_times[idx]])

        if self.augment:
            if torch.rand(1).item() < 0.5:
                scale = 0.8 + 0.4 * torch.rand(1).item()
                X = X * scale

            if torch.rand(1).item() < 0.3:
                n_channels = X.shape[0]
                n_drop = max(1, int(0.05 * n_channels))
                drop_channels = torch.randperm(n_channels)[:n_drop]
                X[drop_channels, :] = 0.0

            if torch.rand(1).item() < 0.2:
                noise = torch.randn_like(X) * 0.05
                X = X + noise

        return X, y


# ============================================================================
# Training Functions
# ============================================================================

def compute_nrmse(predictions, targets):
    """Compute Normalized RMSE"""
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    nrmse = rmse / (targets.max() - targets.min() + 1e-8)
    return nrmse


def train_epoch_sam_enhanced(model, loader, optimizer, criterion, device, use_mixup=True, mixup_alpha=0.2):
    """Train one epoch with SAM optimizer and advanced augmentation"""
    model.train()
    total_loss = 0

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # Apply advanced augmentation (DISABLED - ROCm HIP kernel issues)
        # if np.random.rand() < 0.5:
        #     batch_X = temporal_masking(batch_X, mask_ratio=0.15)

        # if np.random.rand() < 0.3:
        #     batch_X = magnitude_warping(batch_X, sigma=0.2)

        use_batch_mixup = use_mixup and batch_X.size(0) > 1
        if use_batch_mixup:
            mixed_X, y_a, y_b, lam = mixup_data(batch_X, batch_y, alpha=mixup_alpha)
            predictions = model(mixed_X)
            loss = lam * criterion(predictions, y_a) + (1 - lam) * criterion(predictions, y_b)
        else:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

        loss.backward()
        optimizer.first_step(zero_grad=True)

        if use_batch_mixup:
            predictions = model(mixed_X)
            second_loss = lam * criterion(predictions, y_a) + (1 - lam) * criterion(predictions, y_b)
        else:
            predictions = model(batch_X)
            second_loss = criterion(predictions, batch_y)

        second_loss.backward()
        optimizer.second_step(zero_grad=True)

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    """Validate model"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_X)
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()

    nrmse = compute_nrmse(all_preds, all_targets)
    return nrmse


# ============================================================================
# Main Training
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Enhanced Challenge 1 Training')
    parser.add_argument('--data_dirs', nargs='+', default=['data/ds005506-bdf', 'data/ds005507-bdf'])
    parser.add_argument('--max_subjects', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--rho', type=float, default=0.05)
    parser.add_argument('--mixup_alpha', type=float, default=0.0)
    parser.add_argument('--exp_name', type=str, default='enhanced_v1')
    parser.add_argument('--early_stopping', type=int, default=15)
    parser.add_argument('--quick_dry_run', action='store_true',
                        help='Run a fast ROCm sanity check with tiny data and one epoch')

    args = parser.parse_args()

    if args.quick_dry_run:
        args.exp_name = f"{args.exp_name}_dry_run"
        if args.max_subjects is None or args.max_subjects > 2:
            args.max_subjects = 2
        args.epochs = min(args.epochs, 1)
        args.batch_size = min(args.batch_size, 8)
        args.early_stopping = 1

    _print_banner()

    # Setup GPU (MANDATORY)
    device = setup_device()

    print("\n⚙️  Configuration:")
    print(f"   data_dirs: {args.data_dirs}")
    print(f"   max_subjects: {args.max_subjects}")
    print(f"   epochs: {args.epochs}")
    print(f"   batch_size: {args.batch_size}")
    print(f"   lr: {args.lr}")
    print(f"   rho: {args.rho}")
    mixup_status = "ENABLED" if args.mixup_alpha > 0 else "DISABLED"
    print(f"   mixup_alpha: {args.mixup_alpha} ({mixup_status})")
    print(f"   device: {device}")
    print(f"   exp_name: {args.exp_name}")
    print(f"   early_stopping: {args.early_stopping}")
    if args.quick_dry_run:
        print("   mode: QUICK DRY RUN (ROCm sanity check)")

    # Create experiment directory
    exp_dir = Path('experiments') / args.exp_name / datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = exp_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)

    print(f"\n📁 Experiment: {exp_dir}")
    print("=" * 80)

    # Load data
    dataset = ResponseTimeDataset(args.data_dirs, max_subjects=args.max_subjects, augment=True)

    if len(dataset) == 0:
        print("❌ No data loaded!")
        sys.exit(1)

    # Subject-level CV split
    unique_subjects = np.unique(dataset.subject_ids)
    n_train = int(0.8 * len(unique_subjects))
    train_subjects = unique_subjects[:n_train]
    val_subjects = unique_subjects[n_train:]

    train_indices = [i for i, subj in enumerate(dataset.subject_ids) if subj in train_subjects]
    val_indices = [i for i, subj in enumerate(dataset.subject_ids) if subj in val_subjects]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # Create dataloaders (num_workers=0 to avoid MNE/ROCm multiprocessing issues)
    # pin_memory=False to avoid ROCm HIP memory issues
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    print(f"\n{'='*80}")
    print("🧬 Subject-Level Cross-Validation")
    print("=" * 80)
    print(f"   Total subjects: {len(unique_subjects)}")
    print(f"   Train subjects: {len(train_subjects)}")
    print(f"   Val subjects: {len(val_subjects)}")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")

    # Create model
    print(f"\n{'='*80}")
    print("🧠 Model: EnhancedEEGNeX")
    print("=" * 80)

    model = EnhancedEEGNeX(n_channels=129, n_times=200, n_outputs=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")

    # Setup optimizer and criterion
    criterion = nn.MSELoss()
    optimizer = SAM(model.parameters(), AdamW, lr=args.lr, rho=args.rho, weight_decay=1e-4)
    # DISABLED: LR scheduler causes HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION with ROCm
    # scheduler = ReduceLROnPlateau(optimizer.base_optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    print(f"\n{'='*80}")
    print("🎯 SAM Optimizer + Advanced Augmentation")
    print("=" * 80)
    print("   Base optimizer: AdamW")
    print(f"   Learning rate: {args.lr}")
    print(f"   SAM rho: {args.rho}")
    print("   Weight decay: 1e-4")
    print(f"   Mixup alpha: {args.mixup_alpha} ({'ENABLED' if args.mixup_alpha > 0 else 'DISABLED'})")
    print("   Temporal masking: 15%")
    print("   Magnitude warping: 30% chance")

    # Training loop
    print(f"\n{'='*80}")
    print("🚀 Training Started")
    print("=" * 80)

    best_val_nrmse = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_nrmse': [], 'lr': []}

    for epoch in range(args.epochs):
        # Train
        use_mixup = args.mixup_alpha > 0
        train_loss = train_epoch_sam_enhanced(
            model, train_loader, optimizer, criterion, device,
            use_mixup=use_mixup, mixup_alpha=args.mixup_alpha
        )

        # Validate
        val_nrmse = validate(model, val_loader, device)

        # DISABLED: LR scheduler causes ROCm errors
        # scheduler.step(val_nrmse)
        current_lr = optimizer.base_optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['val_nrmse'].append(val_nrmse)
        history['lr'].append(current_lr)

        # Print progress
        is_best = val_nrmse < best_val_nrmse
        best_marker = "✨ BEST!" if is_best else ""

        print(f"Epoch {epoch+1:3d}/{args.epochs} | Train Loss: {train_loss:.4f} | "
              f"Val NRMSE: {val_nrmse:.4f} | LR: {current_lr:.2e} | {best_marker}")

        # Save checkpoint if best
        if is_best:
            best_val_nrmse = val_nrmse
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.base_optimizer.state_dict(),
                'val_nrmse': val_nrmse,
                'history': history
            }
            torch.save(checkpoint, checkpoints_dir / 'best_model.pt')
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.early_stopping:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            break

    # Save final results
    print(f"\n{'='*80}")
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"   Best Val NRMSE: {best_val_nrmse:.4f}")
    print(f"   Experiment: {exp_dir}")
    print(f"   Best model: {checkpoints_dir / 'best_model.pt'}")

    # Save history
    with open(exp_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()
