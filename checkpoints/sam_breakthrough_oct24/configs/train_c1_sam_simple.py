#!/usr/bin/env python3
"""
Advanced Challenge 1 Training - Hybrid Implementation
Combines: Working data loader + SAM + Subject-CV + Advanced augmentation
"""

import sys
import json
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import mne
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr
from braindecode.models import EEGNeX
from tqdm import tqdm

print("=" * 80)
print("🚀 CHALLENGE 1: ADVANCED TRAINING")
print("=" * 80)
print("Features: SAM Optimizer + Subject-CV + Advanced Augmentation")
print("=" * 80)

# ============================================================================
# SAM Optimizer Implementation
# ============================================================================

class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization
    Finds flatter minima for better generalization
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Ascent step"""
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
        """Descent step"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        """Compute gradient norm across all parameters"""
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
# Working Data Loader (from train_challenge1_working.py)
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

                # Find contrastChangeDetection EEG files
                eeg_files = list(subject_dir.glob("*contrastChangeDetection*.bdf"))
                if not eeg_files:
                    continue

                for eeg_file in eeg_files:
                    try:
                        # Load EEG
                        raw = mne.io.read_raw_bdf(eeg_file, preload=True, verbose=False)

                        # Resample to 100Hz
                        if raw.info['sfreq'] != 100:
                            raw.resample(100, verbose=False)

                        data = raw.get_data()

                        # Ensure 129 channels
                        if data.shape[0] != 129:
                            continue

                        # Z-score normalize per channel
                        data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)

                        # Load events to get response times
                        events_file = eeg_file.with_name(eeg_file.name.replace('_eeg.bdf', '_events.tsv'))
                        if not events_file.exists():
                            continue

                        events_df = pd.read_csv(events_file, sep='\t')

                        # Extract response times (trial start to button press)
                        trial_start_events = events_df[events_df['value'].str.contains('contrastTrial_start', case=False, na=False)]
                        button_press_events = events_df[events_df['value'].str.contains('buttonPress', case=False, na=False)]

                        if len(trial_start_events) == 0 or len(button_press_events) == 0:
                            continue

                        # Standardize per channel
                        for _, trial_event in trial_start_events.iterrows():
                            trial_time = trial_event['onset']

                            # Find corresponding button press
                            later_presses = button_press_events[button_press_events['onset'] > trial_time]
                            if len(later_presses) == 0:
                                continue

                            press_event = later_presses.iloc[0]
                            response_time = press_event['onset'] - trial_time

                            # Only use reasonable response times (0.1 to 5 seconds)
                            if response_time < 0.1 or response_time > 5.0:
                                continue

                            # Extract EEG segment starting from trial start
                            start_sample = int(trial_time * 100)
                            end_sample = start_sample + 200  # 2 seconds @ 100Hz

                            if end_sample > data.shape[1]:
                                continue

                            segment = data[:, start_sample:end_sample]

                            self.segments.append(segment)
                            self.response_times.append(response_time)
                            self.subject_ids.append(subject_id)

                    except:
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
            # Basic augmentation
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
# Training Manager with Checkpointing
# ============================================================================

class TrainingManager:
    """Manages training state, checkpointing, and recovery"""

    def __init__(self, exp_dir, resume=False):
        self.exp_dir = Path(exp_dir)
        self.checkpoints_dir = self.exp_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_nrmse': [],
            'lr': [],
            'epoch': 0
        }

        self.best_val_nrmse = float('inf')

        if resume:
            self.load_checkpoint()

    def save_checkpoint(self, model, optimizer, epoch, val_nrmse):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else None,
            'val_nrmse': val_nrmse,
            'history': self.history
        }

        checkpoint_path = self.checkpoints_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if val_nrmse < self.best_val_nrmse:
            self.best_val_nrmse = val_nrmse
            best_path = self.checkpoints_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)

    def load_checkpoint(self):
        """Load latest checkpoint"""
        best_path = self.checkpoints_dir / 'best_model.pt'
        if best_path.exists():
            checkpoint = torch.load(best_path)
            self.history = checkpoint['history']
            self.best_val_nrmse = checkpoint['val_nrmse']
            print(f"   Resumed from epoch {checkpoint['epoch']}, best NRMSE: {self.best_val_nrmse:.4f}")
            return checkpoint
        return None

    def save_history(self):
        """Save training history to JSON"""
        history_path = self.exp_dir / 'history.json'
        # Convert numpy types to Python types for JSON serialization
        history_json = {
            key: [float(x) if hasattr(x, 'item') else x for x in value] if isinstance(value, list) else value
            for key, value in self.history.items()
        }
        with open(history_path, 'w') as f:
            json.dump(history_json, f, indent=2)


# ============================================================================
# Training with SAM
# ============================================================================

def compute_nrmse(predictions, targets):
    """Compute Normalized RMSE"""
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    nrmse = rmse / (targets.max() - targets.min() + 1e-8)
    return nrmse


def train_epoch_sam(model, loader, optimizer, criterion, device):
    """Train one epoch with SAM optimizer"""
    model.train()
    total_loss = 0

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # First forward-backward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # Second forward-backward pass
        predictions = model(batch_X)
        criterion(predictions, batch_y).backward()
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
            predictions = model(batch_X)

            all_preds.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(batch_y.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    nrmse = compute_nrmse(all_preds, all_targets)
    return nrmse, all_preds, all_targets


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Advanced Challenge 1 Training')
    parser.add_argument('--data-dirs', nargs='+', default=['data/ds005506-bdf', 'data/ds005507-bdf'], help='Data directories')
    parser.add_argument('--max-subjects', type=int, default=None, help='Max subjects to load')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--rho', type=float, default=0.05, help='SAM rho parameter')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--exp-name', type=str, default='sam_advanced', help='Experiment name')
    parser.add_argument('--early-stopping', type=int, default=15, help='Early stopping patience')

    args = parser.parse_args()

    print(f"\n⚙️  Configuration:")
    for arg, value in vars(args).items():
        print(f"   {arg}: {value}")

    # Device setup
    device = torch.device(args.device)
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Create experiment directory
    exp_dir = Path('experiments') / args.exp_name / datetime.now().strftime('%Y%m%d_%H%M%S')
    manager = TrainingManager(exp_dir)
    print(f"\n📁 Experiment: {exp_dir}")

    try:
        # Load dataset
        print("\n" + "="*80)
        full_dataset = ResponseTimeDataset(
            data_dirs=args.data_dirs,
            max_subjects=args.max_subjects,
            augment=True
        )

        if len(full_dataset) == 0:
            print("❌ Error: No data loaded!")
            sys.exit(1)

        # Subject-level cross-validation split
        print("\n" + "="*80)
        print("🧬 Subject-Level Cross-Validation")
        print("="*80)

        subject_ids = full_dataset.subject_ids
        unique_subjects = np.unique(subject_ids)
        subject_id_map = {sid: i for i, sid in enumerate(unique_subjects)}
        subject_groups = np.array([subject_id_map[sid] for sid in subject_ids])

        # GroupKFold
        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(np.arange(len(full_dataset)), groups=subject_groups))

        # Use first fold
        train_idx, val_idx = splits[0]

        print(f"   Total subjects: {len(unique_subjects)}")
        print(f"   Train subjects: {len(np.unique(subject_groups[train_idx]))}")
        print(f"   Val subjects: {len(np.unique(subject_groups[val_idx]))}")
        print(f"   Train samples: {len(train_idx)}")
        print(f"   Val samples: {len(val_idx)}")

        # Create data loaders
        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        # Create model
        print("\n" + "="*80)
        print("🧠 Model: EEGNeX")
        print("="*80)

        model = EEGNeX(
            n_chans=129,
            n_times=200,
            n_outputs=1,
            sfreq=100
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {n_params:,}")

        # Create SAM optimizer
        print("\n" + "="*80)
        print("🎯 SAM Optimizer")
        print("="*80)

        optimizer = SAM(
            model.parameters(),
            torch.optim.AdamW,
            lr=args.lr,
            rho=args.rho,
            weight_decay=1e-4
        )

        print(f"   Base optimizer: AdamW")
        print(f"   Learning rate: {args.lr}")
        print(f"   SAM rho: {args.rho}")
        print(f"   Weight decay: 1e-4")

        criterion = nn.MSELoss()
        scheduler = ReduceLROnPlateau(optimizer.base_optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        # Training loop
        print("\n" + "="*80)
        print("🚀 Training Started")
        print("="*80)

        best_val_nrmse = float('inf')
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            # Train
            train_loss = train_epoch_sam(model, train_loader, optimizer, criterion, device)

            # Validate
            val_nrmse, val_preds, val_targets = validate(model, val_loader, device)

            # Learning rate
            current_lr = optimizer.base_optimizer.param_groups[0]['lr']
            scheduler.step(val_nrmse)

            # Save history
            manager.history['train_loss'].append(train_loss)
            manager.history['val_nrmse'].append(val_nrmse)
            manager.history['lr'].append(current_lr)
            manager.history['epoch'] = epoch

            # Save checkpoint
            manager.save_checkpoint(model, optimizer, epoch, val_nrmse)

            # Print progress
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val NRMSE: {val_nrmse:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"{'✨ BEST!' if val_nrmse < best_val_nrmse else ''}")

            # Early stopping
            if val_nrmse < best_val_nrmse:
                best_val_nrmse = val_nrmse
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= args.early_stopping:
                print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                break

        # Final results
        print("\n" + "="*80)
        print("✅ Training Complete!")
        print("="*80)
        print(f"   Best Val NRMSE: {best_val_nrmse:.4f}")
        print(f"   Experiment: {exp_dir}")
        print(f"   Best model: {manager.checkpoints_dir / 'best_model.pt'}")

        # Save history
        manager.save_history()

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        manager.save_history()
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Error during training:")
        print(traceback.format_exc())
        manager.save_history()
        sys.exit(1)


if __name__ == '__main__':
    main()
