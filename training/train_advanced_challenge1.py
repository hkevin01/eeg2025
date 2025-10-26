#!/usr/bin/env python3
"""
Advanced Challenge 1 Training Script
Includes: Conformer, SAM optimizer, Subject-level CV, Advanced augmentation
Crash-resistant with checkpointing and recovery
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
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupKFold
from braindecode.models import EEGNeX
from tqdm import tqdm

print("=" * 70)
print("🚀 Advanced Challenge 1 Training - Post-Crash Recovery")
print("=" * 70)

# ============================================================================
# SAM Optimizer Implementation
# ============================================================================

class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization
    Finds flatter minima for better generalization
    Addresses the validation/test gap issue
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
                p.add_(e_w)  # Climb to the local maximum

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Descent step"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # Get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # Do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """Not used - SAM requires explicit first_step() and second_step()"""
        raise NotImplementedError("SAM requires explicit first_step() and second_step()")

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

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# ============================================================================
# Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling prediction difficulty"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: predictions
        targets: ground truth
        """
        mse = F.mse_loss(inputs, targets, reduction='none')
        pt = torch.exp(-mse)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * mse

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# Advanced Augmentation
# ============================================================================

class AdvancedAugmentation:
    """Advanced EEG augmentation: Mixup, CutMix, and traditional methods"""

    @staticmethod
    def mixup(x1, y1, x2, y2, alpha=0.4):
        """Mixup: Linear interpolation between samples"""
        lam = np.random.beta(alpha, alpha)
        x_mixed = lam * x1 + (1 - lam) * x2
        y_mixed = lam * y1 + (1 - lam) * y2
        return x_mixed, y_mixed

    @staticmethod
    def cutmix(x1, y1, x2, y2, alpha=1.0):
        """CutMix: Cut and paste temporal segments"""
        lam = np.random.beta(alpha, alpha)

        n_times = x1.size(2)
        cut_len = int(n_times * (1 - lam))
        cut_start = np.random.randint(0, max(1, n_times - cut_len))

        x_mixed = x1.clone()
        x_mixed[:, :, cut_start:cut_start+cut_len] = x2[:, :, cut_start:cut_start+cut_len]

        y_mixed = lam * y1 + (1 - lam) * y2
        return x_mixed, y_mixed

    @staticmethod
    def temporal_masking(x, mask_ratio=0.15):
        """Randomly mask temporal segments"""
        n_times = x.size(2)
        mask_len = int(n_times * mask_ratio)
        mask_start = np.random.randint(0, max(1, n_times - mask_len))

        x_masked = x.clone()
        x_masked[:, :, mask_start:mask_start+mask_len] = 0
        return x_masked

    @staticmethod
    def channel_dropout(x, drop_prob=0.1):
        """Randomly drop EEG channels"""
        if np.random.rand() < 0.5:  # Apply 50% of the time
            n_chans = x.size(1)
            n_drop = int(n_chans * drop_prob)
            drop_chans = np.random.choice(n_chans, n_drop, replace=False)
            x_dropped = x.clone()
            x_dropped[:, drop_chans, :] = 0
            return x_dropped
        return x


# ============================================================================
# Real HBN Dataset Loading
# ============================================================================

class ResponseTimeDataset(Dataset):
    """Load EEG windows with response times from BIDS events"""

    def __init__(self, data_dirs, max_subjects=None, augment=False):
        self.segments = []
        self.response_times = []
        self.subject_ids = []  # For subject-level CV
        self.augment = augment

        print(f"📁 Loading HBN Challenge 1 data (augment={augment})...")

        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            participants_file = data_dir / "participants.tsv"

            if not participants_file.exists():
                print(f"   ⚠️  {data_dir}: participants.tsv not found")
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

                        # Find trials with response times
                        for _, event_row in events_df.iterrows():
                            if 'response_time' in events_df.columns:
                                rt = event_row.get('response_time', np.nan)
                            elif 'rt' in events_df.columns:
                                rt = event_row.get('rt', np.nan)
                            else:
                                continue

                            if pd.isna(rt) or rt <= 0:
                                continue

                            onset = event_row.get('onset', np.nan)
                            if pd.isna(onset):
                                continue

                            # Extract 2-second window starting 0.5s after stimulus
                            start_sample = int((onset + 0.5) * 100)
                            end_sample = start_sample + 200  # 2 seconds @ 100Hz

                            if end_sample > data.shape[1]:
                                continue

                            segment = data[:, start_sample:end_sample]

                            self.segments.append(segment)
                            self.response_times.append(rt)
                            self.subject_ids.append(subject_id)

                    except Exception as e:
                        continue

        self.segments = np.array(self.segments, dtype=np.float32)
        self.response_times = np.array(self.response_times, dtype=np.float32)
        self.subject_ids = np.array(self.subject_ids)

        print(f"\n   ✅ Loaded {len(self)} windows with response times")
        print(f"   RT range: {self.response_times.min():.3f} - {self.response_times.max():.3f} seconds")
        print(f"   Unique subjects: {len(np.unique(self.subject_ids))}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        X = torch.FloatTensor(self.segments[idx])
        y = torch.FloatTensor([self.response_times[idx]])

        if self.augment:
            # Basic augmentation (advanced mixing done at batch level)
            if torch.rand(1).item() < 0.5:
                scale = 0.8 + 0.4 * torch.rand(1).item()
                X = X * scale

            if torch.rand(1).item() < 0.2:
                noise = torch.randn_like(X) * 0.05
                X = X + noise

        return X, y


# ============================================================================
# Subject-Level Cross-Validation
# ============================================================================

def extract_subject_id(file_path):
    """Extract subject ID from file path"""
    # Assuming format: .../sub-<ID>/...
    path_str = str(file_path)
    if 'sub-' in path_str:
        start = path_str.find('sub-') + 4
        end = path_str.find('/', start)
        if end == -1:
            end = path_str.find('_', start)
        return path_str[start:end] if end != -1 else path_str[start:start+10]
    return "unknown"


def create_subject_level_splits(file_paths, targets, n_splits=5):
    """
    Create subject-level cross-validation splits
    Ensures no subject appears in both train and validation
    """
    # Extract subject IDs
    subjects = np.array([extract_subject_id(fp) for fp in file_paths])

    print(f"\n📊 Subject-Level Cross-Validation Setup:")
    print(f"  Total samples: {len(file_paths)}")
    print(f"  Unique subjects: {len(np.unique(subjects))}")
    print(f"  Splits: {n_splits}")

    # Group K-Fold
    gkf = GroupKFold(n_splits=n_splits)

    splits = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(file_paths, targets, groups=subjects)):
        train_subjects = set(subjects[train_idx])
        val_subjects = set(subjects[val_idx])

        # Verify no overlap
        overlap = train_subjects & val_subjects
        if overlap:
            print(f"  ⚠️ Warning: Fold {fold} has {len(overlap)} overlapping subjects!")
        else:
            print(f"  ✅ Fold {fold}: Train={len(train_subjects)} subjects, Val={len(val_subjects)} subjects")

        splits.append((train_idx, val_idx))

    return splits


# ============================================================================
# Training Manager with Checkpointing
# ============================================================================

class TrainingManager:
    """Manages training with crash recovery and checkpointing"""

    def __init__(self, experiment_dir, resume=True):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.resume = resume
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_nrmse': [],
            'best_nrmse': float('inf'),
            'best_epoch': 0,
            'completed_epochs': 0
        }

    def save_checkpoint(self, epoch, model, optimizer, metrics, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.base_optimizer.state_dict() if hasattr(optimizer, 'base_optimizer') else optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }

        # Save latest
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved best model (NRMSE: {metrics['val_nrmse']:.4f})")

        # Clean old checkpoints (keep last 5 + best)
        self._cleanup_checkpoints(keep_last=5)

    def _cleanup_checkpoints(self, keep_last=5):
        """Keep only recent checkpoints to save space"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > keep_last:
            for ckpt in checkpoints[:-keep_last]:
                ckpt.unlink()

    def load_checkpoint(self, model, optimizer=None):
        """Load latest checkpoint if exists"""
        if not self.resume:
            return 0

        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if not checkpoints:
            print("  ℹ️ No checkpoint found, starting from scratch")
            return 0

        latest_checkpoint = checkpoints[-1]
        print(f"  📂 Loading checkpoint: {latest_checkpoint.name}")

        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            if hasattr(optimizer, 'base_optimizer'):
                optimizer.base_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.history = checkpoint['history']
        start_epoch = checkpoint['epoch'] + 1

        print(f"  ✅ Resumed from epoch {checkpoint['epoch']}")
        print(f"  📊 Best NRMSE so far: {self.history['best_nrmse']:.4f}")

        return start_epoch

    def update_history(self, train_loss, val_loss, val_nrmse, epoch):
        """Update training history"""
        self.history['train_loss'].append(float(train_loss))
        self.history['val_loss'].append(float(val_loss))
        self.history['val_nrmse'].append(float(val_nrmse))
        self.history['completed_epochs'] = epoch

        if val_nrmse < self.history['best_nrmse']:
            self.history['best_nrmse'] = val_nrmse
            self.history['best_epoch'] = epoch
            return True  # New best
        return False

    def save_history(self):
        """Save training history to JSON"""
        history_path = self.experiment_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


# ============================================================================
# Simple Dataset (placeholder - will be replaced with real data loading)
# ============================================================================

class SimpleEEGDataset(Dataset):
    """Simple dataset for testing"""
    def __init__(self, n_samples=100, n_chans=129, n_times=200):
        self.data = torch.randn(n_samples, n_chans, n_times)
        self.targets = torch.randn(n_samples, 1) * 2 + 3  # Response times 1-5s
        self.file_paths = [f"sub-{i//10:03d}/file_{i}.edf" for i in range(n_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'eeg': self.data[idx],
            'target': self.targets[idx],
            'file_path': self.file_paths[idx]
        }


# ============================================================================
# Main Training Function
# ============================================================================

def train_with_sam(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    num_epochs=100,
    device='cuda',
    manager=None,
    use_mixup=True,
    use_cutmix=False
):
    """
    Training loop with SAM optimizer
    """
    model = model.to(device)
    best_val_nrmse = float('inf')
    patience = 15
    patience_counter = 0

    # Load checkpoint if resuming
    start_epoch = 0
    if manager is not None:
        start_epoch = manager.load_checkpoint(model, optimizer)
        best_val_nrmse = manager.history['best_nrmse']

    augmenter = AdvancedAugmentation()

    print(f"\n🏋️ Starting training from epoch {start_epoch}")
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Augmentation: Mixup={use_mixup}, CutMix={use_cutmix}")

    for epoch in range(start_epoch, num_epochs):
        # ============ Training ============
        model.train()
        train_losses = []

        for batch_idx, batch in enumerate(train_loader):
            x = batch['eeg'].to(device)
            y = batch['target'].to(device)

            # Apply augmentation
            if use_mixup and np.random.rand() < 0.5:
                indices = torch.randperm(x.size(0))
                x2, y2 = x[indices], y[indices]
                x, y = augmenter.mixup(x, y, x2, y2, alpha=0.3)
            elif use_cutmix and np.random.rand() < 0.3:
                indices = torch.randperm(x.size(0))
                x2, y2 = x[indices], y[indices]
                x, y = augmenter.cutmix(x, y, x2, y2, alpha=0.5)

            # SAM first step
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # SAM second step
            pred = model(x)
            criterion(pred, y).backward()
            optimizer.second_step(zero_grad=True)

            train_losses.append(loss.item())

        # ============ Validation ============
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch['eeg'].to(device)
                y = batch['target'].to(device)

                pred = model(x)
                loss = criterion(pred, y)

                val_losses.append(loss.item())
                all_preds.append(pred.cpu())
                all_targets.append(y.cpu())

        # Calculate metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        # NRMSE
        mse = F.mse_loss(all_preds, all_targets)
        rmse = torch.sqrt(mse)
        nrmse = rmse / (all_targets.max() - all_targets.min())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        # Update history
        is_best = False
        if manager is not None:
            is_best = manager.update_history(train_loss, val_loss, nrmse.item(), epoch)
            manager.save_checkpoint(
                epoch, model, optimizer,
                {'train_loss': train_loss, 'val_loss': val_loss, 'val_nrmse': nrmse.item()},
                is_best=is_best
            )
            manager.save_history()

        # Print progress
        status = "🌟 NEW BEST" if is_best else ""
        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val NRMSE: {nrmse:.4f} {status}")

        # Early stopping
        if nrmse < best_val_nrmse:
            best_val_nrmse = nrmse
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping triggered after {patience} epochs without improvement")
                break

    print(f"\n✅ Training complete!")
    print(f"  Best NRMSE: {best_val_nrmse:.4f}")

    return model, best_val_nrmse


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Advanced Challenge 1 Training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--exp-name', type=str, default='advanced_c1', help='Experiment name')
    parser.add_argument('--use-mixup', action='store_true', default=True, help='Use Mixup')
    parser.add_argument('--use-cutmix', action='store_true', help='Use CutMix')
    parser.add_argument('--use-focal-loss', action='store_true', help='Use Focal Loss')
    parser.add_argument('--rho', type=float, default=0.05, help='SAM rho parameter')
    parser.add_argument('--data-dirs', nargs='+', default=['data/ds005506-bdf', 'data/ds005507-bdf'],
                        help='Data directories')
    parser.add_argument('--max-subjects', type=int, default=None, help='Max subjects to load (None=all)')

    args = parser.parse_args()

    print("\n⚙️ Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Create experiment directory
    exp_dir = Path('experiments') / args.exp_name / datetime.now().strftime('%Y%m%d_%H%M%S')
    manager = TrainingManager(exp_dir, resume=args.resume)

    try:
        # Load real HBN Challenge 1 data
        print("\n📊 Loading real HBN dataset...")
        full_dataset = ResponseTimeDataset(
            data_dirs=args.data_dirs,
            max_subjects=args.max_subjects,
            augment=True  # Enable augmentation
        )

        if len(full_dataset) == 0:
            print("❌ Error: No data loaded! Check data directories:")
            for data_dir in args.data_dirs:
                print(f"  - {data_dir}")
            sys.exit(1)

        # Subject-level split
        print("\n🧬 Creating subject-level train/val split...")
        subject_ids = full_dataset.subject_ids
        unique_subjects = np.unique(subject_ids)

        # Create subject ID mapping for GroupKFold
        subject_id_map = {sid: i for i, sid in enumerate(unique_subjects)}
        subject_groups = np.array([subject_id_map[sid] for sid in subject_ids])

        # Use GroupKFold for subject-level split
        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(np.arange(len(full_dataset)), groups=subject_groups))

        # Use first fold for this run
        train_idx, val_idx = splits[0]

        print(f"  Unique subjects: {len(unique_subjects)}")
        print(f"  Train subjects: {len(np.unique(subject_groups[train_idx]))}")
        print(f"  Val subjects: {len(np.unique(subject_groups[val_idx]))}")

        # Create subject-level isolated datasets
        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")

        # Create model
        print("\n🧠 Creating model...")
        model = EEGNeX(
            n_chans=129,
            n_times=200,
            n_outputs=1,
            sfreq=100
        )
        print(f"  Model: EEGNeX")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create optimizer (SAM)
        print("\n🎯 Creating SAM optimizer...")
        base_lr = args.lr
        optimizer = SAM(
            model.parameters(),
            torch.optim.AdamW,
            lr=base_lr,
            rho=args.rho,
            weight_decay=1e-4
        )
        print(f"  Base optimizer: AdamW")
        print(f"  Learning rate: {base_lr}")
        print(f"  SAM rho: {args.rho}")

        # Create loss
        if args.use_focal_loss:
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
            print(f"  Loss: Focal Loss")
        else:
            criterion = nn.MSELoss()
            print(f"  Loss: MSE Loss")

        # Train
        model, best_nrmse = train_with_sam(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=args.epochs,
            device=args.device,
            manager=manager,
            use_mixup=args.use_mixup,
            use_cutmix=args.use_cutmix
        )

        print(f"\n🎉 Training completed successfully!")
        print(f"  Best NRMSE: {best_nrmse:.4f}")
        print(f"  Experiment: {exp_dir}")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        if manager:
            manager.save_history()
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Error during training:")
        print(traceback.format_exc())
        if manager:
            manager.save_history()
        sys.exit(1)


if __name__ == '__main__':
    main()
