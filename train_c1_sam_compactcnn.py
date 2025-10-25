"""
SAM Training for Challenge 1 - CompactCNN Architecture
Baseline: Oct 16 model (score 1.0015)
Target: < 0.9 (10% improvement)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import argparse
import sys
import os

# Add src to path
sys.path.append('src')

# Additional imports
import mne
import pandas as pd


class CompactResponseTimeCNN(nn.Module):
    """Compact CNN for response time prediction (Challenge 1)

    Oct 16 baseline - Score: 1.0015
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Conv1: 129 channels x 200 timepoints -> 32x100
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Conv2: 32x100 -> 64x50
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),

            # Conv3: 64x50 -> 128x25
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            # Global pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        features = self.features(x)
        output = self.regressor(features)
        return output


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer"""

    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def step(self, closure=None):
        raise NotImplementedError("SAM doesn't work like regular optimizers, use first_step and second_step")


class ResponseTimeDataset(Dataset):
    """Dataset for response time prediction with augmentation"""

    def __init__(self, windows, response_times, subjects, augment=False):
        self.windows = windows
        self.response_times = response_times
        self.subjects = subjects
        self.augment = augment

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx].copy()
        y = self.response_times[idx]
        subject = self.subjects[idx]

        if self.augment:
            # Random scaling
            if np.random.random() < 0.5:
                scale = np.random.uniform(0.8, 1.2)
                x = x * scale

            # Channel dropout
            if np.random.random() < 0.3:
                n_drop = np.random.randint(1, 13)  # Drop 1-12 channels (10%)
                drop_idx = np.random.choice(x.shape[0], n_drop, replace=False)
                x[drop_idx] = 0

            # Gaussian noise
            if np.random.random() < 0.5:
                noise = np.random.randn(*x.shape) * 0.01
                x = x + noise

        return torch.FloatTensor(x), torch.FloatTensor([y]), subject


def load_challenge1_data(max_subjects=None):
    """Load Challenge 1 data from HBN dataset"""
    print("📁 Loading Challenge 1 data...")

    # Load from both datasets
    datasets = ['ds005506-bdf', 'ds005507-bdf']
    all_windows = []
    all_rts = []
    all_subjects = []

    for ds_name in datasets:
        print(f"   �� Loading {ds_name}...")
        try:
            data_dict = load_data_for_task(
                task='contrastChangeDetection',
                dataset_name=ds_name,
                bids_root='/data',
                description_fields=['subject', 'session', 'run', 'rt_from_stimulus']
            )

            if data_dict and 'windows' in data_dict:
                windows = data_dict['windows']
                descriptions = data_dict['description']

                # Extract response times and subjects
                for i, desc in enumerate(descriptions):
                    if desc is not None and 'rt_from_stimulus' in desc:
                        rt = desc['rt_from_stimulus']
                        if rt is not None and not np.isnan(rt) and rt > 0:
                            all_windows.append(windows[i])
                            all_rts.append(rt)
                            all_subjects.append(desc['subject'])

                print(f"      ✓ Loaded {len(all_windows)} windows so far")
        except Exception as e:
            print(f"      ⚠️  Error loading {ds_name}: {e}")

    if len(all_windows) == 0:
        raise ValueError("No data loaded!")

    # Convert to arrays
    windows = np.array(all_windows)
    rts = np.array(all_rts)
    subjects = np.array(all_subjects)

    # Limit subjects if requested
    if max_subjects is not None:
        unique_subjects = np.unique(subjects)
        selected_subjects = unique_subjects[:max_subjects]
        mask = np.isin(subjects, selected_subjects)
        windows = windows[mask]
        rts = rts[mask]
        subjects = subjects[mask]

    print(f"\n✅ Loaded {len(windows)} windows with response times")
    print(f"   RT range: {rts.min():.3f} - {rts.max():.3f} seconds")
    print(f"   Unique subjects: {len(np.unique(subjects))}")
    print(f"   Window shape: {windows.shape}")

    return windows, rts, subjects


def calculate_nrmse(predictions, targets):
    """Calculate Normalized RMSE"""
    mse = F.mse_loss(predictions, targets)
    rmse = torch.sqrt(mse)
    nrmse = rmse / (targets.max() - targets.min())
    return nrmse.item()


def train_epoch_sam(model, loader, optimizer, device, epoch):
    """Train one epoch with SAM optimizer"""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    for batch_idx, (x, y, _) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        # First forward-backward pass (ascent step)
        predictions = model(x).squeeze(-1)
        loss = F.mse_loss(predictions, y.squeeze(-1))
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # Second forward-backward pass (descent step)
        predictions = model(x).squeeze(-1)
        loss = F.mse_loss(predictions, y.squeeze(-1))
        loss.backward()
        optimizer.second_step(zero_grad=True)

        total_loss += loss.item()
        all_preds.append(predictions.detach())
        all_targets.append(y.squeeze(-1).detach())

        if (batch_idx + 1) % 10 == 0:
            print(f"   Batch {batch_idx+1}/{len(loader)}, Loss: {loss.item():.6f}")

    avg_loss = total_loss / len(loader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    nrmse = calculate_nrmse(all_preds, all_targets)

    return avg_loss, nrmse


def validate(model, loader, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            predictions = model(x).squeeze(-1)
            loss = F.mse_loss(predictions, y.squeeze(-1))

            total_loss += loss.item()
            all_preds.append(predictions)
            all_targets.append(y.squeeze(-1))

    avg_loss = total_loss / len(loader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    nrmse = calculate_nrmse(all_preds, all_targets)

    return avg_loss, nrmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--rho', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--exp-name', type=str, default='sam_c1_compactcnn')
    parser.add_argument('--max-subjects', type=int, default=None)
    parser.add_argument('--early-stopping', type=int, default=15)
    args = parser.parse_args()

    print("\n" + "="*80)
    print("🧠 Challenge 1 SAM Training - CompactCNN")
    print("="*80)
    print(f"Configuration:")
    print(f"   epochs: {args.epochs}")
    print(f"   batch_size: {args.batch_size}")
    print(f"   lr: {args.lr}")
    print(f"   rho: {args.rho}")
    print(f"   device: {args.device}")
    print(f"   exp_name: {args.exp_name}")
    print(f"   early_stopping: {args.early_stopping}")

    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/{args.exp_name}/{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    print(f"\n📁 Experiment: {exp_dir}")
    print()
    print("="*80)
    print()

    # Load data
    windows, rts, subjects = load_challenge1_data(max_subjects=args.max_subjects)

    # Subject-level cross-validation split
    print("\n🔀 Creating subject-level train/val split...")
    unique_subjects = np.unique(subjects)
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)

    # Use first fold for train/val split
    train_idx, val_idx = next(gkf.split(windows, rts, subjects))

    train_subjects = np.unique(subjects[train_idx])
    val_subjects = np.unique(subjects[val_idx])

    print(f"   Train subjects: {len(train_subjects)}")
    print(f"   Val subjects: {len(val_subjects)}")
    print(f"   Train windows: {len(train_idx)}")
    print(f"   Val windows: {len(val_idx)}")

    # Create datasets
    train_dataset = ResponseTimeDataset(
        windows[train_idx], rts[train_idx], subjects[train_idx], augment=True
    )
    val_dataset = ResponseTimeDataset(
        windows[val_idx], rts[val_idx], subjects[val_idx], augment=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    print(f"\n📊 Created dataloaders:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    # Create model
    print(f"\n🏗️  Creating CompactCNN model...")
    model = CompactResponseTimeCNN()

    # Load baseline weights
    baseline_path = "weights/BACKUP_C1_OCT16_1.0015.pt"
    if Path(baseline_path).exists():
        print(f"   Loading baseline from: {baseline_path}")
        state_dict = torch.load(baseline_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
        print("   ✅ Baseline loaded")
    else:
        print(f"   ⚠️  Baseline not found, training from scratch")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")

    # Create SAM optimizer
    print(f"\n⚙️  Creating SAM optimizer (rho={args.rho})...")
    base_optimizer = torch.optim.Adam
    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=args.rho)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer.base_optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
        'train_nrmse': [],
        'val_loss': [],
        'val_nrmse': [],
        'lr': []
    }

    best_val_nrmse = float('inf')
    patience_counter = 0

    print(f"\n🚀 Starting training...")
    print("="*80)
    print()

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("-" * 80)

        # Train
        train_loss, train_nrmse = train_epoch_sam(model, train_loader, optimizer, device, epoch)

        # Validate
        val_loss, val_nrmse = validate(model, val_loader, device)

        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.base_optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['train_nrmse'].append(train_nrmse)
        history['val_loss'].append(val_loss)
        history['val_nrmse'].append(val_nrmse)
        history['lr'].append(current_lr)

        # Print metrics
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {train_loss:.6f}, Train NRMSE: {train_nrmse:.4f}")
        print(f"   Val Loss: {val_loss:.6f}, Val NRMSE: {val_nrmse:.4f}")
        print(f"   LR: {current_lr:.6f}")

        # Save best model
        if val_nrmse < best_val_nrmse:
            best_val_nrmse = val_nrmse
            patience_counter = 0

            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': val_nrmse,
                'val_loss': val_loss,
                'train_nrmse': train_nrmse,
                'train_loss': train_loss,
            }
            torch.save(checkpoint, ckpt_dir / 'best_model.pt')
            torch.save(model.state_dict(), ckpt_dir / 'best_weights.pt')

            print(f"   ✅ Best Val NRMSE: {best_val_nrmse:.4f} (saved)")
        else:
            patience_counter += 1
            print(f"   Best Val NRMSE: {best_val_nrmse:.4f} (patience: {patience_counter}/{args.early_stopping})")

        print()

        # Early stopping
        if patience_counter >= args.early_stopping:
            print(f"⚠️  Early stopping triggered after {epoch+1} epochs")
            break

        # Save history every 5 epochs
        if (epoch + 1) % 5 == 0:
            history['epoch'] = epoch + 1
            with open(exp_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)

    # Save final history
    history['final_epoch'] = epoch + 1
    history['best_val_nrmse'] = best_val_nrmse
    with open(exp_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("="*80)
    print("✅ Training complete!")
    print(f"📁 Results saved to: {exp_dir}")
    print(f"🏆 Best Val NRMSE: {best_val_nrmse:.4f}")
    print(f"📊 Target was: < 0.9")
    if best_val_nrmse < 0.9:
        print("   🎉 TARGET ACHIEVED!")
    elif best_val_nrmse < 1.0:
        print("   ✅ Good performance (< 1.0)")
    else:
        print("   ⚠️  Needs improvement")
    print("="*80)


if __name__ == "__main__":
    main()
