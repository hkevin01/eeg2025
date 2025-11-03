#!/usr/bin/env python3
"""
Challenge 1: Multi-Seed Ensemble Training for V16
==================================================
Strategy: Train 5-10 models with different random seeds
Expected improvement: -0.001 to -0.003 (0.1-0.3%)

Features:
- Same CompactResponseTimeCNN architecture (proven)
- 30 epochs per model (vs 20 in V10)
- Better augmentation
- EMA (Exponential Moving Average) for stability
- Save best checkpoint per seed
"""

import os
import sys

# Force CPU training (GPU has issues)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HIP_VISIBLE_DEVICES'] = ''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime

# Force CPU if GPU issues
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

print("="*80)
print("🎯 Challenge 1 - Multi-Seed Ensemble Training V16")
print("="*80)


# ============================================================================
# MODEL ARCHITECTURE (Same as V10/V15)
# ============================================================================

class CompactResponseTimeCNN(nn.Module):
    """Compact CNN for response time prediction - 75K parameters"""

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


# ============================================================================
# DATASET WITH STRONG AUGMENTATION
# ============================================================================

class ResponseTimeDataset(Dataset):
    """Dataset for Challenge 1 with augmentation - uses preprocessed data"""

    def __init__(self, data_file='data/processed/challenge1_data.h5', split='train', augment=True):
        self.augment = augment
        self.split = split

        print(f"\n📂 Loading preprocessed data: {data_file}")

        try:
            with h5py.File(data_file, 'r') as f:
                if split == 'train':
                    self.segments = f['X_train'][:]
                    self.response_times = f['y_train'][:]
                else:
                    self.segments = f['X_val'][:]
                    self.response_times = f['y_val'][:]

            print(f"   ✅ Loaded {len(self)} trials ({split})")
            print(f"   Shape: {self.segments.shape}")
            print(f"   RT range: {self.response_times.min():.3f} - {self.response_times.max():.3f}s")
            print(f"   RT mean: {self.response_times.mean():.3f} ± {self.response_times.std():.3f}s")

        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            raise

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = torch.FloatTensor(self.segments[idx])
        rt = torch.FloatTensor([self.response_times[idx]])

        if self.augment:
            # 1. Amplitude scaling (80-120%)
            if torch.rand(1).item() < 0.5:
                scale = 0.8 + 0.4 * torch.rand(1).item()
                segment = segment * scale

            # 2. Time shift (±5 samples)
            if torch.rand(1).item() < 0.5:
                shift = torch.randint(-5, 6, (1,)).item()
                segment = torch.roll(segment, shifts=shift, dims=1)

            # 3. Gaussian noise (σ=0.02)
            if torch.rand(1).item() < 0.5:
                noise = torch.randn_like(segment) * 0.02
                segment = segment + noise

            # 4. Channel dropout (5% of channels)
            if torch.rand(1).item() < 0.3:
                n_dropout = int(0.05 * 129)
                dropout_channels = torch.randperm(129)[:n_dropout]
                segment[dropout_channels] = 0

        return segment, rt


# ============================================================================
# EMA (Exponential Moving Average)
# ============================================================================

class EMA:
    """Exponential Moving Average for model parameters"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_single_seed(seed, data_file, save_dir, epochs=30):
    """Train single model with specific seed"""

    print(f"\n{'='*80}")
    print(f"🌱 Training with seed {seed}")
    print(f"{'='*80}")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data (already split in preprocessed file)
    train_dataset = ResponseTimeDataset(data_file, split='train', augment=True)
    val_dataset = ResponseTimeDataset(data_file, split='val', augment=False)

    print(f"\nTrain: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Model
    model = CompactResponseTimeCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # EMA
    ema = EMA(model, decay=0.999)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')
    best_val_nrmse = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for X, y in pbar:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            # Update EMA
            ema.update()

            train_loss += loss.item()
            train_preds.extend(pred.detach().cpu().numpy())
            train_targets.extend(y.detach().cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        train_loss /= len(train_loader)
        train_preds = np.array(train_preds).flatten()
        train_targets = np.array(train_targets).flatten()
        train_nrmse = np.sqrt(np.mean((train_preds - train_targets)**2)) / (train_targets.max() - train_targets.min())

        # Validation with EMA
        ema.apply_shadow()
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = criterion(pred, y)
                val_loss += loss.item()
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        val_loss /= len(val_loader)
        val_preds = np.array(val_preds).flatten()
        val_targets = np.array(val_targets).flatten()
        val_nrmse = np.sqrt(np.mean((val_preds - val_targets)**2)) / (val_targets.max() - val_targets.min())

        ema.restore()

        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.6f}, NRMSE: {train_nrmse:.4f}")
        print(f"  Val Loss: {val_loss:.6f}, NRMSE: {val_nrmse:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # Save best model
        if val_nrmse < best_val_nrmse:
            best_val_nrmse = val_nrmse
            best_val_loss = val_loss
            patience_counter = 0

            # Save with EMA weights
            ema.apply_shadow()
            save_path = save_dir / f'c1_seed{seed}_best.pt'
            torch.save(model.state_dict(), save_path)
            ema.restore()

            print(f"  ✅ New best! Saved to {save_path.name}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

        scheduler.step()

    print(f"\n✅ Seed {seed} complete!")
    print(f"   Best Val Loss: {best_val_loss:.6f}")
    print(f"   Best Val NRMSE: {best_val_nrmse:.4f}")

    return best_val_nrmse


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=str, default='42,123,456,789,1337',
                        help='Comma-separated list of seeds')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Epochs per model')
    parser.add_argument('--data-file', type=str,
                        default='data/processed/challenge1_data.h5',
                        help='Path to preprocessed data file')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    print(f"\n📊 Configuration:")
    print(f"   Seeds: {seeds}")
    print(f"   Epochs per seed: {args.epochs}")
    print(f"   Data file: {args.data_file}")

    # Create checkpoint directory
    save_dir = Path('checkpoints/c1_v16_ensemble')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Train each seed
    results = []
    for seed in seeds:
        val_nrmse = train_single_seed(seed, args.data_file, save_dir, args.epochs)
        results.append((seed, val_nrmse))

    # Summary
    print(f"\n{'='*80}")
    print("🎉 ALL SEEDS COMPLETE!")
    print(f"{'='*80}")
    print("\nResults:")
    for seed, nrmse in results:
        print(f"   Seed {seed}: Val NRMSE = {nrmse:.4f}")

    mean_nrmse = np.mean([r[1] for r in results])
    print(f"\n   Mean Val NRMSE: {mean_nrmse:.4f}")
    print(f"\n✅ Checkpoints saved in: {save_dir}")


if __name__ == '__main__':
    main()

