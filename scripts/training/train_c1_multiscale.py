#!/usr/bin/env python3
"""
Phase 1: Multi-Scale CNN for Challenge 1
Target: C1 < 0.95 (5% improvement from 1.00019)

Key improvements:
1. Multi-scale temporal feature extraction
2. Enhanced data augmentation (time warp, channel dropout, frequency masking)
3. Deeper architecture with residual connections
4. 10-seed ensemble for robustness
"""

import os
import sys
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import random

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configuration
CONFIG = {
    'seed': 42,
    'batch_size': 64,
    'epochs': 80,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'dropout_conv': [0.5, 0.6, 0.7],
    'dropout_fc': 0.7,
    'ema_decay': 0.999,
    'augment_prob': 0.8,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

print("="*70)
print("🎯 Phase 1: Multi-Scale CNN for Challenge 1")
print("="*70)
print(f"Current best: C1 = 1.00019")
print(f"Target: C1 < 0.95 (5% improvement)")
print(f"Method: Multi-scale temporal features + Enhanced augmentation")
print("="*70)
print()

for k, v in CONFIG.items():
    print(f"{k}: {v}")
print()


class EEGAugmentation:
    """Enhanced EEG augmentation suite"""
    
    @staticmethod
    def time_warp(x, sigma=0.2):
        """Non-linear time warping"""
        time_steps = x.shape[1]
        warp = np.cumsum(np.random.randn(time_steps) * sigma)
        warp = warp - warp.min()
        warp = warp / warp.max() * (time_steps - 1)
        warp = np.clip(warp, 0, time_steps - 1).astype(int)
        return x[:, warp]
    
    @staticmethod
    def channel_dropout(x, p=0.1):
        """Randomly drop channels"""
        n_channels = x.shape[0]
        n_drop = int(n_channels * p)
        if n_drop > 0:
            drop_idx = np.random.choice(n_channels, n_drop, replace=False)
            x[drop_idx, :] = 0
        return x
    
    @staticmethod
    def gaussian_noise(x, std=0.02):
        """Add Gaussian noise"""
        noise = np.random.randn(*x.shape).astype(np.float32) * std
        return x + noise
    
    @staticmethod
    def amplitude_scale(x, scale_range=(0.85, 1.15)):
        """Random amplitude scaling"""
        scale = np.random.uniform(*scale_range)
        return x * scale
    
    @staticmethod
    def time_shift(x, max_shift=15):
        """Random time shift"""
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(x, shift, axis=1)
    
    @staticmethod
    def temporal_cutout(x, max_len=30):
        """Random temporal masking"""
        cutout_len = np.random.randint(10, max_len)
        cutout_start = np.random.randint(0, x.shape[1] - cutout_len)
        x_aug = x.copy()
        x_aug[:, cutout_start:cutout_start+cutout_len] = 0
        return x_aug


class AdvancedH5Dataset(Dataset):
    """Dataset with comprehensive augmentation"""
    def __init__(self, h5_paths, augment=False):
        self.augment = augment
        self.augmentor = EEGAugmentation()
        self.data = []
        self.labels = []
        
        for h5_path in h5_paths:
            print(f"Loading {h5_path}...")
            with h5py.File(h5_path, 'r') as f:
                X = f['eeg'][:]
                y = f['labels'][:]
                self.data.append(X)
                self.labels.append(y)
        
        self.data = np.concatenate(self.data, axis=0).astype(np.float32)
        self.labels = np.concatenate(self.labels, axis=0).astype(np.float32)
        
        # Z-score normalization per channel
        for ch in range(self.data.shape[1]):
            mean = self.data[:, ch, :].mean()
            std = self.data[:, ch, :].std()
            if std > 0:
                self.data[:, ch, :] = (self.data[:, ch, :] - mean) / std
        
        print(f"Loaded {len(self.data)} samples (z-score normalized)")
        print(f"Data shape: {self.data.shape}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx].copy()
        y = self.labels[idx]
        
        if self.augment and np.random.rand() < CONFIG['augment_prob']:
            # Apply random combination of augmentations
            if np.random.rand() < 0.5:
                x = self.augmentor.time_warp(x, sigma=0.15)
            
            if np.random.rand() < 0.5:
                x = self.augmentor.channel_dropout(x, p=0.1)
            
            if np.random.rand() < 0.5:
                x = self.augmentor.gaussian_noise(x, std=0.02)
            
            if np.random.rand() < 0.5:
                x = self.augmentor.amplitude_scale(x)
            
            if np.random.rand() < 0.4:
                x = self.augmentor.time_shift(x, max_shift=15)
            
            if np.random.rand() < 0.3:
                x = self.augmentor.temporal_cutout(x, max_len=30)
        
        return torch.from_numpy(x).float(), torch.tensor(y).float()


class MultiScaleBlock(nn.Module):
    """Extract features at multiple temporal scales"""
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super().__init__()
        # Different kernel sizes for different scales
        self.scale_fine = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.scale_medium = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.scale_coarse = nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=7)
        
        # Batch norm and activation for each scale
        self.bn = nn.BatchNorm1d(out_channels * 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection (if dimensions match)
        self.residual = None
        if in_channels == out_channels * 3:
            self.residual = nn.Identity()
        elif in_channels != out_channels * 3:
            self.residual = nn.Conv1d(in_channels, out_channels * 3, kernel_size=1)
    
    def forward(self, x):
        # Extract multi-scale features
        fine = self.scale_fine(x)
        medium = self.scale_medium(x)
        coarse = self.scale_coarse(x)
        
        # Concatenate all scales
        out = torch.cat([fine, medium, coarse], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Add residual connection
        if self.residual is not None:
            res = self.residual(x)
            out = out + res
        
        return out


class MultiScaleCNN(nn.Module):
    """Multi-Scale CNN for Challenge 1"""
    def __init__(self, dropout_conv, dropout_fc):
        super().__init__()
        
        # Multi-scale blocks with increasing depth
        self.block1 = MultiScaleBlock(129, 32, dropout=dropout_conv[0])   # → 96 channels
        self.pool1 = nn.AvgPool1d(2)
        
        self.block2 = MultiScaleBlock(96, 48, dropout=dropout_conv[1])    # → 144 channels
        self.pool2 = nn.AvgPool1d(2)
        
        self.block3 = MultiScaleBlock(144, 64, dropout=dropout_conv[2])   # → 192 channels
        self.pool3 = nn.AvgPool1d(2)
        
        # Global context
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Classifier with residual
        self.fc1 = nn.Linear(192, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc1_relu = nn.ReLU()
        self.fc1_dropout = nn.Dropout(dropout_fc)
        
        self.fc2 = nn.Linear(128, 64)
        self.fc2_bn = nn.BatchNorm1d(64)
        self.fc2_relu = nn.ReLU()
        self.fc2_dropout = nn.Dropout(dropout_fc)
        
        self.fc_out = nn.Linear(64, 1)
    
    def forward(self, x):
        # Multi-scale feature extraction
        x = self.block1(x)
        x = self.pool1(x)
        
        x = self.block2(x)
        x = self.pool2(x)
        
        x = self.block3(x)
        x = self.pool3(x)
        
        # Global pooling
        x = self.gap(x).squeeze(-1)
        
        # Classifier
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.fc1_relu(x)
        x = self.fc1_dropout(x)
        
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = self.fc2_relu(x)
        x = self.fc2_dropout(x)
        
        x = self.fc_out(x)
        return x.squeeze(-1)


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
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
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


def nrmse_loss(pred, target):
    """Normalized RMSE loss"""
    mse = F.mse_loss(pred, target)
    rmse = torch.sqrt(mse)
    target_std = torch.std(target)
    if target_std > 0:
        return rmse / target_std
    return rmse


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(loader)


def train_model(seed=42):
    """Train a single model with given seed"""
    set_seed(seed)
    print(f"\n{'='*70}")
    print(f"Training with seed {seed}")
    print(f"{'='*70}")
    
    device = CONFIG['device']
    
    # Load data
    h5_files = [
        'data/cached/challenge1_R1_windows.h5',
        'data/cached/challenge1_R2_windows.h5',
        'data/cached/challenge1_R3_windows.h5',
        'data/cached/challenge1_R4_windows.h5'
    ]
    
    dataset = AdvancedH5Dataset(h5_files, augment=False)
    
    # Split train/val 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG['seed'])
    )
    
    # Enable augmentation for train set
    dataset.augment = True
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                             shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"\n📊 Dataset split:")
    print(f"   Train: {len(train_dataset):,} samples")
    print(f"   Val: {len(val_dataset):,} samples")
    
    # Create model
    model = MultiScaleCNN(
        dropout_conv=CONFIG['dropout_conv'],
        dropout_fc=CONFIG['dropout_fc']
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=CONFIG['lr'], 
                     weight_decay=CONFIG['weight_decay'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nrmse_loss
    
    # EMA
    ema = EMA(model, decay=CONFIG['ema_decay'])
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience = 15
    patience_counter = 0
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        ema.update()
        
        # Validate with EMA
        ema.apply_shadow()
        val_loss = validate(model, test_loader, criterion, device)
        ema.restore()
        
        scheduler.step()
        
        print(f"Epoch {epoch:3d}: Train={train_loss:.6f}, Val={val_loss:.6f}, LR={scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in ema.shadow.items()}
            patience_counter = 0
            print(f"  ✅ New best: {best_val_loss:.6f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping at epoch {epoch}")
            break
    
    # Load best model
    for name, param in model.named_parameters():
        if name in best_model_state:
            param.data = best_model_state[name].to(device)
    
    # Final validation
    final_val_loss = validate(model, test_loader, criterion, device)
    print(f"\n🎯 Final validation NRMSE: {final_val_loss:.6f}")
    
    # Save model
    save_dir = '/home/kevin/Projects/eeg2025/checkpoints/phase1_multiscale'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'model_seed{seed}.pth')
    torch.save(model.state_dict(), save_path)
    print(f"💾 Saved to {save_path}")
    
    return final_val_loss


if __name__ == '__main__':
    print(f"\n🚀 Starting Phase 1: Multi-Scale CNN Training\n")
    
    # Train 10 models with different seeds
    seeds = [42, 123, 456, 789, 1024, 2048, 3333, 4567, 8910, 9999]
    results = []
    
    for seed in seeds:
        try:
            val_loss = train_model(seed)
            results.append((seed, val_loss))
        except Exception as e:
            print(f"\n❌ Error with seed {seed}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 Training Summary")
    print(f"{'='*70}")
    for seed, val_loss in results:
        print(f"Seed {seed:5d}: {val_loss:.6f}")
    
    if results:
        avg_val_loss = np.mean([v for _, v in results])
        std_val_loss = np.std([v for _, v in results])
        print(f"\n🎯 Average: {avg_val_loss:.6f} ± {std_val_loss:.6f}")
        print(f"🏆 Best: {min(v for _, v in results):.6f}")
        print(f"😢 Worst: {max(v for _, v in results):.6f}")
        
        if avg_val_loss < 0.95:
            print(f"\n✅ SUCCESS! Phase 1 target achieved: {avg_val_loss:.6f} < 0.95")
        else:
            print(f"\n⚠️  Close but not quite: {avg_val_loss:.6f} (target < 0.95)")
    
    print(f"\n{'='*70}")
    print("✅ Phase 1 Training Complete!")
    print(f"{'='*70}")
