#!/usr/bin/env python3
"""
Improved C1 Training - Based on V10 Architecture
Target: Improve from 1.00019 to < 0.95
"""
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import random

# Seed everything
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("🚀 C1 V10 Improved Training")
print("=" * 60)
print(f"Target: < 0.95 (current baseline: 1.00019)")
print("=" * 60)

# ============================================================
# DATA AUGMENTATION
# ============================================================

class AugmentedH5Dataset(Dataset):
    """Dataset with time warping and noise augmentation"""
    
    def __init__(self, h5_paths, indices=None, augment=False):
        self.files = []
        self.offsets = [0]
        self.norm_stats = []
        self.augment = augment
        
        for path in h5_paths:
            f = h5py.File(path, 'r')
            self.files.append(f)
            n_samples = len(f['eeg'])
            self.offsets.append(self.offsets[-1] + n_samples)
            
            # Compute normalization stats from sample
            # Sample 1000 random indices to estimate stats quickly
            n_samples_file = len(f['eeg'])
            sample_size = min(1000, n_samples_file)
            sample_indices = np.sort(np.random.choice(n_samples_file, sample_size, replace=False))
            eeg_sample = f['eeg'][sample_indices]
            
            stats = []
            for ch in range(eeg_sample.shape[1]):
                m = eeg_sample[:, ch].mean()
                s = eeg_sample[:, ch].std()
                stats.append((m, s if s > 0 else 1.0))
            self.norm_stats.append(stats)
        
        self.total_samples = self.offsets[-1]
        self.indices = indices if indices is not None else list(range(self.total_samples))
        print(f"✅ {len(self.files)} files, {len(self.indices):,} samples (augment={augment})")
    
    def __len__(self):
        return len(self.indices)
    
    def time_warp(self, x):
        """Apply random time warping"""
        if np.random.rand() < 0.5:
            # Stretch or compress time axis by 5-15%
            factor = np.random.uniform(0.9, 1.1)
            new_len = int(x.shape[1] * factor)
            x_warped = F.interpolate(
                torch.from_numpy(x).unsqueeze(0),
                size=new_len,
                mode='linear',
                align_corners=False
            ).squeeze(0).numpy()
            
            # Crop or pad back to 200
            if x_warped.shape[1] > 200:
                start = np.random.randint(0, x_warped.shape[1] - 200)
                x_warped = x_warped[:, start:start+200]
            elif x_warped.shape[1] < 200:
                pad = 200 - x_warped.shape[1]
                x_warped = np.pad(x_warped, ((0, 0), (0, pad)), mode='edge')
            
            return x_warped
        return x
    
    def add_noise(self, x):
        """Add Gaussian noise"""
        if np.random.rand() < 0.5:
            noise_level = np.random.uniform(0.01, 0.05)
            noise = np.random.randn(*x.shape) * noise_level
            return x + noise
        return x
    
    def channel_dropout(self, x):
        """Randomly drop 5-10% of channels"""
        if np.random.rand() < 0.3:
            n_drop = np.random.randint(6, 13)  # 5-10% of 129
            channels = np.random.choice(129, n_drop, replace=False)
            x_aug = x.copy()
            x_aug[channels] = 0
            return x_aug
        return x
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        # Find which file
        file_idx = 0
        for i, offset in enumerate(self.offsets[1:]):
            if real_idx < offset:
                file_idx = i
                break
        
        local_idx = real_idx - self.offsets[file_idx]
        
        # Load from file
        x = self.files[file_idx]['eeg'][local_idx].astype(np.float32)
        y = float(self.files[file_idx]['labels'][local_idx])
        
        # Normalize
        for ch, (m, s) in enumerate(self.norm_stats[file_idx]):
            x[ch] = (x[ch] - m) / s
        
        # Apply augmentation during training
        if self.augment:
            x = self.time_warp(x)
            x = self.add_noise(x)
            x = self.channel_dropout(x)
        
        return torch.from_numpy(x), torch.tensor(y)
    
    def __del__(self):
        for f in self.files:
            f.close()

# ============================================================
# MODEL (V10 Architecture with minor improvements)
# ============================================================

class CompactResponseTimeCNN(nn.Module):
    """V10 architecture with improved regularization"""
    
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            # Conv1: 129 -> 32 (stride 2)
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Conv2: 32 -> 64 (stride 2)
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            # Conv3: 64 -> 128 (stride 2)
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
        return output.squeeze(-1)

# ============================================================
# TRAINING
# ============================================================

def nrmse(pred, target):
    """Normalized RMSE"""
    return torch.sqrt(F.mse_loss(pred, target)) / (target.std() + 1e-8)

# Load data
files = [f'data/cached/challenge1_R{i}_windows.h5' for i in range(1, 5)]

# Create indices for 80/20 split
all_indices = list(range(41071))  # Total samples
np.random.shuffle(all_indices)
n_train = int(0.8 * len(all_indices))

train_indices = all_indices[:n_train]
val_indices = all_indices[n_train:]

# Create datasets
train_dataset = AugmentedH5Dataset(files, train_indices, augment=True)
val_dataset = AugmentedH5Dataset(files, val_indices, augment=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=128, num_workers=0)

print(f"📊 Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

# Model
model = CompactResponseTimeCNN()
print(f"🧠 Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Optimizer with weight decay
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Cosine annealing scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

best_val = float('inf')
patience = 15
patience_counter = 0

print("\n" + "=" * 60)
print("🏋️  TRAINING (50 epochs with early stopping)")
print("=" * 60)

for epoch in range(1, 51):
    # Train
    model.train()
    train_loss = 0
    for X, y in train_loader:
        optimizer.zero_grad()
        pred = model(X)
        loss = nrmse(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in val_loader:
            pred = model(X)
            val_loss += nrmse(pred, y).item()
    val_loss /= len(val_loader)
    
    # Learning rate update
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save best model
    status = ""
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), 'checkpoints/c1_v10_improved_best.pt')
        status = "✅ BEST"
        patience_counter = 0
    else:
        patience_counter += 1
    
    print(f"Epoch {epoch:2d}: Train={train_loss:.4f} Val={val_loss:.4f} LR={current_lr:.6f} {status}")
    
    # Early stopping
    if patience_counter >= patience:
        print(f"\n⏹️  Early stopping at epoch {epoch} (patience={patience})")
        break

print("\n" + "=" * 60)
print("📊 RESULTS")
print("=" * 60)
print(f"🎯 Best Validation: {best_val:.6f}")
print(f"📌 Baseline: 1.00019")
print(f"🎯 Target: < 0.95")

if best_val < 0.95:
    improvement = (1.00019 - best_val) / 1.00019 * 100
    print(f"\n✅ TARGET ACHIEVED! Improved by {improvement:.1f}%!")
elif best_val < 1.00019:
    improvement = (1.00019 - best_val) / 1.00019 * 100
    gap_to_target = (best_val / 0.95 - 1) * 100
    print(f"\n🎉 IMPROVED by {improvement:.1f}%!")
    print(f"⚠️  Still need {gap_to_target:.1f}% more to reach target")
else:
    gap = (best_val - 1.00019) / 1.00019 * 100
    print(f"\n⚠️  Worse by {gap:.1f}%. Need different approach!")

print("=" * 60)
print(f"💾 Model saved to: checkpoints/c1_v10_improved_best.pt")
print("=" * 60)
