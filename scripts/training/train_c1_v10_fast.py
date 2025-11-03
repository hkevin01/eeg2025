#!/usr/bin/env python3
"""
Fast C1 Training - V10 Architecture with Augmentation
Target: < 0.95 from 1.00019
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
import time

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("🚀 C1 V10 Fast Training")
print("=" * 60)

# ============================================================
# EFFICIENT DATASET
# ============================================================

class FastH5Dataset(Dataset):
    def __init__(self, h5_paths, indices=None, augment=False, norm_stats=None):
        self.files = []
        self.offsets = [0]
        self.augment = augment
        self.norm_stats = norm_stats if norm_stats else []
        
        compute_stats = (norm_stats is None)
        
        for path in h5_paths:
            f = h5py.File(path, 'r')
            self.files.append(f)
            n_samples = len(f['eeg'])
            self.offsets.append(self.offsets[-1] + n_samples)
            
            if compute_stats:
                # Compute stats from first 500 samples
                sample = f['eeg'][:min(500, n_samples)]
                stats = [(sample[:, ch].mean(), sample[:, ch].std() or 1.0) 
                         for ch in range(129)]
                self.norm_stats.append(stats)
        
        self.total_samples = self.offsets[-1]
        self.indices = indices if indices is not None else list(range(self.total_samples))
        print(f"✅ {len(self.files)} files, {len(self.indices):,} samples (augment={augment})")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        # Find file
        file_idx = next(i for i, offset in enumerate(self.offsets[1:]) if real_idx < offset)
        local_idx = real_idx - self.offsets[file_idx]
        
        # Load
        x = self.files[file_idx]['eeg'][local_idx].astype(np.float32)
        y = float(self.files[file_idx]['labels'][local_idx])
        
        # Normalize
        for ch, (m, s) in enumerate(self.norm_stats[file_idx]):
            x[ch] = (x[ch] - m) / s
        
        # Augmentation
        if self.augment:
            # Time warping
            if np.random.rand() < 0.4:
                factor = np.random.uniform(0.92, 1.08)
                new_len = int(200 * factor)
                x_t = torch.from_numpy(x).unsqueeze(0)
                x_warped = F.interpolate(x_t, size=new_len, mode='linear', align_corners=False)
                x_warped = x_warped.squeeze(0).numpy()
                
                if x_warped.shape[1] > 200:
                    start = np.random.randint(0, x_warped.shape[1] - 200)
                    x = x_warped[:, start:start+200]
                elif x_warped.shape[1] < 200:
                    x = np.pad(x_warped, ((0,0), (0, 200-x_warped.shape[1])), mode='edge')
                else:
                    x = x_warped
            
            # Noise
            if np.random.rand() < 0.4:
                x += np.random.randn(*x.shape) * np.random.uniform(0.01, 0.03)
            
            # Channel dropout
            if np.random.rand() < 0.25:
                n_drop = np.random.randint(6, 13)
                channels = np.random.choice(129, n_drop, replace=False)
                x[channels] = 0
        
        return torch.from_numpy(x), torch.tensor(y)
    
    def __del__(self):
        for f in self.files:
            f.close()

# ============================================================
# MODEL
# ============================================================

class CompactResponseTimeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
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

def nrmse(pred, target):
    return torch.sqrt(F.mse_loss(pred, target)) / (target.std() + 1e-8)

# ============================================================
# MAIN
# ============================================================

files = [f'data/cached/challenge1_R{i}_windows.h5' for i in range(1, 5)]

# Split
all_indices = list(range(41071))
np.random.shuffle(all_indices)
n_train = int(0.8 * len(all_indices))
train_indices = all_indices[:n_train]
val_indices = all_indices[n_train:]

print("\n📊 Creating datasets...")
# Create train dataset first (computes stats)
train_dataset = FastH5Dataset(files, train_indices, augment=True)
# Reuse stats for validation
val_dataset = FastH5Dataset(files, val_indices, augment=False, norm_stats=train_dataset.norm_stats)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=128, num_workers=0)

print(f"📊 Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

# Model
model = CompactResponseTimeCNN()
print(f"🧠 Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

best_val = float('inf')
patience = 15
patience_counter = 0

print("\n" + "=" * 60)
print("🏋️  TRAINING")
print("=" * 60)

start_time = time.time()

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
    
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    
    status = ""
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), 'checkpoints/c1_v10_improved_best.pt')
        status = "✅"
        patience_counter = 0
    else:
        patience_counter += 1
    
    elapsed = time.time() - start_time
    print(f"Ep {epoch:2d}: Train={train_loss:.4f} Val={val_loss:.4f} LR={lr:.6f} {status} [{elapsed/60:.1f}m]")
    
    if patience_counter >= patience:
        print(f"\n⏹️  Early stopping at epoch {epoch}")
        break

print("\n" + "=" * 60)
print("📊 RESULTS")
print("=" * 60)
print(f"🎯 Best Val: {best_val:.6f}")
print(f"📌 Baseline: 1.00019")
print(f"🎯 Target: < 0.95")

if best_val < 0.95:
    improvement = (1.00019 - best_val) / 1.00019 * 100
    print(f"\n✅ TARGET ACHIEVED! ({improvement:.1f}% improvement)")
elif best_val < 1.00019:
    improvement = (1.00019 - best_val) / 1.00019 * 100
    gap = (best_val / 0.95 - 1) * 100
    print(f"\n🎉 IMPROVED by {improvement:.1f}%!")
    print(f"⚠️  Need {gap:.1f}% more for target")
else:
    gap = (best_val - 1.00019) / 1.00019 * 100
    print(f"\n⚠️  Worse by {gap:.1f}%")

print("=" * 60)
print(f"💾 Saved: checkpoints/c1_v10_improved_best.pt")
print(f"⏱️  Time: {(time.time()-start_time)/60:.1f} minutes")
print("=" * 60)
