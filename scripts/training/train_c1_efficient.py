#!/usr/bin/env python3
"""
Memory-efficient C1 training (loads on-the-fly)
"""
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

torch.manual_seed(42)
np.random.seed(42)

print("="*60)
print("🚀 Efficient C1 Training")
print("="*60)

# On-the-fly loading dataset
class H5Dataset(Dataset):
    def __init__(self, h5_paths, indices=None):
        self.files = []
        self.offsets = [0]
        self.norm_stats = []
        
        for path in h5_paths:
            f = h5py.File(path, 'r')
            self.files.append(f)
            n_samples = len(f['eeg'])
            self.offsets.append(self.offsets[-1] + n_samples)
            
            # Compute normalization stats
            eeg_data = f['eeg'][:]
            stats = []
            for ch in range(eeg_data.shape[1]):
                m = eeg_data[:, ch].mean()
                s = eeg_data[:, ch].std()
                stats.append((m, s if s > 0 else 1.0))
            self.norm_stats.append(stats)
        
        self.total_samples = self.offsets[-1]
        self.indices = indices if indices is not None else list(range(self.total_samples))
        print(f"✅ {len(self.files)} files, {len(self.indices):,} samples")
    
    def __len__(self):
        return len(self.indices)
    
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
        
        return torch.from_numpy(x), torch.tensor(y)
    
    def __del__(self):
        for f in self.files:
            f.close()

# Compact model
class CompactCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(129, 64, 5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 128, 3, padding=1)
        self.pool3 = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 1)
        self.drop = nn.Dropout(0.4)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.drop(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x).squeeze(-1)

def nrmse(pred, target):
    return torch.sqrt(F.mse_loss(pred, target)) / (target.std() + 1e-8)

# Load data
files = [f'data/cached/challenge1_R{i}_windows.h5' for i in range(1, 5)]
full_dataset = H5Dataset(files)

# 80/20 split
n_total = len(full_dataset)
n_train = int(0.8 * n_total)
indices = list(range(n_total))
np.random.shuffle(indices)

train_indices = indices[:n_train]
val_indices = indices[n_train:]

train_dataset = H5Dataset(files, train_indices)
val_dataset = H5Dataset(files, val_indices)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, num_workers=0)

print(f"📊 Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

# Model
model = CompactCNN()
print(f"🧠 Params: {sum(p.numel() for p in model.parameters()):,}")

optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
best_val = float('inf')

print("\n" + "="*60)
print("🏋️  TRAINING (30 epochs)")
print("="*60)

for epoch in range(1, 31):
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
    
    # Val
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in val_loader:
            pred = model(X)
            val_loss += nrmse(pred, y).item()
    val_loss /= len(val_loader)
    
    status = ""
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), 'checkpoints/c1_efficient_best.pt')
        status = "✅ BEST"
    
    print(f"Epoch {epoch:2d}: Train={train_loss:.4f} Val={val_loss:.4f} {status}")

print("\n" + "="*60)
print(f"🎯 Best Validation: {best_val:.6f}")
print(f"🎯 Baseline: 1.00019")
print(f"🎯 Target: < 0.95")

if best_val < 0.95:
    print("\n✅ TARGET REACHED!")
elif best_val < 1.00019:
    improvement = (1.00019 - best_val) / 1.00019 * 100
    print(f"\n🎉 IMPROVED by {improvement:.1f}%!")
else:
    gap = (best_val - 1.00019) / 1.00019 * 100
    print(f"\n⚠️  Worse by {gap:.1f}%. Need to iterate!")
print("="*60)
