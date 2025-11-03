#!/usr/bin/env python3
"""
C1 Training - V10 with Memory Caching
Loads all data into RAM for fast training
"""
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import time

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("🚀 C1 V10 Cached Training")
print("=" * 60)

# ============================================================
# Load all data into memory
# ============================================================

print("\n📥 Loading data into memory...")
start_load = time.time()

files = [f'data/cached/challenge1_R{i}_windows.h5' for i in range(1, 5)]
all_data = []
all_labels = []

for i, path in enumerate(files, 1):
    print(f"  Loading file {i}/4: {path}")
    with h5py.File(path, 'r') as f:
        data = f['eeg'][:].astype(np.float32)
        labels = f['labels'][:].astype(np.float32)
        all_data.append(data)
        all_labels.append(labels)
        print(f"    Loaded {len(data):,} samples")

all_data = np.concatenate(all_data, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

print(f"\n✅ Loaded {len(all_data):,} total samples")
print(f"   Data shape: {all_data.shape}")
print(f"   Time: {time.time() - start_load:.1f}s")

# Normalize per channel
print("\n🔄 Normalizing...")
for ch in range(129):
    m = all_data[:, ch].mean()
    s = all_data[:, ch].std()
    if s > 0:
        all_data[:, ch] = (all_data[:, ch] - m) / s

# Split train/val
print("\n📊 Splitting train/val...")
indices = np.arange(len(all_data))
np.random.shuffle(indices)
n_train = int(0.8 * len(indices))

train_idx = indices[:n_train]
val_idx = indices[n_train:]

X_train = torch.from_numpy(all_data[train_idx])
y_train = torch.from_numpy(all_labels[train_idx])
X_val = torch.from_numpy(all_data[val_idx])
y_val = torch.from_numpy(all_labels[val_idx])

print(f"   Train: {len(X_train):,} | Val: {len(X_val):,}")

# Create dataloaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

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
# TRAINING
# ============================================================

model = CompactResponseTimeCNN()
print(f"\n🧠 Parameters: {sum(p.numel() for p in model.parameters()):,}")

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
    eta = (elapsed / epoch) * (50 - epoch) / 60
    print(f"Ep {epoch:2d}: Train={train_loss:.4f} Val={val_loss:.4f} LR={lr:.6f} {status} [ETA:{eta:.0f}m]")
    
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
print(f"⏱️  Total time: {(time.time()-start_time)/60:.1f} minutes")
print("=" * 60)
