#!/usr/bin/env python3
"""
Lightweight CNN for fast C1 training (CPU-optimized)
"""
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("="*60)
print("🚀 Fast C1 Training (CPU-Optimized)")
print("="*60)

# Lightweight dataset
class FastDataset(Dataset):
    def __init__(self, h5_paths):
        data_list, label_list = [], []
        for path in h5_paths:
            with h5py.File(path, 'r') as f:
                data_list.append(f['eeg'][:])
                label_list.append(f['labels'][:])
        
        self.data = np.concatenate(data_list).astype(np.float32)
        self.labels = np.concatenate(label_list).astype(np.float32)
        
        # Normalize
        for ch in range(self.data.shape[1]):
            m, s = self.data[:, ch].mean(), self.data[:, ch].std()
            if s > 0:
                self.data[:, ch] = (self.data[:, ch] - m) / s
        
        print(f"✅ Loaded {len(self)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx])

# Smaller model
class LightCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(129, 64, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, 1)
        self.drop = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.drop(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.drop(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x).squeeze(-1)
        return self.fc(x).squeeze(-1)

def nrmse(pred, target):
    return torch.sqrt(F.mse_loss(pred, target)) / (target.std() + 1e-8)

# Load data
files = [f'data/cached/challenge1_R{i}_windows.h5' for i in range(1, 5)]
dataset = FastDataset(files)

train_size = int(0.8 * len(dataset))
train_ds, val_ds = torch.utils.data.random_split(
    dataset, [train_size, len(dataset) - train_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=128, num_workers=0)

print(f"📊 Train: {len(train_ds):,} | Val: {len(val_ds):,}")

# Model
model = LightCNN()
print(f"🧠 Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
best_val = float('inf')

print("\n" + "="*60)
print("🏋️  TRAINING")
print("="*60)

for epoch in range(1, 41):
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
        torch.save(model.state_dict(), 'checkpoints/c1_fast_best.pt')
        status = "✅"
    
    print(f"Ep {epoch:2d}: Train={train_loss:.4f} Val={val_loss:.4f} {status}")

print("\n" + "="*60)
print(f"🎯 Best Val: {best_val:.6f}")
print(f"🎯 Baseline: 1.00019")
print(f"🎯 Target: < 0.95")

if best_val < 0.95:
    print("\n✅ TARGET ACHIEVED!")
elif best_val < 1.00019:
    print(f"\n🎉 IMPROVED! ({best_val:.6f} < 1.00019)")
else:
    print(f"\n⚠️  Not improved. Gap: {best_val - 1.00019:.6f}")
print("="*60)
