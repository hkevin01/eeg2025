#!/usr/bin/env python3
"""
Quick Phase 1: Multi-Scale CNN for Challenge 1 (Single Seed)
Target: C1 < 0.95 (5% improvement from 1.00019)
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

# Set seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Config
CONFIG = {
    'batch_size': 16,  # Very small batch for safety
    'epochs': 50,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'dropout': 0.6,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Try GPU again with smaller batch
    'pin_memory': False,  # Disable pinned memory
    'num_workers': 0,  # Single-threaded to avoid issues
}

print("="*70)
print("🎯 Quick Phase 1: Multi-Scale CNN for Challenge 1")
print("="*70)
print(f"Device: {CONFIG['device']}")

# Dataset
class SimpleH5Dataset(Dataset):
    def __init__(self, h5_paths):
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
        
        # Z-score per channel
        for ch in range(self.data.shape[1]):
            mean = self.data[:, ch, :].mean()
            std = self.data[:, ch, :].std()
            if std > 0:
                self.data[:, ch, :] = (self.data[:, ch, :] - mean) / std
        
        print(f"Loaded {len(self.data)} samples")
        print(f"Shape: {self.data.shape}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return torch.from_numpy(x).float(), torch.tensor(y).float()

# Model
class MultiScaleCNN(nn.Module):
    def __init__(self, n_channels=129, n_times=200, dropout=0.6):
        super().__init__()
        
        # Multi-scale conv blocks
        self.conv1_s = nn.Conv1d(n_channels, 32, kernel_size=3, padding=1)
        self.conv1_m = nn.Conv1d(n_channels, 32, kernel_size=7, padding=3)
        self.conv1_l = nn.Conv1d(n_channels, 32, kernel_size=15, padding=7)
        
        self.bn1 = nn.BatchNorm1d(96)  # 32*3 channels
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(dropout * 0.5)
        
        # Second layer
        self.conv2 = nn.Conv1d(96, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(dropout * 0.7)
        
        # Third layer
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        self.drop3 = nn.Dropout(dropout)
        
        # Global pooling + FC
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.drop_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        # Multi-scale conv
        x1 = F.relu(self.conv1_s(x))
        x2 = F.relu(self.conv1_m(x))
        x3 = F.relu(self.conv1_l(x))
        x = torch.cat([x1, x2, x3], dim=1)
        
        x = self.pool1(F.relu(self.bn1(x)))
        x = self.drop1(x)
        
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.drop2(x)
        
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.drop3(x)
        
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        x = self.fc2(x)
        
        return x.squeeze(-1)

# NRMSE loss
def nrmse_loss(pred, target):
    mse = F.mse_loss(pred, target)
    std = target.std()
    return torch.sqrt(mse) / (std + 1e-8)

# Training
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Validation
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            total_loss += loss.item()
    return total_loss / len(loader)

# Main
if __name__ == '__main__':
    device = torch.device(CONFIG['device'])
    
    # Load data
    h5_files = [
        'data/cached/challenge1_R1_windows.h5',
        'data/cached/challenge1_R2_windows.h5',
        'data/cached/challenge1_R3_windows.h5',
        'data/cached/challenge1_R4_windows.h5'
    ]
    
    dataset = SimpleH5Dataset(h5_files)
    
    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                             shuffle=True, num_workers=CONFIG['num_workers'], 
                             pin_memory=CONFIG['pin_memory'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                           shuffle=False, num_workers=CONFIG['num_workers'], 
                           pin_memory=CONFIG['pin_memory'])
    
    print(f"\n📊 Split: Train={len(train_dataset):,}, Val={len(val_dataset):,}")
    
    # Model
    model = MultiScaleCNN(dropout=CONFIG['dropout']).to(device)
    print(f"\n🧠 Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=CONFIG['lr'], 
                     weight_decay=CONFIG['weight_decay'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nrmse_loss
    
    # Training loop
    best_val = float('inf')
    patience = 0
    max_patience = 10
    
    print("\n" + "="*70)
    print("🚀 TRAINING")
    print("="*70)
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        improved = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'checkpoints/c1_multiscale_quick.pt')
            improved = "✅ BEST"
            patience = 0
        else:
            patience += 1
        
        print(f"Epoch {epoch:3d}: Train={train_loss:.6f} Val={val_loss:.6f} {improved}")
        
        if patience >= max_patience:
            print(f"\n⚠️  Early stop at epoch {epoch}")
            break
    
    print("\n" + "="*70)
    print(f"🎯 Best Validation NRMSE: {best_val:.6f}")
    print(f"🎯 Current Baseline: 1.00019")
    print(f"🎯 Target: < 0.95")
    
    if best_val < 0.95:
        print(f"\n✅ SUCCESS! Target achieved!")
    elif best_val < 1.00019:
        print(f"\n�� Improved over baseline! ({best_val:.6f} < 1.00019)")
    else:
        print(f"\n⚠️  Not improved yet. Keep iterating!")
    
    print("="*70)
