#!/usr/bin/env python3
"""
Challenge 1 Ensemble Training
==============================
Train 5 copies of V8 model with different random seeds.
Average predictions for improved generalization.

Expected: 1-2% improvement (Val NRMSE ~0.150-0.155 from 0.160418)
Time: 30-40 minutes
Risk: Low (same architecture as proven V8)
"""

import os
import sys
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
from datetime import datetime

warnings.filterwarnings('ignore')

print("🎯 Challenge 1: Ensemble Training")
print("="*70)
print("Strategy: Train 5 V8 models with different seeds")
print("Expected: Val NRMSE 0.150-0.155 (3-6% improvement)")
print("Time: 30-40 minutes")
print("="*70)
print()

# Configuration (V8 proven settings - FIXED)
CONFIG = {
    'seeds': [42, 123, 456, 789, 999],
    'batch_size': 64,
    'epochs': 25,
    'lr': 0.001,
    'weight_decay': 0.05,
    'dropout': [0.3, 0.4, 0.5],  # V8 dropout values
    'patience': 8,
    'mixup_alpha': 0.2,  # CRITICAL: V8 used mixup!
}

print("📋 Configuration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
print()


class H5Dataset(Dataset):
    """V8 dataset (proven settings)"""
    def __init__(self, h5_paths, augment=False):
        self.augment = augment
        self.data = []
        self.labels = []
        
        for h5_path in h5_paths:
            print(f"Loading {h5_path}...")
            with h5py.File(h5_path, 'r') as f:
                X = f['eeg'][:]  # C1 uses 'eeg' not 'data'
                y = f['labels'][:]  # C1 uses 'labels' not 'targets'
                self.data.append(X)
                self.labels.append(y)
        
        self.data = np.concatenate(self.data, axis=0).astype(np.float32)
        self.labels = np.concatenate(self.labels, axis=0).astype(np.float32)
        
        print(f"Loaded {len(self.data)} samples")
        print(f"Data shape: {self.data.shape}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X = self.data[idx].copy()
        y = self.labels[idx]
        
        if self.augment:
            # Time shift
            shift = np.random.randint(-10, 11)
            if shift > 0:
                X = np.concatenate([X[:, shift:], X[:, :shift]], axis=1)
            elif shift < 0:
                X = np.concatenate([X[:, shift:], X[:, :shift]], axis=1)
            
            # Amplitude scaling
            scale = np.random.uniform(0.9, 1.1)
            X = X * scale
            
            # Gaussian noise
            noise = np.random.normal(0, 0.01, X.shape).astype(np.float32)
            X = X + noise
            
            # Mixup (10% chance)
            if np.random.random() < 0.1:
                idx2 = np.random.randint(0, len(self.data))
                X2 = self.data[idx2]
                y2 = self.labels[idx2]
                lam = np.random.beta(0.2, 0.2)
                X = lam * X + (1 - lam) * X2
                y = lam * y + (1 - lam) * y2
        
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.float32)


class CompactCNN(nn.Module):
    """V8 proven architecture (75K params) - CORRECT VERSION"""
    def __init__(self, dropout=[0.3, 0.4, 0.5]):
        super().__init__()
        
        self.features = nn.Sequential(
            # Conv1: 129 channels x 200 timepoints -> 32x100
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout[0]),

            # Conv2: 32x100 -> 64x50
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout[1]),

            # Conv3: 64x50 -> 128x25
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout[2]),

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


def validate(model, loader, criterion, device):
    """Validation"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            
            total_loss += loss.item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    # Compute NRMSE correctly: RMSE / std(y_true)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)
    std_targets = np.std(all_targets)
    nrmse = rmse / std_targets if std_targets > 0 else 0.0
    
    return total_loss / len(loader), nrmse


def train_single_model(seed, train_loader, val_loader, device, save_dir):
    """Train one model with given seed"""
    print(f"\n{'='*70}")
    print(f"🌱 Training Model with Seed {seed}")
    print(f"{'='*70}")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create model
    model = CompactCNN(dropout=CONFIG['dropout']).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Parameters: {n_params:,}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2
    )
    
    best_val_nrmse = float('inf')
    patience_counter = 0
    
    for epoch in range(CONFIG['epochs']):
        # Train
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            # Mixup augmentation (V8 used this!)
            if CONFIG['mixup_alpha'] > 0 and np.random.rand() < 0.5:
                lam = np.random.beta(CONFIG['mixup_alpha'], CONFIG['mixup_alpha'])
                batch_size = X.size(0)
                index = torch.randperm(batch_size).to(device)
                mixed_X = lam * X + (1 - lam) * X[index]
                y_a, y_b = y, y[index]
                
                optimizer.zero_grad()
                pred = model(mixed_X)
                loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
            else:
                optimizer.zero_grad()
                pred = model(X)
                loss = criterion(pred, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        val_loss, val_nrmse = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        train_loss /= len(train_loader)
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}: "
              f"Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, "
              f"Val NRMSE={val_nrmse:.6f}, LR={optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_nrmse < best_val_nrmse:
            best_val_nrmse = val_nrmse
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_path = os.path.join(save_dir, f'model_seed{seed}_best.pth')
            torch.save({
                'seed': seed,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_nrmse': val_nrmse,
                'val_loss': val_loss,
            }, checkpoint_path)
            
            # Save weights only
            weights_path = os.path.join(save_dir, f'weights_seed{seed}.pt')
            torch.save(model.state_dict(), weights_path)
            
            print(f"  ✅ New best! Saved to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"  ⏹️  Early stopping at epoch {epoch+1}")
                break
    
    print(f"\n🏆 Best Val NRMSE for Seed {seed}: {best_val_nrmse:.6f}")
    return best_val_nrmse


def main():
    # Setup
    device = torch.device('cpu')  # CPU training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'checkpoints/challenge1_ensemble_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"💾 Checkpoint directory: {save_dir}")
    print()
    
    # Data paths
    train_paths = [
        'data/cached/challenge1_R1_windows.h5',
        'data/cached/challenge1_R2_windows.h5',
        'data/cached/challenge1_R3_windows.h5',
    ]
    val_paths = ['data/cached/challenge1_R4_windows.h5']
    
    # Create datasets (reuse for all seeds)
    print("📦 Loading Data...")
    train_dataset = H5Dataset(train_paths, augment=True)
    val_dataset = H5Dataset(val_paths, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    print(f"✅ Data loaded!")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print()
    
    # Train all models
    start_time = time.time()
    val_nrmses = []
    
    for seed in CONFIG['seeds']:
        val_nrmse = train_single_model(seed, train_loader, val_loader, device, save_dir)
        val_nrmses.append(val_nrmse)
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("🎉 ENSEMBLE TRAINING COMPLETE")
    print("="*70)
    print(f"⏱️  Total Time: {elapsed/60:.1f} minutes")
    print()
    print("📊 Individual Model Results:")
    for seed, nrmse in zip(CONFIG['seeds'], val_nrmses):
        print(f"  Seed {seed:3d}: Val NRMSE = {nrmse:.6f}")
    print()
    print(f"📈 Ensemble Statistics:")
    print(f"  Mean Val NRMSE: {np.mean(val_nrmses):.6f}")
    print(f"  Std Val NRMSE:  {np.std(val_nrmses):.6f}")
    print(f"  Min Val NRMSE:  {np.min(val_nrmses):.6f}")
    print(f"  Max Val NRMSE:  {np.max(val_nrmses):.6f}")
    print()
    print(f"💾 Models saved in: {save_dir}")
    print()
    
    # Compare with V8
    v8_nrmse = 0.160418
    improvement = (v8_nrmse - np.mean(val_nrmses)) / v8_nrmse * 100
    print(f"📊 Comparison with V8:")
    print(f"  V8 Val NRMSE:       {v8_nrmse:.6f}")
    print(f"  Ensemble Val NRMSE: {np.mean(val_nrmses):.6f}")
    print(f"  Improvement:        {improvement:+.2f}%")
    
    if np.mean(val_nrmses) < v8_nrmse:
        print(f"  ✅ Ensemble BETTER than V8!")
    else:
        print(f"  ⚠️  Ensemble NOT better than V8")
    
    print()
    print("="*70)


if __name__ == '__main__':
    main()
