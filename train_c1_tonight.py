#!/usr/bin/env python3
"""
Challenge 1 Training - Anti-Overfitting Edition
Tonight's training run to beat untrained baseline!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
from pathlib import Path
import time
from datetime import datetime

print("\n" + "="*70)
print("🧠 Challenge 1: Anti-Overfitting Training")
print("="*70)
print("Goal: Beat untrained baseline (1.0015)")
print("Strategy: Strong regularization + early stopping + data augmentation")
print("="*70 + "\n")

# Configuration
CONFIG = {
    'batch_size': 64,
    'epochs': 15,
    'lr': 0.001,
    'weight_decay': 0.05,
    'dropout_conv': [0.5, 0.6, 0.7],
    'dropout_fc': [0.6, 0.5],
    'patience': 5,
    'mixup_alpha': 0.2,
    'device': 'cpu'  # Force CPU to avoid GPU library issues
}

print(f"Device: {CONFIG['device']}")
print(f"Batch size: {CONFIG['batch_size']}")
print(f"Max epochs: {CONFIG['epochs']}")
print(f"Learning rate: {CONFIG['lr']}")
print(f"Weight decay: {CONFIG['weight_decay']}")
print(f"Dropout: {CONFIG['dropout_conv']}, {CONFIG['dropout_fc']}")
print()


class CachedH5Dataset(Dataset):
    """Load Challenge 1 data from cached H5 files"""
    def __init__(self, h5_paths, augment=False):
        self.augment = augment
        self.data = []
        self.labels = []
        
        for h5_path in h5_paths:
            print(f"Loading {h5_path}...")
            with h5py.File(h5_path, 'r') as f:
                X = f['eeg'][:]  # Shape: (n_samples, 129, 200)
                y = f['labels'][:]  # Shape: (n_samples,)
                self.data.append(X)
                self.labels.append(y)
        
        self.data = np.concatenate(self.data, axis=0).astype(np.float32)
        self.labels = np.concatenate(self.labels, axis=0).astype(np.float32)
        
        print(f"Loaded {len(self.data)} samples")
        print(f"Data shape: {self.data.shape}")
        print(f"Labels shape: {self.labels.shape}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx].astype(np.float32)
        y = self.labels[idx].astype(np.float32)
        
        # Data augmentation
        if self.augment:
            # Time shift
            if np.random.rand() < 0.5:
                shift = np.random.randint(-10, 10)
                x = np.roll(x, shift, axis=1)
            
            # Amplitude scaling
            if np.random.rand() < 0.5:
                scale = np.random.uniform(0.9, 1.1)
                x = x * scale
            
            # Add noise
            if np.random.rand() < 0.3:
                noise = np.random.normal(0, 0.01, x.shape).astype(np.float32)
                x = x + noise
        
        return torch.from_numpy(x).float(), torch.tensor(y).float()


class ImprovedCompactCNN(nn.Module):
    """CompactCNN with strong regularization"""
    def __init__(self, dropout_conv, dropout_fc):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_conv[0]),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_conv[1]),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_conv[2]),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_fc[0]),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_fc[1]),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.regressor(features)
        return output


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_epoch(model, loader, criterion, optimizer, device, mixup_alpha=0.2):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        
        # Mixup
        if mixup_alpha > 0:
            X, y_a, y_b, lam = mixup_data(X, y, mixup_alpha)
            optimizer.zero_grad()
            pred = model(X).squeeze()
            loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
        else:
            optimizer.zero_grad()
            pred = model(X).squeeze()
            loss = criterion(pred, y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)}, Loss: {loss.item():.6f}")
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).squeeze()
            loss = criterion(pred, y)
            total_loss += loss.item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # NRMSE
    rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
    nrmse = rmse / (all_labels.max() - all_labels.min())
    
    return total_loss / len(loader), nrmse


def main():
    # Load data
    print("📦 Loading Data...")
    train_paths = [
        'data/cached/challenge1_R1_windows.h5',
        'data/cached/challenge1_R2_windows.h5',
        'data/cached/challenge1_R3_windows.h5',
    ]
    val_paths = [
        'data/cached/challenge1_R4_windows.h5',
    ]
    
    train_dataset = CachedH5Dataset(train_paths, augment=True)
    val_dataset = CachedH5Dataset(val_paths, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()
    
    # Create model
    print("🏗️  Creating Model...")
    model = ImprovedCompactCNN(
        dropout_conv=CONFIG['dropout_conv'],
        dropout_fc=CONFIG['dropout_fc']
    ).to(CONFIG['device'])
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print()
    
    # Setup training
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], 
                           weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    
    # Training loop
    print("🚀 Starting Training...")
    print("="*70)
    
    best_val_loss = float('inf')
    best_nrmse = float('inf')
    patience_counter = 0
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(f'checkpoints/challenge1_improved_{timestamp}')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(CONFIG['epochs']):
        start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        print("-" * 70)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, 
                                CONFIG['device'], CONFIG['mixup_alpha'])
        
        # Validate
        val_loss, val_nrmse = validate(model, val_loader, criterion, CONFIG['device'])
        
        # Learning rate schedule
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        epoch_time = time.time() - start_time
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Val NRMSE: {val_nrmse:.6f}")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_nrmse': val_nrmse,
                'config': CONFIG
            }
            
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_dir / 'best_weights.pt')
            
            print(f"  ✅ New best! Saved to {checkpoint_dir}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{CONFIG['patience']})")
        
        # Early stopping
        if patience_counter >= CONFIG['patience']:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            break
    
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    print(f"Best Val NRMSE: {best_nrmse:.6f}")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"\nCompare with untrained baseline: 1.0015")
    print("If Val NRMSE < 0.18, likely to beat baseline!")
    print("="*70)


if __name__ == '__main__':
    main()
