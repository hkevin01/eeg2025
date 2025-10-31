"""
Advanced Challenge 1 Training - Target: < 0.91
Multiple strategies to push beyond current 1.0002
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
import time
from datetime import datetime

# Configuration
CONFIG = {
    'batch_size': 32,  # Smaller for more updates
    'epochs': 50,  # Longer training
    'lr': 0.001,
    'weight_decay': 0.1,  # Even stronger!
    'dropout_conv': [0.6, 0.7, 0.75],  # Stronger dropout
    'dropout_fc': [0.7, 0.6],
    'patience': 10,  # More patience
    'mixup_alpha': 0.4,  # Stronger mixup
    'device': 'cpu',
    'use_cosine_annealing': True,
    'channel_dropout_prob': 0.3,  # NEW: drop entire channels
    'cutout_prob': 0.3,  # NEW: temporal cutout
}

print("="*70)
print("🚀 AGGRESSIVE C1 TRAINING - TARGET: < 0.91")
print("="*70)
print(f"Current best: 1.0002 (test), 0.160418 (val NRMSE)")
print(f"Target: < 0.91 (9% improvement needed!)")
print("="*70)
print()

for k, v in CONFIG.items():
    print(f"{k}: {v}")
print()


class AdvancedH5Dataset(Dataset):
    """Enhanced dataset with aggressive augmentation"""
    def __init__(self, h5_paths, augment=False):
        self.augment = augment
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
        
        if self.augment:
            # 1. Time shift
            if np.random.rand() < 0.5:
                shift = np.random.randint(-15, 15)
                x = np.roll(x, shift, axis=1)
            
            # 2. Amplitude scaling
            if np.random.rand() < 0.5:
                scale = np.random.uniform(0.85, 1.15)
                x = x * scale
            
            # 3. Add noise
            if np.random.rand() < 0.4:
                noise = np.random.normal(0, 0.02, x.shape).astype(np.float32)
                x = x + noise
            
            # 4. Channel dropout (NEW!)
            if np.random.rand() < CONFIG['channel_dropout_prob']:
                n_drop = np.random.randint(5, 20)
                drop_channels = np.random.choice(x.shape[0], n_drop, replace=False)
                x[drop_channels, :] = 0
            
            # 5. Temporal cutout (NEW!)
            if np.random.rand() < CONFIG['cutout_prob']:
                cutout_len = np.random.randint(10, 30)
                cutout_start = np.random.randint(0, x.shape[1] - cutout_len)
                x[:, cutout_start:cutout_start+cutout_len] = 0
        
        return torch.from_numpy(x).float(), torch.tensor(y).float()


class EnhancedCompactCNN(nn.Module):
    """Deeper model with attention mechanisms"""
    def __init__(self, dropout_conv, dropout_fc):
        super().__init__()
        
        # Deeper feature extraction
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(129, 48, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(dropout_conv[0]),
            
            # Block 2
            nn.Conv1d(48, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_conv[1]),
            
            # Block 3 (NEW - deeper)
            nn.Conv1d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(dropout_conv[2]),
        )
        
        # Temporal attention (NEW!)
        self.temporal_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 96),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(dropout_fc[0]),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_fc[1]),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        
        # Apply attention
        attn = self.temporal_attention(x).unsqueeze(-1)
        x = x * attn
        
        # Classification
        x = self.classifier(x)
        return x.squeeze(-1)


def mixup_data(x, y, alpha=0.4):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(model, loader, criterion, optimizer, device, use_mixup=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch_idx, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        
        # Mixup
        if use_mixup and np.random.rand() < 0.5:
            X, y_a, y_b, lam = mixup_data(X, y, CONFIG['mixup_alpha'])
            
            optimizer.zero_grad()
            pred = model(X)
            loss = mixup_criterion(criterion, pred, y_a, y_b, lam)
        else:
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
        
        loss.backward()
        
        # Gradient clipping (NEW!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)}, Loss: {loss.item():.6f}")
    
    return total_loss / n_batches


def validate(model, loader, criterion, device):
    """Validate model"""
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
    
    # Compute NRMSE
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    target_std = np.std(all_targets)
    nrmse = rmse / target_std if target_std > 0 else rmse
    
    return total_loss / len(loader), nrmse


def main():
    # Data paths
    train_paths = [
        'data/cached/challenge1_R1_windows.h5',
        'data/cached/challenge1_R2_windows.h5',
        'data/cached/challenge1_R3_windows.h5',
    ]
    val_paths = ['data/cached/challenge1_R4_windows.h5']
    
    # Create datasets
    print("📦 Loading Data...")
    train_dataset = AdvancedH5Dataset(train_paths, augment=True)
    val_dataset = AdvancedH5Dataset(val_paths, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()
    
    # Create model
    print("🏗️  Creating Enhanced Model...")
    device = torch.device(CONFIG['device'])
    model = EnhancedCompactCNN(CONFIG['dropout_conv'], CONFIG['dropout_fc']).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    print()
    
    # Loss and optimizer
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], 
                                  weight_decay=CONFIG['weight_decay'])
    
    # Learning rate scheduler
    if CONFIG['use_cosine_annealing']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
    
    # Training loop
    print("🚀 Starting Training...")
    print("="*70)
    
    best_val_nrmse = float('inf')
    patience_counter = 0
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(f'checkpoints/challenge1_aggressive_{timestamp}')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(CONFIG['epochs']):
        start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        print("-"*70)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_nrmse = validate(model, val_loader, criterion, device)
        
        # Update LR
        if CONFIG['use_cosine_annealing']:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - start_time
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Val NRMSE: {val_nrmse:.6f}")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_nrmse < best_val_nrmse:
            best_val_nrmse = val_nrmse
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': val_nrmse,
                'val_loss': val_loss,
            }, ckpt_dir / 'best_model.pth')
            
            # Save just weights for submission
            torch.save(model.state_dict(), ckpt_dir / 'best_weights.pt')
            
            print(f"  ✅ New best! Saved to {ckpt_dir}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{CONFIG['patience']})")
        
        # Early stopping
        if patience_counter >= CONFIG['patience']:
            print(f"\n⏹️  Early stopping triggered at epoch {epoch+1}")
            break
    
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    print(f"Best Val NRMSE: {best_val_nrmse:.6f}")
    print(f"Checkpoint: {ckpt_dir}")
    print()
    print("Compare with current: 0.160418")
    if best_val_nrmse < 0.160418:
        improvement = ((0.160418 - best_val_nrmse) / 0.160418) * 100
        print(f"✅ Improved by {improvement:.2f}%!")
    else:
        print("⚠️  Did not beat current best")
    print("="*70)


if __name__ == '__main__':
    main()
