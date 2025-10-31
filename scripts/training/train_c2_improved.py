"""
Challenge 2 Training - Stimulus Reconstruction
Target: Improve from 1.0087 to ~1.0 or better
Strategy: Flexible architecture with strong anti-overfitting
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

# Configuration - FLEXIBLE and ANTI-OVERFITTING
CONFIG = {
    'batch_size': 32,
    'epochs': 40,
    'lr': 0.0005,  # Lower LR for more stable training
    'weight_decay': 0.05,  # Strong regularization
    'dropout_encoder': [0.4, 0.5, 0.6],  # Progressive dropout
    'dropout_decoder': [0.5, 0.4],
    'patience': 8,
    'mixup_alpha': 0.3,
    'device': 'cpu',
    'use_cosine_annealing': True,
    'gradient_clip': 1.0,
    'label_smoothing': 0.1,  # For classification stability
}

print("="*70)
print("🧠 CHALLENGE 2 TRAINING - Stimulus Reconstruction")
print("="*70)
print(f"Current baseline: 1.0087")
print(f"Target: < 1.00 (preferably ~0.95-1.00)")
print(f"Strategy: Flexible encoder-decoder with strong regularization")
print("="*70)
print()

for k, v in CONFIG.items():
    print(f"{k}: {v}")
print()


class C2Dataset(Dataset):
    """Challenge 2 dataset with augmentation"""
    def __init__(self, h5_paths, augment=False):
        self.augment = augment
        self.data = []
        self.labels = []
        
        for h5_path in h5_paths:
            print(f"Loading {h5_path}...")
            with h5py.File(h5_path, 'r') as f:
                X = f['data'][:]  # (n_samples, 129, 400)
                y_flat = f['targets'][:]  # (n_samples,) - flattened targets
                
                # Convert flat targets back to 3 dimensions
                # Target = dim1*9 + dim2*3 + dim3 (each dim in 0,1,2)
                y = np.zeros((len(y_flat), 3), dtype=np.int64)
                y[:, 0] = y_flat // 9  # dim1
                y[:, 1] = (y_flat % 9) // 3  # dim2
                y[:, 2] = y_flat % 3  # dim3
                
                self.data.append(X)
                self.labels.append(y)
        
        self.data = np.concatenate(self.data, axis=0).astype(np.float32)
        self.labels = np.concatenate(self.labels, axis=0).astype(np.int64)
        
        # Z-score normalization per channel
        print("Applying z-score normalization per channel...")
        for ch in range(self.data.shape[1]):
            mean = self.data[:, ch, :].mean()
            std = self.data[:, ch, :].std()
            if std > 0:
                self.data[:, ch, :] = (self.data[:, ch, :] - mean) / std
        
        print(f"Loaded {len(self.data)} samples")
        print(f"Data shape: {self.data.shape}")
        print(f"Labels shape: {self.labels.shape}")
        print(f"Label range: {self.labels.min()} to {self.labels.max()}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx].copy()
        y = self.labels[idx]
        
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
            
            # Channel dropout (milder for C2)
            if np.random.rand() < 0.2:
                n_drop = np.random.randint(3, 10)
                drop_channels = np.random.choice(x.shape[0], n_drop, replace=False)
                x[drop_channels, :] = 0
        
        return torch.from_numpy(x).float(), torch.tensor(y).long()


class FlexibleStimulusDecoder(nn.Module):
    """
    Flexible encoder-decoder for stimulus reconstruction
    Design principles:
    - Multi-scale temporal feature extraction
    - Attention mechanisms for important features
    - Strong regularization to prevent overfitting
    - Residual connections for gradient flow
    """
    def __init__(self, dropout_encoder, dropout_decoder, num_classes=3):
        super().__init__()
        
        # Encoder: Extract features from EEG
        self.encoder = nn.Sequential(
            # Block 1: Capture high-freq features
            nn.Conv1d(129, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_encoder[0]),
            
            # Block 2: Medium-freq features
            nn.Conv1d(64, 96, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(dropout_encoder[1]),
            
            # Block 3: Low-freq features
            nn.Conv1d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_encoder[2]),
        )
        
        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.Sigmoid()
        )
        
        # Global feature extraction
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Decoder: Reconstruct stimulus classes
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Dropout(dropout_decoder[0]),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(dropout_decoder[1]),
        )
        
        # Separate heads for each stimulus dimension
        self.head_dim1 = nn.Linear(64, num_classes)
        self.head_dim2 = nn.Linear(64, num_classes)
        self.head_dim3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # Encode
        features = self.encoder(x)
        
        # Apply attention
        attn = self.temporal_attention(features).unsqueeze(-1)
        features = features * attn
        
        # Global pooling
        features = self.global_pool(features)
        
        # Decode
        decoded = self.decoder(features)
        
        # Predict each dimension
        out1 = self.head_dim1(decoded)
        out2 = self.head_dim2(decoded)
        out3 = self.head_dim3(decoded)
        
        return out1, out2, out3


def mixup_data(x, y, alpha=0.3):
    """Mixup for multi-label"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


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
            out1, out2, out3 = model(X)
            
            loss = lam * (
                criterion(out1, y_a[:, 0]) + 
                criterion(out2, y_a[:, 1]) + 
                criterion(out3, y_a[:, 2])
            ) + (1 - lam) * (
                criterion(out1, y_b[:, 0]) + 
                criterion(out2, y_b[:, 1]) + 
                criterion(out3, y_b[:, 2])
            )
            loss = loss / 3.0
        else:
            optimizer.zero_grad()
            out1, out2, out3 = model(X)
            
            loss = (
                criterion(out1, y[:, 0]) + 
                criterion(out2, y[:, 1]) + 
                criterion(out3, y[:, 2])
            ) / 3.0
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip'])
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
    correct = [0, 0, 0]
    total = 0
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out1, out2, out3 = model(X)
            
            loss = (
                criterion(out1, y[:, 0]) + 
                criterion(out2, y[:, 1]) + 
                criterion(out3, y[:, 2])
            ) / 3.0
            
            total_loss += loss.item()
            
            # Accuracy per dimension
            _, pred1 = out1.max(1)
            _, pred2 = out2.max(1)
            _, pred3 = out3.max(1)
            
            correct[0] += pred1.eq(y[:, 0]).sum().item()
            correct[1] += pred2.eq(y[:, 1]).sum().item()
            correct[2] += pred3.eq(y[:, 2]).sum().item()
            total += y.size(0)
    
    acc = [c / total * 100 for c in correct]
    return total_loss / len(loader), acc


def main():
    # Data paths
    train_paths = [
        'data/cached/challenge2_R1_windows.h5',
        'data/cached/challenge2_R2_windows.h5',
        'data/cached/challenge2_R3_windows.h5',
    ]
    val_paths = ['data/cached/challenge2_R4_windows.h5']
    
    # Create datasets
    print("📦 Loading Data...")
    train_dataset = C2Dataset(train_paths, augment=True)
    val_dataset = C2Dataset(val_paths, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                           shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()
    
    # Create model
    print("🏗️  Creating Flexible Stimulus Decoder...")
    device = torch.device(CONFIG['device'])
    model = FlexibleStimulusDecoder(
        CONFIG['dropout_encoder'], 
        CONFIG['dropout_decoder']
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    print()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Learning rate scheduler
    if CONFIG['use_cosine_annealing']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
    
    # Training loop
    print("🚀 Starting Training...")
    print("="*70)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(f'checkpoints/challenge2_improved_{timestamp}')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(CONFIG['epochs']):
        start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        print("-"*70)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
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
        print(f"  Val Acc: Dim1={val_acc[0]:.2f}%, Dim2={val_acc[1]:.2f}%, Dim3={val_acc[2]:.2f}%")
        print(f"  Avg Acc: {np.mean(val_acc):.2f}%")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
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
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Checkpoint: {ckpt_dir}")
    print()
    print("Compare with current baseline: 1.0087")
    print("If Val Loss significantly improved, create V9 submission!")
    print("="*70)


if __name__ == '__main__':
    main()
