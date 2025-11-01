#!/usr/bin/env python3
"""
Challenge 1 - Phase 1 Aggressive Training
Goal: Push from 1.00019 → 0.98-0.99
- 5 seeds with EMA
- 50 epochs (extended)
- Aggressive augmentation
- Better optimization
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import h5py
import time
from datetime import datetime

# ============================================================
# HYPERPARAMETERS - AGGRESSIVE SETTINGS
# ============================================================
SEEDS = [42, 123, 456, 789, 1337]  # 5 seeds
EPOCHS = 50  # Extended training
BATCH_SIZE = 32
LR = 0.002  # Slightly higher
WEIGHT_DECAY = 0.01  # Reduced for more capacity
MIN_LR = 1e-6

# Early stopping (more patient)
EARLY_STOP_PATIENCE = 15
LR_PATIENCE = 7
LR_FACTOR = 0.5

# EMA
EMA_DECAY = 0.999

# Mixup (more aggressive)
MIXUP_ALPHA = 0.2  # Increased
PROB_MIXUP_BATCH = 0.6  # More frequent

# Aggressive augmentation
PROB_TIME_SHIFT = 0.7
PROB_AMP_SCALE = 0.7
PROB_NOISE = 0.5

# Gradient clipping
GRAD_CLIP = 1.0

# Label smoothing
LABEL_SMOOTH = 0.05

# ============================================================
# Model Architecture - Enhanced CompactCNN
# ============================================================
class EnhancedCompactCNN(nn.Module):
    """Enhanced CompactCNN with more aggressive regularization"""
    
    def __init__(self, dropout_rate=0.6):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate + 0.05),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate + 0.1),
        )
        
        # Spatial attention
        self.spatial_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(128, 128, 1),
            nn.Sigmoid()
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        features = self.features(x)
        attention = self.spatial_attn(features)
        features = features * attention
        output = self.regressor(features)
        return output

# ============================================================
# EMA Helper Class
# ============================================================
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.module = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# ============================================================
# Data Loading
# ============================================================
def load_c1_data():
    """Load Challenge 1 data"""
    print("Loading Challenge 1 data...")
    
    data_path = Path("data/processed/challenge1_data.h5")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        X_train = torch.from_numpy(f['X_train'][:]).float()
        y_train = torch.from_numpy(f['y_train'][:]).float()
        X_val = torch.from_numpy(f['X_val'][:]).float()
        y_val = torch.from_numpy(f['y_val'][:]).float()
    
    print(f"Train: X {X_train.shape}, y {y_train.shape}")
    print(f"Val:   X {X_val.shape}, y {y_val.shape}")
    
    return X_train, y_train, X_val, y_val

# ============================================================
# Augmentation Functions
# ============================================================
def time_shift_augment(X, max_shift=5):
    """Time shift augmentation"""
    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    return torch.roll(X, shifts=shift, dims=1)  # dims=1 for time axis (channels, time)

def amplitude_scale_augment(X, scale_range=0.1):
    """Amplitude scaling augmentation"""
    scale = 1.0 + torch.rand(1).item() * 2 * scale_range - scale_range
    return X * scale

def noise_augment(X, noise_std=0.02):
    """Add Gaussian noise"""
    noise = torch.randn_like(X) * noise_std
    return X + noise

def apply_augmentation(X):
    """Apply random augmentations"""
    if torch.rand(1).item() < PROB_TIME_SHIFT:
        X = time_shift_augment(X)
    if torch.rand(1).item() < PROB_AMP_SCALE:
        X = amplitude_scale_augment(X)
    if torch.rand(1).item() < PROB_NOISE:
        X = noise_augment(X)
    return X

def mixup_batch(X, y, alpha=0.2):
    """Mixup augmentation for batch"""
    if alpha > 0 and torch.rand(1).item() < PROB_MIXUP_BATCH:
        lam = np.random.beta(alpha, alpha)
        batch_size = X.size(0)
        index = torch.randperm(batch_size)
        
        mixed_X = lam * X + (1 - lam) * X[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]
        return mixed_X, mixed_y
    return X, y

def smooth_labels(y, epsilon=0.05):
    """Label smoothing for regression"""
    noise = torch.randn_like(y) * epsilon
    return y + noise

# ============================================================
# Training Function
# ============================================================
def train_one_seed(seed):
    """Train model for one seed"""
    print(f"\n{'='*80}")
    print(f"Training Seed {seed}")
    print(f"{'='*80}\n")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load data
    X_train, y_train, X_val, y_val = load_c1_data()
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Device
    device = torch.device('cpu')  # Use CPU (ROCm unstable)
    
    # Model
    model = EnhancedCompactCNN(dropout_rate=0.6).to(device)
    ema = ModelEMA(model, decay=EMA_DECAY)
    
    # Loss & optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR
    )
    
    # Training tracking
    best_val_loss = float('inf')
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Augmentation
            X_aug = torch.stack([apply_augmentation(x) for x in X_batch])
            
            # Label smoothing
            y_smooth = smooth_labels(y_batch, LABEL_SMOOTH)
            
            # Mixup
            X_aug, y_smooth = mixup_batch(X_aug, y_smooth, MIXUP_ALPHA)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(X_aug).squeeze()
            loss = criterion(outputs, y_smooth)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            
            optimizer.step()
            
            # Update EMA
            ema.update(model)
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation (with EMA)
        ema.apply_shadow()
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        ema.restore()
        
        # LR scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate NRMSE
        val_nrmse = np.sqrt(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Val NRMSE: {val_nrmse:.6f} | "
              f"LR: {current_lr:.6f}")
        
        # Save best model (EMA version)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save EMA checkpoint
            ema.apply_shadow()
            checkpoint = {
                'epoch': epoch + 1,
                'seed': seed,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'nrmse': val_nrmse,
                'metrics': {
                    'val_nrmse': val_nrmse,
                    'val_loss': val_loss,
                }
            }
            torch.save(checkpoint, f'checkpoints/c1_phase1_seed{seed}_ema_best.pt')
            ema.restore()
            
            print(f"  → Best model saved! (Val NRMSE: {val_nrmse:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n  Early stopping triggered at epoch {epoch+1}")
                break
    
    elapsed = (time.time() - start_time) / 60
    print(f"\nSeed {seed} training complete!")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Best Val NRMSE: {np.sqrt(best_val_loss):.6f}")
    print(f"Time: {elapsed:.1f} minutes")
    
    return best_val_loss

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("="*80)
    print("Challenge 1 - Phase 1 Aggressive Training")
    print("="*80)
    print(f"Seeds: {SEEDS}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LR}")
    print(f"EMA Decay: {EMA_DECAY}")
    print(f"Mixup Alpha: {MIXUP_ALPHA}")
    print("="*80)
    print()
    
    # Create checkpoints directory
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Train all seeds
    results = {}
    for seed in SEEDS:
        best_val_loss = train_one_seed(seed)
        results[seed] = best_val_loss
    
    # Summary
    print("\n" + "="*80)
    print("Training Complete - All Seeds")
    print("="*80)
    print("\nResults:")
    for seed, val_loss in results.items():
        nrmse = np.sqrt(val_loss)
        print(f"  Seed {seed:4d}: Val Loss {val_loss:.6f}, NRMSE {nrmse:.6f}")
    
    avg_nrmse = np.mean([np.sqrt(v) for v in results.values()])
    std_nrmse = np.std([np.sqrt(v) for v in results.values()])
    cv = (std_nrmse / avg_nrmse) * 100
    
    print(f"\nEnsemble Statistics:")
    print(f"  Mean NRMSE: {avg_nrmse:.6f}")
    print(f"  Std NRMSE:  {std_nrmse:.6f}")
    print(f"  CV:         {cv:.2f}%")
    print(f"\n✅ All seeds trained successfully!")
    print(f"Expected competition score: ~{avg_nrmse * 0.01:.5f}")  # Rough estimate
