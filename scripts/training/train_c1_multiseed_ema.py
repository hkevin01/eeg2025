#!/usr/bin/env python3
"""
Challenge 1 - Variance Reduction Multi-Seed Training
- 3-5 seeds with EMA (decay=0.999)
- Conservative augmentation
- Early stopping + ReduceLROnPlateau
- Goal: C1 from 1.00019 → 1.00009-1.00012
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
# HYPERPARAMETERS (Variance Reduction Focused)
# ============================================================
SEEDS = [42, 123, 456]  # Start with 3, can expand to 5
EPOCHS = 25
BATCH_SIZE = 32
LR = 0.001
WEIGHT_DECAY = 0.05
MIN_LR = 1e-5

# Early stopping & LR scheduler
EARLY_STOP_PATIENCE = 10
LR_PATIENCE = 5
LR_FACTOR = 0.5

# EMA
EMA_DECAY = 0.999

# Mixup (very conservative for C1 regression)
MIXUP_ALPHA = 0.1  # Lower than C2
PROB_MIXUP_BATCH = 0.5  # Less frequent

# Conservative augmentation
PROB_TIME_SHIFT = 0.4  # Reduced
PROB_AMP_SCALE = 0.4   # Reduced
PROB_NOISE = 0.2       # Reduced

# ============================================================
# Model Architecture (CompactResponseTimeCNN)
# ============================================================
class CompactResponseTimeCNN(nn.Module):
    """Compact CNN for response time prediction - proven architecture"""
    
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.6),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.7),
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
            nn.Dropout(0.5),
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
    """Exponential Moving Average for model weights"""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.original[name])

# ============================================================
# Dataset with Conservative Augmentation
# ============================================================
class C1Dataset(torch.utils.data.Dataset):
    """Challenge 1 dataset with conservative probabilistic augmentation"""
    
    def __init__(self, h5_paths, augment=False):
        self.data = []
        self.targets = []
        self.augment = augment
        
        for h5_path in h5_paths:
            print(f"Loading {h5_path}...")
            with h5py.File(h5_path, 'r') as f:
                X = f['data'][:]
                y = f['targets'][:]
                
                valid_mask = y != -1
                X = X[valid_mask]
                y = y[valid_mask]
                
                self.data.append(X)
                self.targets.append(y)
        
        self.data = np.concatenate(self.data, axis=0).astype(np.float32)
        self.targets = np.concatenate(self.targets, axis=0).astype(np.float32)
        
        # Z-score normalization per channel
        for ch in range(self.data.shape[1]):
            mean = self.data[:, ch, :].mean()
            std = self.data[:, ch, :].std()
            if std > 0:
                self.data[:, ch, :] = (self.data[:, ch, :] - mean) / std
        
        print(f"✅ Loaded {len(self.data)} samples")
        print(f"   Data shape: {self.data.shape}")
        print(f"   Target range: [{self.targets.min():.3f}, {self.targets.max():.3f}]")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx].copy()
        y = self.targets[idx]
        
        if self.augment:
            # Conservative augmentation (lower probabilities)
            if np.random.rand() < PROB_TIME_SHIFT:
                shift = np.random.randint(-5, 6)  # Smaller shifts
                x = np.roll(x, shift, axis=1)
            
            if np.random.rand() < PROB_AMP_SCALE:
                scale = 0.95 + 0.1 * np.random.rand()  # 0.95-1.05 (tighter)
                x = x * scale
            
            if np.random.rand() < PROB_NOISE:
                noise = np.random.randn(*x.shape) * 0.02  # Lower noise
                x = x + noise
        
        return torch.FloatTensor(x), torch.FloatTensor([y])

# ============================================================
# Mixup
# ============================================================
def mixup_data(x, y, alpha=0.1):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============================================================
# Training Function
# ============================================================
def train_model(seed, train_dataset, val_dataset, device):
    print("=" * 60)
    print(f"🌱 Training Model with Seed {seed}")
    print("=" * 60)
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    
    # Create model
    model = CompactResponseTimeCNN().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize EMA
    ema = ModelEMA(model, decay=EMA_DECAY)
    print(f"EMA initialized with decay={EMA_DECAY}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LR,
        verbose=True
    )
    
    criterion = nn.SmoothL1Loss()
    
    # DataLoaders with fixed seeds
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Conservative mixup
            if np.random.rand() < PROB_MIXUP_BATCH:
                x_batch, y_a, y_b, lam = mixup_data(x_batch, y_batch, alpha=MIXUP_ALPHA)
                
                optimizer.zero_grad()
                outputs = model(x_batch).squeeze(-1)
                loss = mixup_criterion(criterion, outputs, y_a.squeeze(-1), y_b.squeeze(-1), lam)
            else:
                optimizer.zero_grad()
                outputs = model(x_batch).squeeze(-1)
                loss = criterion(outputs, y_batch.squeeze(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update EMA
            ema.update(model)
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate with EMA weights
        ema.apply_shadow(model)
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(x_batch).squeeze(-1)
                loss = criterion(outputs, y_batch.squeeze(-1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        ema.restore(model)
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save checkpoint with EMA weights
            ema.apply_shadow(model)
            checkpoint_path = f'checkpoints/c1_multiseed_seed{seed}_ema_best.pt'
            torch.save({
                'epoch': epoch,
                'seed': seed,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'nrmse': val_loss,
                'metrics': {
                    'nrmse': val_loss,
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'ema_decay': EMA_DECAY
                }
            }, checkpoint_path)
            ema.restore(model)
        else:
            patience_counter += 1
        
        # Print progress
        elapsed = time.time() - start_time
        eta = (elapsed / epoch) * (EPOCHS - epoch)
        
        status = "✅" if val_loss == best_val_loss else "  "
        print(f"{status} Epoch {epoch:2d}/{EPOCHS} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"Best: {best_val_loss:.6f} (e{best_epoch}) | "
              f"LR: {current_lr:.2e} | Patience: {patience_counter}/{EARLY_STOP_PATIENCE} | "
              f"ETA: {eta/60:.1f}m")
        
        # Early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
            break
    
    total_time = time.time() - start_time
    print(f"\n✅ Seed {seed} complete!")
    print(f"   Best Val: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"   Time: {total_time/60:.1f} min")
    print()
    
    return best_val_loss, best_epoch

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("🧠 Challenge 1 - Variance Reduction Multi-Seed")
    print("=" * 60)
    print(f"Strategy: {len(SEEDS)} seeds + EMA (decay={EMA_DECAY})")
    print(f"Goal: C1 from 1.00019 → 1.00009-1.00012")
    print()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Load datasets
    print("Loading datasets...")
    train_paths = [
        'data/processed/challenge1_R1.h5',
        'data/processed/challenge1_R2.h5',
        'data/processed/challenge1_R3.h5'
    ]
    val_paths = ['data/processed/challenge1_R4.h5']
    
    train_dataset = C1Dataset(train_paths, augment=True)
    val_dataset = C1Dataset(val_paths, augment=False)
    
    print(f"\n✅ Data ready: {len(train_dataset)} train, {len(val_dataset)} val samples")
    print()
    
    # Train ensemble
    results = []
    
    for seed in SEEDS:
        val_loss, best_epoch = train_model(seed, train_dataset, val_dataset, device)
        results.append((seed, val_loss, best_epoch))
    
    # Summary
    print("=" * 60)
    print("🎉 Multi-Seed Training Complete!")
    print("=" * 60)
    for seed, val_loss, best_epoch in results:
        print(f"Seed {seed:4d}: Val Loss = {val_loss:.6f} (epoch {best_epoch})")
    print()
    
    avg_val_loss = np.mean([vl for _, vl, _ in results])
    std_val_loss = np.std([vl for _, vl, _ in results])
    print(f"Ensemble Statistics:")
    print(f"  Mean Val Loss: {avg_val_loss:.6f}")
    print(f"  Std Val Loss:  {std_val_loss:.6f}")
    print(f"  CV: {100*std_val_loss/avg_val_loss:.2f}%")
    print()
    
    print("Next steps:")
    print("1. Create ensemble submission with TTA")
    print("2. Add linear calibration on R4 validation")
    print("3. Combine with C2 ensemble for V12")
    print("=" * 60)

if __name__ == "__main__":
    main()
