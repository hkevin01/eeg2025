#!/usr/bin/env python3
"""
AMD RX 5600 XT Optimized Training - Simple Version
===================================================

Optimized for AMD Radeon RX 5600 XT (RDNA 1.0, gfx1010)
- Suppresses hipBLASLt warnings
- Conservative memory management for 6GB VRAM
- Competition-specific enhancements
"""
import os
import sys
from pathlib import Path

# Fix AMD hipBLASLt issue BEFORE importing torch
os.environ['ROCBLAS_LAYER'] = '1'
os.environ['HIPBLASLT_LOG_LEVEL'] = '0'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.1.0'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.models.eeg_dataset_simple import SimpleEEGDataset

# AMD RX 5600 XT Configuration
CONFIG = {
    'batch_size': 8,  # Conservative for 6GB VRAM
    'epochs': 20,
    'learning_rate': 1e-4,
    'max_samples': 1500,
    'd_model': 96,
    'n_heads': 6,
    'n_layers': 4,
    'dropout': 0.1,
    'num_workers': 2,
}

class SimpleEEGModel(nn.Module):
    """Simple EEG model optimized for AMD RX 5600 XT"""
    def __init__(self, n_channels=129, d_model=96, n_heads=6, n_layers=4, num_classes=2):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(n_channels, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 5000, d_model) * 0.02)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=CONFIG['dropout'],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(CONFIG['dropout']),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(CONFIG['dropout']),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, channels, time)
        x = x.transpose(1, 2)  # (batch, time, channels)
        x = self.input_proj(x)  # (batch, time, d_model)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device, non_blocking=False), target.to(device, non_blocking=False)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # AMD cleanup
        if batch_idx % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if (batch_idx + 1) % 10 == 0:
            acc = 100. * correct / total
            print(f"   Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f} | Acc: {acc:.1f}%")
            
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1024**2
                print(f"   GPU Memory: {mem:.0f}MB")
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device, non_blocking=False), target.to(device, non_blocking=False)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

def main():
    print("="*80)
    print("🚀 AMD RX 5600 XT OPTIMIZED TRAINING")
    print("="*80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print("✅ hipBLASLt warnings suppressed")
    
    print(f"\nConfig:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    data_dir = Path(__file__).parent.parent / "data" / "raw" / "hbn"
    full_dataset = SimpleEEGDataset(data_dir=data_dir, max_subjects=None, verbose=True)
    
    # Limit samples
    if len(full_dataset) > CONFIG['max_samples']:
        indices = torch.randperm(len(full_dataset))[:CONFIG['max_samples']]
        from torch.utils.data import Subset
        dataset = Subset(full_dataset, indices)
        print(f"✅ Using {len(dataset)} samples (limited from {len(full_dataset)})")
    else:
        dataset = full_dataset
        print(f"✅ Using all {len(dataset)} samples")
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=False  # Better for AMD
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=False
    )
    
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    # Model
    print("\n🧠 Creating model...")
    model = SimpleEEGModel(
        n_channels=129,
        d_model=CONFIG['d_model'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        num_classes=2
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-5)
    
    # Cosine scheduler with warmup
    warmup_epochs = 3
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (CONFIG['epochs'] - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    print("\n" + "="*80)
    print("🔥 TRAINING")
    print("="*80)
    
    best_val_acc = 0
    
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print results
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': CONFIG
            }, checkpoint_dir / "amd_rx5600xt_best.pth")
            
            print(f"  💾 New best model saved! Val Acc: {val_acc:.2f}%")
        
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print("🏁 Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("="*80)

if __name__ == "__main__":
    main()
