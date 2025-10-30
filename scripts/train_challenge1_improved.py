#!/usr/bin/env python3
"""
Challenge 1 Training with Anti-Overfitting Measures
Goal: Beat untrained baseline (1.0015) through careful regularization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import yaml
import numpy as np
from pathlib import Path
import sys

sys.path.append('src')

# Key insight: Train carefully to avoid overfitting!
print("="*70)
print("🧠 Challenge 1: Anti-Overfitting Training")
print("="*70)
print("Goal: Beat untrained baseline (1.0015)")
print("Strategy: Strong regularization + early stopping")
print("="*70)
print()


class ImprovedCompactCNN(nn.Module):
    """CompactCNN with stronger regularization"""
    def __init__(self, dropout_rates=[0.5, 0.6, 0.7], fc_dropout=[0.6, 0.5]):
        super().__init__()
        
        self.features = nn.Sequential(
            # Conv1: 129 -> 32
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rates[0]),  # Stronger!
            
            # Conv2: 32 -> 64
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1]),
            
            # Conv3: 64 -> 128
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rates[2]),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(fc_dropout[0]),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(fc_dropout[1]),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.regressor(features)
        return output


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=5, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.mode == 'min':
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        return self.early_stop


def train_improved_model():
    """Train with anti-overfitting measures"""
    
    # Load config
    with open('config/challenge1_anti_overfit.yaml') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Create model with strong regularization
    model = ImprovedCompactCNN(
        dropout_rates=config['model']['architecture']['dropout'],
        fc_dropout=config['model']['architecture']['fc_dropout']
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Dropout rates: {config['model']['architecture']['dropout']}")
    print(f"FC dropout: {config['model']['architecture']['fc_dropout']}\n")
    
    # Optimizer with strong weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Weight decay: {config['training']['weight_decay']} (strong L2!)\n")
    
    # Cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['max_epochs'],
        eta_min=config['training']['scheduler']['eta_min']
    )
    
    # Loss with label smoothing
    criterion = nn.SmoothL1Loss()  # Robust to outliers
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping']['patience'],
        min_delta=config['training']['early_stopping']['min_delta']
    )
    
    print("🎯 Training Configuration:")
    print(f"  Max epochs: {config['training']['max_epochs']} (SHORT!)")
    print(f"  Early stopping patience: {config['training']['early_stopping']['patience']}")
    print(f"  Mixup alpha: {config['mixup']['alpha']}")
    print(f"  Gradient clipping: {config['training']['clip_grad_norm']}")
    print()
    
    print("="*70)
    print("Starting training...")
    print("="*70)
    
    # TODO: Load actual data here
    # For now, just save the model architecture
    
    save_path = Path('checkpoints/challenge1_improved_anti_overfit.pth')
    save_path.parent.mkdir(exist_ok=True, parents=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'n_params': n_params
    }, save_path)
    
    print(f"\n✅ Model saved to: {save_path}")
    print("\nNext steps:")
    print("1. Integrate with actual data loader")
    print("2. Train for max 15 epochs")
    print("3. Monitor validation NRMSE carefully")
    print("4. Stop early if not improving")
    print("5. Compare with untrained baseline (1.0015)")


if __name__ == '__main__':
    train_improved_model()
