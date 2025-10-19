"""
Training script for Hybrid Neuroscience + CNN Model.

Features:
- Uses HDF5 memory-mapped data (2-4GB RAM)
- Strong regularization to prevent overfitting
- Early stopping with patience
- Validation monitoring
- Compares to baseline (0.26 NRMSE)

Anti-overfitting measures:
- Dropout 0.4
- Weight decay 1e-4
- Early stopping (patience=10)
- Monitor train/val gap
- Conservative learning rate
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import time

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT / 'src'))

from utils.hdf5_dataset import HDF5Dataset
from models.hybrid_cnn import HybridNeuroModel

# Configuration
CONFIG = {
    'hdf5_files': [
        PROJECT_ROOT / 'data/cached/challenge1_R1_windows.h5',
        PROJECT_ROOT / 'data/cached/challenge1_R2_windows.h5',
        PROJECT_ROOT / 'data/cached/challenge1_R3_windows.h5',
    ],
    'batch_size': 32,
    'num_workers': 4,
    'epochs': 50,
    'learning_rate': 1e-4,  # Conservative LR
    'weight_decay': 1e-4,  # L2 regularization
    'dropout': 0.4,  # Strong dropout
    'early_stopping_patience': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': PROJECT_ROOT / 'checkpoints',
    'use_neuro_features': True,  # Set to False for CNN-only baseline
}


def nrmse(y_true, y_pred):
    """Normalized Root Mean Squared Error (competition metric)."""
    mse = ((y_true - y_pred) ** 2).mean()
    rmse = np.sqrt(mse)
    std = y_true.std()
    return rmse / (std + 1e-8)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Training')
    for eeg, labels in pbar:
        eeg, labels = eeg.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(eeg)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy().flatten())
        all_labels.extend(labels.detach().cpu().numpy().flatten())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(loader)
    train_nrmse = nrmse(np.array(all_labels), np.array(all_preds))
    
    return avg_loss, train_nrmse


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for eeg, labels in tqdm(loader, desc='Validating'):
            eeg, labels = eeg.to(device), labels.to(device)
            
            outputs = model(eeg)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(loader)
    val_nrmse = nrmse(np.array(all_labels), np.array(all_preds))
    
    return avg_loss, val_nrmse


def main():
    print("="*80)
    print("🧠 HYBRID NEUROSCIENCE + CNN TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        if key != 'hdf5_files':
            print(f"  {key}: {value}")
    print(f"  hdf5_files: {len(CONFIG['hdf5_files'])} files")
    
    # Create save directory
    CONFIG['save_dir'].mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"\n{'='*80}")
    print("📁 Loading HDF5 Dataset...")
    print(f"{'='*80}")
    
    dataset = HDF5Dataset(CONFIG['hdf5_files'])
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    
    # Create model
    print(f"\n{'='*80}")
    print("🏗️  Creating Model...")
    print(f"{'='*80}")
    
    model = HybridNeuroModel(
        num_channels=129,
        seq_length=200,
        dropout=CONFIG['dropout'],
        use_neuro_features=CONFIG['use_neuro_features']
    )
    model = model.to(CONFIG['device'])
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: HybridNeuroModel")
    print(f"Parameters: {n_params:,}")
    print(f"Use Neuro Features: {CONFIG['use_neuro_features']}")
    print(f"Device: {CONFIG['device']}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training loop
    print(f"\n{'='*80}")
    print("🚀 Starting Training...")
    print(f"{'='*80}")
    
    best_val_nrmse = float('inf')
    patience_counter = 0
    train_history = []
    val_history = []
    
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        print("-" * 80)
        
        # Train
        train_loss, train_nrmse = train_epoch(
            model, train_loader, criterion, optimizer, CONFIG['device']
        )
        
        # Validate
        val_loss, val_nrmse = validate(
            model, val_loader, criterion, CONFIG['device']
        )
        
        # Update scheduler
        scheduler.step(val_nrmse)
        
        # Save history
        train_history.append(train_nrmse)
        val_history.append(val_nrmse)
        
        # Print metrics
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train NRMSE: {train_nrmse:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val NRMSE:   {val_nrmse:.4f}")
        print(f"  Train/Val Gap: {abs(train_nrmse - val_nrmse):.4f}")
        
        # Check for improvement
        if val_nrmse < best_val_nrmse:
            best_val_nrmse = val_nrmse
            patience_counter = 0
            
            # Save best model
            save_path = CONFIG['save_dir'] / 'hybrid_neuro_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': val_nrmse,
                'config': CONFIG,
            }, save_path)
            print(f"  ✅ New best model saved! (Val NRMSE: {val_nrmse:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{CONFIG['early_stopping_patience']})")
            
        # Early stopping
        if patience_counter >= CONFIG['early_stopping_patience']:
            print(f"\n⚠️  Early stopping triggered!")
            break
        
        # Warning if overfitting
        if abs(train_nrmse - val_nrmse) > 0.1:
            print(f"  ⚠️  WARNING: Large train/val gap ({abs(train_nrmse - val_nrmse):.4f}) - possible overfitting!")
    
    # Final results
    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"\nBest Validation NRMSE: {best_val_nrmse:.4f}")
    print(f"Baseline (CNN only):    0.26")
    
    if best_val_nrmse < 0.26:
        improvement = ((0.26 - best_val_nrmse) / 0.26) * 100
        print(f"🎉 Improvement: {improvement:.1f}% better than baseline!")
    else:
        decline = ((best_val_nrmse - 0.26) / 0.26) * 100
        print(f"⚠️  Decline: {decline:.1f}% worse than baseline")
        print(f"   Recommendation: Use baseline model instead")
    
    print(f"\nModel saved to: {CONFIG['save_dir'] / 'hybrid_neuro_best.pth'}")
    
    # Save training history
    history_path = CONFIG['save_dir'] / 'hybrid_training_history.npz'
    np.savez(
        history_path,
        train_nrmse=train_history,
        val_nrmse=val_history,
        best_val_nrmse=best_val_nrmse
    )
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()
