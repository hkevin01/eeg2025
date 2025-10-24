"""
Challenge 2 Training with HDF5 Data - Overnight Run
Uses preprocessed HDF5 windows for fast, memory-efficient training.
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
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / 'src'))

from utils.hdf5_dataset import HDF5Dataset
from braindecode.models import EEGNeX
from utils.gpu_utils import setup_device

# Configuration
CONFIG = {
    'hdf5_files': [
        PROJECT_ROOT / 'data/cached/challenge2_R1_windows.h5',
        PROJECT_ROOT / 'data/cached/challenge2_R2_windows.h5',
    ],
    'batch_size': 32,
    'num_workers': 4,
    'epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'device': 'auto',  # Will auto-detect GPU
    'save_dir': PROJECT_ROOT / 'outputs/challenge2',
    'checkpoint_freq': 10,  # Save every 10 epochs
}

def calculate_nrmse(y_true, y_pred):
    """NRMSE - Competition metric for Challenge 2"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    y_range = y_true.max() - y_true.min()
    if y_range == 0:
        return 0.0
    return rmse / y_range

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for eeg, labels in pbar:
        eeg, labels = eeg.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(eeg)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)

def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for eeg, labels in tqdm(loader, desc='Validating', leave=False):
            eeg, labels = eeg.to(device), labels.to(device)
            outputs = model(eeg)
            loss = nn.MSELoss()(outputs, labels)
            
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    val_loss = total_loss / len(loader)
    nrmse = calculate_nrmse(all_labels, all_preds)
    
    return val_loss, nrmse

def main():
    print("="*80)
    print("🎯 Challenge 2: Externalizing Factor Prediction")
    print("   Training with HDF5 Data - Overnight Run")
    print("="*80)
    
    # Setup device
    device, gpu_config = setup_device(optimize=True)
    print(f"\nUsing device: {device}")
    
    # Load HDF5 dataset
    print("\n📁 Loading HDF5 Dataset...")
    dataset = HDF5Dataset(CONFIG['hdf5_files'])
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val:   {len(val_dataset):,} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=(device.type == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=(device.type == 'cuda')
    )
    
    # Create model - Standard EEGNeX for competition compatibility
    print("\n🏗️  Creating model...")
    model = EEGNeX(
        n_chans=129,
        n_outputs=1,
        n_times=400,  # 4 seconds @ 100 Hz (from HDF5)
        sfreq=100
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model: EEGNeX")
    print(f"   Parameters: {n_params:,}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    
    # Create save directory
    CONFIG['save_dir'].mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\n🚀 Starting training for {CONFIG['epochs']} epochs...")
    print("="*80)
    
    best_nrmse = float('inf')
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_nrmse = evaluate(model, val_loader, device)
        
        # Update scheduler
        scheduler.step(val_nrmse)
        
        epoch_time = time.time() - start_time
        
        # Print progress
        print(f"Epoch {epoch:3d}/{CONFIG['epochs']} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Val NRMSE:  {val_nrmse:.6f} {'🎉 NEW BEST!' if val_nrmse < best_nrmse else ''}")
        
        # Save best model
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            checkpoint_path = CONFIG['save_dir'] / 'challenge2_hdf5_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'nrmse': best_nrmse,
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f"  💾 Saved best model (NRMSE: {best_nrmse:.6f})")
        
        # Periodic checkpoint
        if epoch % CONFIG['checkpoint_freq'] == 0:
            checkpoint_path = CONFIG['save_dir'] / f'challenge2_hdf5_epoch{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'nrmse': val_nrmse,
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f"  💾 Saved checkpoint at epoch {epoch}")
        
        print()
    
    print("="*80)
    print("✅ Training complete!")
    print(f"   Best NRMSE: {best_nrmse:.6f}")
    print(f"   Best model: {CONFIG['save_dir']}/challenge2_hdf5_best.pt")
    print("="*80)
    
    # Close dataset
    dataset.close()

if __name__ == '__main__':
    main()
