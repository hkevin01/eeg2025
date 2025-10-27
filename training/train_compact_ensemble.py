"""
Train 3x CompactCNN with Improvements
Path B Strategy: Better training > Better architecture

Improvements:
- All releases (R1+R2+R3+R4) for training
- Data augmentation (time jitter, noise, channel dropout)
- R5 as validation hold-out
- 3 models with different seeds (42, 123, 456)
- Ensemble submission at the end
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import h5py
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime


# ============================================================================
# COMPACT CNN ARCHITECTURE (Proven 1.0015)
# ============================================================================

class CompactResponseTimeCNN(nn.Module):
    """Simple 3-layer CNN - proven to work with score 1.0015"""
    
    def __init__(self, n_channels=129, sequence_length=200):
        super().__init__()
        
        # Feature extraction - 3 conv layers with progressive downsampling
        self.features = nn.Sequential(
            # Conv 1: 129 → 32
            nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            # Conv 2: 32 → 64
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            # Conv 3: 64 → 128
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            # Global average pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels=129, time=200)
        Returns:
            predictions: (batch, 1)
        """
        features = self.features(x)
        predictions = self.regressor(features)
        return predictions


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class EEGAugmentation:
    """Data augmentation for EEG signals"""
    
    def __init__(self, 
                 time_jitter_ms=10,
                 noise_std=0.02,
                 channel_dropout_prob=0.05,
                 sfreq=100):
        self.time_jitter_samples = int(time_jitter_ms * sfreq / 1000)
        self.noise_std = noise_std
        self.channel_dropout_prob = channel_dropout_prob
    
    def __call__(self, x):
        """
        Args:
            x: (channels, time) tensor
        Returns:
            Augmented (channels, time) tensor
        """
        # Time jitter (random circular shift)
        if self.time_jitter_samples > 0:
            shift = np.random.randint(-self.time_jitter_samples, 
                                     self.time_jitter_samples + 1)
            x = torch.roll(x, shifts=shift, dims=-1)
        
        # Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        # Channel dropout
        if self.channel_dropout_prob > 0:
            n_channels = x.shape[0]
            n_drop = int(n_channels * self.channel_dropout_prob)
            if n_drop > 0:
                drop_indices = np.random.choice(n_channels, n_drop, replace=False)
                x[drop_indices] = 0.0
        
        return x


# ============================================================================
# DATASET
# ============================================================================

class EEGDataset(Dataset):
    """Dataset for EEG windows - loads from cached H5 files"""
    
    def __init__(self, release_numbers, cache_dir='data/cached', 
                 challenge=1, augmentation=None):
        """
        Args:
            release_numbers: List of release numbers to load (e.g., [1,2,3,4])
            cache_dir: Directory containing cached H5 files
            challenge: Challenge number (1 or 2)
            augmentation: Optional augmentation function
        """
        self.augmentation = augmentation
        
        # Load all releases
        all_eeg = []
        all_labels = []
        
        for release_num in release_numbers:
            cache_file = f"{cache_dir}/challenge{challenge}_R{release_num}_windows.h5"
            
            if not os.path.exists(cache_file):
                print(f"⚠️  Warning: {cache_file} not found, skipping")
                continue
            
            with h5py.File(cache_file, 'r') as f:
                eeg = f['eeg'][:]  # (N, 129, 200)
                labels = f['labels'][:]  # (N,)
                
                all_eeg.append(eeg)
                all_labels.append(labels)
                
                print(f"✅ Loaded R{release_num}: {len(labels)} samples")
        
        # Concatenate all releases
        self.eeg_data = torch.from_numpy(np.concatenate(all_eeg, axis=0)).float()
        self.labels = torch.from_numpy(np.concatenate(all_labels, axis=0)).float()
        
        print(f"\nTotal dataset: {len(self.labels)} samples")
        print(f"  EEG shape: {self.eeg_data.shape}")
        print(f"  Labels: min={self.labels.min():.3f}, max={self.labels.max():.3f}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.eeg_data[idx]
        y = self.labels[idx]
        
        if self.augmentation is not None:
            x = self.augmentation(x)
        
        return x, y
        
        if self.transform is not None:
            x = self.transform(x)
        
        return x, y


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data).squeeze(-1)  # (batch, 1) -> (batch,)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Validating"):
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze(-1)  # (batch, 1) -> (batch,)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            predictions.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    # Calculate metrics
    val_loss = total_loss / len(dataloader)
    
    # Pearson correlation
    predictions_flat = predictions.flatten()
    targets_flat = targets.flatten()
    pearson_r = np.corrcoef(predictions_flat, targets_flat)[0, 1]
    
    # NRMSE
    mse = np.mean((predictions_flat - targets_flat) ** 2)
    rmse = np.sqrt(mse)
    target_range = targets_flat.max() - targets_flat.min()
    nrmse = rmse / target_range if target_range > 0 else rmse
    
    return val_loss, pearson_r, nrmse


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_compact_cnn(seed, config):
    """Train single CompactCNN model with given seed"""
    
    print(f"\n{'='*60}")
    print(f"Training CompactCNN with seed {seed}")
    print(f"{'='*60}\n")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = CompactResponseTimeCNN(n_channels=129, sequence_length=200)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create datasets
    augmentation = EEGAugmentation(
        time_jitter_ms=config['time_jitter_ms'],
        noise_std=config['noise_std'],
        channel_dropout_prob=config['channel_dropout_prob']
    )
    
    # Training data: R1+R2+R3+R4 with augmentation
    train_dataset = EEGDataset(
        release_numbers=config['train_releases'],
        cache_dir=config['data_path'],
        challenge=1,
        augmentation=augmentation
    )
    
    # Validation data: Use 20% of R4 as validation
    # (since R5 doesn't exist in cached data)
    val_dataset = EEGDataset(
        release_numbers=[config['val_release']],
        cache_dir=config['data_path'],
        challenge=1,
        augmentation=None
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Loss, optimizer, scheduler
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs']
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_pearson_r = -1.0
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_pearson_r': [],
        'val_nrmse': []
    }
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print("-" * 60)
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        
        # Validate
        val_loss, pearson_r, nrmse = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_pearson_r'].append(pearson_r)
        history['val_nrmse'].append(nrmse)
        
        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Pearson r: {pearson_r:.4f}")
        print(f"  Val NRMSE: {nrmse:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if pearson_r > best_pearson_r:
            best_pearson_r = pearson_r
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_path = config['output_dir'] / f"compact_cnn_seed{seed}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'pearson_r': pearson_r,
                'nrmse': nrmse,
                'history': history,
                'config': config,
                'seed': seed
            }, checkpoint_path)
            
            print(f"  💾 Saved best model (r={pearson_r:.4f}, NRMSE={nrmse:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\n⚠️  Early stopping triggered (patience={config['patience']})")
                break
    
    print(f"\n✅ Training complete!")
    print(f"Best validation Pearson r: {best_pearson_r:.4f}")
    print(f"Best validation NRMSE: {history['val_nrmse'][history['val_pearson_r'].index(best_pearson_r)]:.4f}")
    
    return checkpoint_path, best_pearson_r


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train 3x CompactCNN with improvements')
    parser.add_argument('--data_path', type=str, default='data/cached',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='checkpoints/compact_ensemble',
                       help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    args = parser.parse_args()
    
    # Configuration
    config = {
        'data_path': args.data_path,
        'output_dir': Path(args.output_dir),
        'train_releases': [1, 2, 3],  # R1-R3 for training
        'val_release': 4,  # R4 for validation (R5 not available)
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': 1e-5,
        'patience': 10,
        'time_jitter_ms': 10,
        'noise_std': 0.02,
        'channel_dropout_prob': 0.05,
        'seeds': [42, 123, 456]
    }
    
    # Create output directory
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = config['output_dir'] / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump({k: str(v) if isinstance(v, Path) else v 
                  for k, v in config.items()}, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING 3× COMPACTCNN WITH IMPROVEMENTS")
    print("="*60)
    print("\nConfiguration:")
    print(f"  Training releases: {config['train_releases']}")
    print(f"  Validation release: {config['val_release']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Seeds: {config['seeds']}")
    print(f"\nAugmentation:")
    print(f"  Time jitter: ±{config['time_jitter_ms']}ms")
    print(f"  Noise std: {config['noise_std']}")
    print(f"  Channel dropout: {config['channel_dropout_prob']*100}%")
    print(f"\nOutput: {config['output_dir']}")
    
    # Train 3 models
    results = []
    for seed in config['seeds']:
        checkpoint_path, best_r = train_compact_cnn(seed, config)
        results.append({
            'seed': seed,
            'checkpoint': str(checkpoint_path),
            'best_pearson_r': best_r
        })
    
    # Save results summary
    summary_path = config['output_dir'] / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("ALL TRAINING COMPLETE!")
    print("="*60)
    print("\nResults:")
    for result in results:
        print(f"  Seed {result['seed']}: r={result['best_pearson_r']:.4f}")
    
    print(f"\nCheckpoints saved to: {config['output_dir']}")
    print(f"Summary saved to: {summary_path}")
    
    print("\nNext steps:")
    print("1. Create ensemble submission combining all 3 models")
    print("2. Test ensemble locally on validation set")
    print("3. Package and submit to competition")


if __name__ == '__main__':
    main()
