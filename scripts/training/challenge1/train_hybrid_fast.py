"""
Train hybrid model (CNN + neuroscience features) with fast pre-computed features.
This tests whether explicit neuroscience features improve over baseline.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from tqdm import tqdm
import time

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT / 'src'))

from models.hybrid_cnn import HybridNeuroModel


class FastHDF5Dataset(Dataset):
    """Fast HDF5 dataset that loads pre-computed features."""
    
    def __init__(self, hdf5_paths, use_features=True):
        self.hdf5_paths = hdf5_paths
        self.use_features = use_features
        
        # Get total length and create index mapping
        self.length = 0
        self.file_indices = []
        
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as f:
                n_samples = f['eeg'].shape[0]
                
                # Check if features exist
                if use_features and 'neuro_features' not in f:
                    raise ValueError(f"neuro_features not found in {hdf5_path}. Run preprocessing first!")
                
                self.file_indices.extend([(hdf5_path, i) for i in range(n_samples)])
                self.length += n_samples
        
        print(f"Dataset: {self.length} windows from {len(hdf5_paths)} files")
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        hdf5_path, file_idx = self.file_indices[idx]
        
        with h5py.File(hdf5_path, 'r') as f:
            eeg = f['eeg'][file_idx]
            rt = f['reaction_times'][file_idx]
            
            # Load pre-computed features
            if self.use_features:
                features = f['neuro_features'][file_idx]
            else:
                features = np.zeros(6, dtype=np.float32)
        
        return torch.from_numpy(eeg), torch.from_numpy(features), torch.tensor(rt, dtype=torch.float32)


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for eeg, features, rt in tqdm(train_loader, desc="Training"):
        eeg = eeg.to(device)
        features = features.to(device)
        rt = rt.to(device)
        
        pred_rt = model(eeg, features).squeeze()
        loss = criterion(pred_rt, rt)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for eeg, features, rt in tqdm(val_loader, desc="Validation"):
            eeg = eeg.to(device)
            features = features.to(device)
            rt = rt.to(device)
            
            pred_rt = model(eeg, features).squeeze()
            loss = criterion(pred_rt, rt)
            total_loss += loss.item()
            
            predictions.extend(pred_rt.cpu().numpy())
            targets.extend(rt.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    
    # Calculate NRMSE
    predictions = np.array(predictions)
    targets = np.array(targets)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    nrmse = rmse / (targets.max() - targets.min())
    
    return avg_loss, nrmse


def main():
    print("="*80)
    print("🧠 HYBRID MODEL TRAINING (CNN + Neuroscience Features)")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    data_dir = PROJECT_ROOT / 'data' / 'cached'
    hdf5_files = sorted(data_dir.glob('challenge1_R*.h5'))[:3]  # R1, R2, R3 for training
    
    print(f"\nLoading data from {len(hdf5_files)} files...")
    
    # Check if features exist
    print("\nVerifying pre-computed features...")
    for fpath in hdf5_files:
        with h5py.File(fpath, 'r') as f:
            if 'neuro_features' in f:
                print(f"  ✅ {fpath.name}: {f['neuro_features'].shape}")
            else:
                print(f"  ❌ {fpath.name}: No features! Run preprocessing first.")
                return
    
    full_dataset = FastHDF5Dataset(hdf5_files, use_features=True)
    
    # Split train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Model (HYBRID - with features)
    model = HybridNeuroModel(
        input_channels=129,
        use_neuro_features=True,  # HYBRID: use features!
        dropout_rate=0.4
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")
    print("Mode: HYBRID (CNN + 6 neuroscience features)")
    print("  Features: P300, motor prep, N200, alpha suppression")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    n_epochs = 50
    best_nrmse = float('inf')
    patience = 10
    patience_counter = 0
    
    print(f"\n{'='*80}")
    print("TRAINING START")
    print(f"{'='*80}\n")
    
    for epoch in range(n_epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_nrmse = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch+1}/{n_epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val NRMSE: {val_nrmse:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': val_nrmse,
                'val_loss': val_loss
            }
            torch.save(checkpoint, PROJECT_ROOT / 'checkpoints' / 'hybrid_best.pth')
            print(f"  ✅ New best! NRMSE: {val_nrmse:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping triggered (patience={patience})")
                break
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"\n✅ Best validation NRMSE: {best_nrmse:.4f}")
    print(f"📁 Model saved to: checkpoints/hybrid_best.pth")
    
    # Load and compare with baseline
    baseline_path = PROJECT_ROOT / 'checkpoints' / 'baseline_best.pth'
    if baseline_path.exists():
        baseline_ckpt = torch.load(baseline_path)
        baseline_nrmse = baseline_ckpt['val_nrmse']
        
        print(f"\n{'='*80}")
        print("COMPARISON WITH BASELINE")
        print(f"{'='*80}")
        print(f"Baseline NRMSE: {baseline_nrmse:.4f}")
        print(f"Hybrid NRMSE:   {best_nrmse:.4f}")
        
        if best_nrmse < baseline_nrmse:
            improvement = ((baseline_nrmse - best_nrmse) / baseline_nrmse) * 100
            print(f"\n🎉 HYBRID IS BETTER by {improvement:.1f}%!")
            print("   Neuroscience features are helping!")
        else:
            degradation = ((best_nrmse - baseline_nrmse) / baseline_nrmse) * 100
            print(f"\n⚠️  Baseline is better by {degradation:.1f}%")
            print("   Features may not be providing value for this task")
    

if __name__ == "__main__":
    main()
