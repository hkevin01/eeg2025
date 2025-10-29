#!/usr/bin/env python3
"""
Train Challenge 1 models with SUBJECT-AWARE validation.
This ensures zero subject overlap between train and validation sets.
"""

import os
# Force CPU mode to avoid ROCm/CUDA conflicts
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import h5py
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import json

# Force CPU device
DEVICE = torch.device('cpu')
print(f"🖥️  Using device: {DEVICE}")

# ======================== MODEL DEFINITION ========================

class CompactCNN(nn.Module):
    """Compact CNN for EEG regression - 75K parameters."""
    def __init__(self, in_channels=129, time_steps=200):
        super().__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze()

# ======================== DATASET ========================

class EEGDataset(Dataset):
    """Dataset that loads from H5 cache with subject IDs."""
    def __init__(self, cache_files):
        self.eeg_data = []
        self.labels = []
        self.subject_ids = []
        
        print(f"📂 Loading {len(cache_files)} cache files...")
        
        for cache_file in cache_files:
            print(f"   Loading {cache_file.name}...")
            with h5py.File(cache_file, 'r') as f:
                eeg = f['eeg'][:]
                labels = f['labels'][:]
                subject_ids = f['subject_ids'][:]
                
                self.eeg_data.append(eeg)
                self.labels.append(labels)
                self.subject_ids.append(subject_ids)
                
                print(f"      {len(eeg)} samples, {len(np.unique(subject_ids))} subjects")
        
        # Concatenate
        self.eeg_data = np.concatenate(self.eeg_data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.subject_ids = np.concatenate(self.subject_ids, axis=0)
        
        print(f"✅ Total: {len(self.eeg_data)} samples, {len(np.unique(self.subject_ids))} unique subjects")
    
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        eeg = torch.FloatTensor(self.eeg_data[idx])
        label = torch.FloatTensor([self.labels[idx]])
        return eeg, label

# ======================== SUBJECT-AWARE SPLIT ========================

def create_subject_aware_split(dataset, val_ratio=0.2, seed=42):
    """
    Create train/val split with ZERO subject overlap.
    This is THE KEY to reliable validation!
    """
    print(f"\n🔍 Creating subject-aware split (val_ratio={val_ratio})...")
    
    # Get unique subjects
    unique_subjects = np.unique(dataset.subject_ids)
    n_subjects = len(unique_subjects)
    print(f"   Total unique subjects: {n_subjects}")
    
    # Split subjects
    np.random.seed(seed)
    shuffled_subjects = np.random.permutation(unique_subjects)
    
    n_val_subjects = int(n_subjects * val_ratio)
    val_subjects = set(shuffled_subjects[:n_val_subjects])
    train_subjects = set(shuffled_subjects[n_val_subjects:])
    
    print(f"   Train subjects: {len(train_subjects)}")
    print(f"   Val subjects: {len(val_subjects)}")
    
    # Create indices
    train_indices = []
    val_indices = []
    
    for idx in range(len(dataset)):
        subject_id = dataset.subject_ids[idx]
        if subject_id in val_subjects:
            val_indices.append(idx)
        else:
            train_indices.append(idx)
    
    print(f"   Train samples: {len(train_indices)}")
    print(f"   Val samples: {len(val_indices)}")
    print(f"   ✅ Zero subject overlap guaranteed!")
    
    return train_indices, val_indices

# ======================== TRAINING ========================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for eeg, labels in train_loader:
        eeg, labels = eeg.to(device), labels.squeeze().to(device)
        
        optimizer.zero_grad()
        outputs = model(eeg)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for eeg, labels in val_loader:
            eeg, labels = eeg.to(device), labels.squeeze().to(device)
            outputs = model(eeg)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate NRMSE
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    mse = np.mean((all_preds - all_labels) ** 2)
    rmse = np.sqrt(mse)
    nrmse = rmse / (all_labels.max() - all_labels.min())
    
    return total_loss / len(val_loader), nrmse

# ======================== MAIN TRAINING FUNCTION ========================

def train_model(rsets, model_name, val_ratio=0.2, epochs=30, batch_size=128, lr=0.001):
    """
    Train a model with subject-aware validation.
    
    Args:
        rsets: List of R-set names (e.g., ['R1', 'R2', 'R3'])
        model_name: Name for saving (e.g., 'r123_val4')
        val_ratio: Validation split ratio
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
    """
    print("=" * 70)
    print(f"🚀 Training: {model_name}")
    print(f"   R-sets: {rsets}")
    print(f"   Val ratio: {val_ratio}")
    print("=" * 70)
    
    # Setup - use global DEVICE
    device = DEVICE
    print(f"💻 Device: {device}")
    
    # Load data
    cache_dir = Path('data/cached')
    cache_files = [cache_dir / f'challenge1_{rset}_windows.h5' for rset in rsets]
    dataset = EEGDataset(cache_files)
    
    # Subject-aware split
    train_indices, val_indices = create_subject_aware_split(dataset, val_ratio=val_ratio)
    
    # Create data loaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=4
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_indices),
        num_workers=4
    )
    
    # Create model
    model = CompactCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    best_val_nrmse = float('inf')
    best_epoch = 0
    history = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_nrmse = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step(val_nrmse)
        
        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_nrmse': val_nrmse,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val NRMSE: {val_nrmse:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_nrmse < best_val_nrmse:
            best_val_nrmse = val_nrmse
            best_epoch = epoch + 1
            
            # Save checkpoint
            checkpoint_dir = Path('checkpoints/c1_subject_aware')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': val_nrmse,
                'rsets': rsets,
                'val_ratio': val_ratio,
            }, checkpoint_dir / f'{model_name}_best.pth')
            
            print(f"   💾 Saved best model (NRMSE: {val_nrmse:.4f})")
    
    total_time = time.time() - start_time
    
    # Save final results
    # Convert numpy types to Python types for JSON serialization
    # Extract history as lists from list of dicts
    train_losses = [float(h['train_loss']) for h in history]
    val_losses = [float(h['val_loss']) for h in history]
    val_nrmses = [float(h['val_nrmse']) for h in history]
    
    results = {
        'model_name': model_name,
        'rsets': rsets,
        'val_ratio': float(val_ratio),
        'best_val_nrmse': float(best_val_nrmse),
        'best_epoch': int(best_epoch),
        'total_epochs': int(epochs),
        'total_time': float(total_time),
        'history': {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'val_nrmse': val_nrmses,
        },
        'train_samples': int(len(train_indices)),
        'val_samples': int(len(val_indices)),
        'train_subjects': int(len(set(dataset.subject_ids[i] for i in train_indices))),
        'val_subjects': int(len(set(dataset.subject_ids[i] for i in val_indices))),
    }
    
    results_file = checkpoint_dir / f'{model_name}_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"✅ Training complete!")
    print(f"   Best Val NRMSE: {best_val_nrmse:.4f} (epoch {best_epoch})")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Results saved to: {results_file}")
    print("=" * 70)
    
    return results

# ======================== MAIN ========================

if __name__ == '__main__':
    print("\n🧠 EEG Challenge 1 - Subject-Aware Training")
    print("=" * 70)
    print("This training uses SUBJECT-AWARE validation:")
    print("  ✅ Zero subject overlap between train and validation")
    print("  ✅ Should give reliable validation metrics")
    print("  ✅ Can correlate with test scores")
    print("=" * 70)
    
    # Train 3 different models for Phase 1 correlation test
    
    print("\n🔬 Phase 1: Training 3 models to test validation correlation")
    
    # Model 1: R1-R3 (original approach but with subject-aware split)
    print("\n" + "=" * 70)
    print("Model 1: R1-R3 with subject-aware validation")
    results1 = train_model(
        rsets=['R1', 'R2', 'R3'],
        model_name='r123_subject_aware',
        val_ratio=0.2,
        epochs=30,
        batch_size=128,
        lr=0.001
    )
    
    # Model 2: R1-R2-R4 (different combination)
    print("\n" + "=" * 70)
    print("Model 2: R1-R2-R4 with subject-aware validation")
    results2 = train_model(
        rsets=['R1', 'R2', 'R4'],
        model_name='r124_subject_aware',
        val_ratio=0.2,
        epochs=30,
        batch_size=128,
        lr=0.001
    )
    
    # Model 3: ALL R-sets (but with proper validation this time)
    print("\n" + "=" * 70)
    print("Model 3: ALL R-sets with subject-aware validation")
    results3 = train_model(
        rsets=['R1', 'R2', 'R3', 'R4'],
        model_name='all_rsets_subject_aware',
        val_ratio=0.2,
        epochs=30,
        batch_size=128,
        lr=0.001
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("🎉 PHASE 1 COMPLETE - 3 Models Trained")
    print("=" * 70)
    print("\n📊 Summary:")
    print(f"Model 1 (R1-R3):  Val NRMSE = {results1['best_val_nrmse']:.4f}")
    print(f"Model 2 (R1-R2-R4): Val NRMSE = {results2['best_val_nrmse']:.4f}")
    print(f"Model 3 (ALL):    Val NRMSE = {results3['best_val_nrmse']:.4f}")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Create submissions for all 3 models")
    print("2. Submit to competition")
    print("3. Check correlation: Val NRMSE vs Test C1")
    print("4. If correlation > 0.7, proceed to Phase 2 optimization")
    print("5. If correlation < 0.7, validation still unreliable")
    print("\n" + "=" * 70)
