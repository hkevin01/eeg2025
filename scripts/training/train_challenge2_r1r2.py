#!/usr/bin/env python3
"""
Challenge 2 Training - R1 & R2 Only
====================================
Start training with available cache (R1, R2)
While R3-R5 are still being created.
"""
import os
import sys
import time
import warnings
import h5py
import numpy as np
import sqlite3
from pathlib import Path
from datetime import datetime
import json

warnings.filterwarnings('ignore')

# CPU for stability
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import torch.nn as nn
from torch import optim
from torch.nn.functional import l1_loss
from torch.utils.data import Dataset, DataLoader, random_split
from braindecode.models import EEGNeX

print("="*80)
print("🚀 CHALLENGE 2: Training with R1 & R2 (Start Fast!)")
print("="*80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

CACHE_DIR = Path("data/cached")
DB_FILE = Path("data/metadata.db")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

BATCH_SIZE = 64
MAX_EPOCHS = 20
LEARNING_RATE = 0.002
PATIENCE = 5

SFREQ = 100
CROP_SIZE = 2.0  # Random crop to 2 seconds

# ============================================================================
# HDF5 DATASET CLASS
# ============================================================================

class CachedEEGDataset(Dataset):
    """Fast dataset that loads from HDF5 cache."""
    
    def __init__(self, cache_files, crop_size_samples=200, augment=True):
        self.crop_size = crop_size_samples
        self.augment = augment
        
        print(f"Loading from {len(cache_files)} cache files...")
        all_data = []
        all_targets = []
        
        for cache_file in cache_files:
            print(f"  Loading {cache_file.name}...")
            with h5py.File(cache_file, 'r') as f:
                data = f['data'][:]
                targets = f['targets'][:]
                all_data.append(data)
                all_targets.append(targets)
                print(f"    Loaded {len(data)} windows")
        
        self.data = np.concatenate(all_data, axis=0)
        self.targets = np.concatenate(all_targets, axis=0)
        
        print(f"Total dataset: {len(self.data)} windows")
        print(f"  Shape: {self.data.shape}")
        print(f"  Memory: {self.data.nbytes / 1024**2:.1f} MB")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.targets[idx]
        
        # Random crop if augmenting
        if self.augment and X.shape[1] > self.crop_size:
            start = np.random.randint(0, X.shape[1] - self.crop_size + 1)
            X = X[:, start:start + self.crop_size]
        elif X.shape[1] > self.crop_size:
            start = (X.shape[1] - self.crop_size) // 2
            X = X[:, start:start + self.crop_size]
        
        return torch.FloatTensor(X), torch.FloatTensor([y])

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def create_training_run(challenge, model_name, config):
    """Register a new training run in the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO training_runs (challenge, model_name, start_time, status, config)
        VALUES (?, ?, ?, ?, ?)
    ''', (challenge, model_name, datetime.now().isoformat(), 'running', json.dumps(config)))
    
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return run_id

def log_epoch(run_id, epoch, train_loss, val_loss, lr, duration):
    """Log epoch metrics to database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO epoch_history (run_id, epoch, train_loss, val_loss, learning_rate, duration_seconds, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (run_id, epoch, train_loss, val_loss, lr, duration, datetime.now().isoformat()))
    
    conn.commit()
    conn.close()

def save_checkpoint_info(run_id, epoch, val_loss, file_path, is_best=False):
    """Register checkpoint in database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO model_checkpoints (run_id, epoch, val_loss, file_path, is_best, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (run_id, epoch, val_loss, str(file_path), is_best, datetime.now().isoformat()))
    
    conn.commit()
    conn.close()

def update_run_status(run_id, status, best_val_loss=None):
    """Update training run status."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    if best_val_loss is not None:
        cursor.execute('''
            UPDATE training_runs 
            SET status = ?, end_time = ?, best_val_loss = ?
            WHERE id = ?
        ''', (status, datetime.now().isoformat(), best_val_loss, run_id))
    else:
        cursor.execute('''
            UPDATE training_runs 
            SET status = ?, end_time = ?
            WHERE id = ?
        ''', (status, datetime.now().isoformat(), run_id))
    
    conn.commit()
    conn.close()

# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    # Check cache files
    cache_files = [
        CACHE_DIR / "challenge2_R1_windows.h5",
        CACHE_DIR / "challenge2_R2_windows.h5",
    ]
    
    missing_files = [f for f in cache_files if not f.exists()]
    if missing_files:
        print("❌ Missing cache files:")
        for f in missing_files:
            print(f"   {f}")
        return
    
    print("✅ Cache files found: R1, R2")
    print()
    
    # Register training run
    config = {
        'batch_size': BATCH_SIZE,
        'max_epochs': MAX_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'patience': PATIENCE,
        'crop_size': CROP_SIZE,
        'model': 'EEGNeX',
        'optimizer': 'Adamax',
        'loss': 'L1',
        'note': 'Training with R1+R2 only (R3-R5 still creating)',
    }
    
    run_id = create_training_run(challenge=2, model_name='EEGNeX_R1R2', config=config)
    print(f"📊 Training run registered: ID = {run_id}")
    print()
    
    # Load data
    print("="*80)
    print("PHASE 1: DATA LOADING")
    print("="*80)
    print()
    
    load_start = time.time()
    
    print("Loading combined dataset (R1 + R2)...")
    full_dataset = CachedEEGDataset(cache_files, crop_size_samples=int(CROP_SIZE * SFREQ), augment=True)
    
    # Split 80/20 for train/val
    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
    
    print(f"\n✅ Data loaded in {time.time() - load_start:.1f}s")
    print(f"   Train: {len(train_dataset)} windows")
    print(f"   Val:   {len(val_dataset)} windows")
    print()
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Create model
    print("="*80)
    print("PHASE 2: MODEL CREATION")
    print("="*80)
    print()
    
    model = EEGNeX(
        n_outputs=1,
        n_chans=129,
        n_times=int(CROP_SIZE * SFREQ),
        sfreq=SFREQ,
    )
    
    print(f"Model: EEGNeX")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    optimizer = optim.Adamax(model.parameters(), lr=LEARNING_RATE)
    criterion = l1_loss
    
    # Training loop
    print("="*80)
    print("PHASE 3: TRAINING")
    print("="*80)
    print()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        
        # Train
        model.train()
        train_losses = []
        
        for batch_idx, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward
            y_pred = model(X)
            loss = criterion(y_pred, y)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{MAX_EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.6f}")
        
        # Validate
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for X, y in val_loader:
                y_pred = model(X)
                loss = criterion(y_pred, y)
                val_losses.append(loss.item())
        
        # Metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        epoch_time = time.time() - epoch_start
        
        # Log
        log_epoch(run_id, epoch + 1, train_loss, val_loss, LEARNING_RATE, epoch_time)
        
        print(f"\n📊 Epoch {epoch+1}/{MAX_EPOCHS}")
        print(f"   Train Loss: {train_loss:.6f}")
        print(f"   Val Loss:   {val_loss:.6f}")
        print(f"   Time:       {epoch_time:.1f}s")
        
        # Save checkpoint
        checkpoint_path = CHECKPOINT_DIR / f"challenge2_r1r2_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        
        # Check if best
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_path = CHECKPOINT_DIR / "challenge2_r1r2_best.pth"
            torch.save(model.state_dict(), best_path)
            print(f"   ⭐ New best model! Saved to {best_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"   Patience: {patience_counter}/{PATIENCE}")
        
        save_checkpoint_info(run_id, epoch + 1, val_loss, checkpoint_path, is_best)
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping triggered (patience={PATIENCE})")
            break
        
        print()
    
    # Complete
    update_run_status(run_id, 'completed', best_val_loss)
    
    print("="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Best Model:    checkpoints/challenge2_r1r2_best.pth")
    print()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
