#!/usr/bin/env python3
"""
Challenge 2: Quick Training Script
===================================
Train externalizing prediction model efficiently.

Based on: train_challenge2_multi_release.py
Optimized for: Fast training with good results
"""
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Force CPU for stability
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HIP_VISIBLE_DEVICES'] = ''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from eegdash import EEGChallengeDataset

print("="*80)
print("🎯 CHALLENGE 2: EXTERNALIZING PREDICTION TRAINING")
print("="*80)
print("Model: CompactExternalizingCNN (150K params)")
print("Training: R1, R2, R3, R4")
print("Validation: R5")
print("Target: Predict externalizing factor from EEG")
print("="*80)
print()

# ============================================================================
# DATASET
# ============================================================================

class ExternalizingDataset(Dataset):
    """Dataset for externalizing prediction"""

    def __init__(self, releases, mini=True, max_datasets=100):
        print(f"\n📦 Loading data from releases: {releases}")
        print(f"   Mini mode: {mini}, Max datasets per release: {max_datasets}")
        
        self.windows = []
        self.scores = []
        self.sfreq = 100
        self.window_size = int(2 * self.sfreq)  # 2 seconds
        self.window_stride = int(1 * self.sfreq)  # 1 second overlap
        
        for release in releases:
            print(f"\n   Loading {release}...")
            try:
                dataset = EEGChallengeDataset(
                    release=release,
                    task="resting",
                    mini=mini,
                    cache_dir='data/raw'
                )
                
                valid_datasets = []
                for i, ds in enumerate(dataset):
                    if i >= max_datasets:
                        break
                    externalizing = ds.description.get("externalizing", None)
                    if externalizing is not None and not np.isnan(externalizing):
                        valid_datasets.append((ds, externalizing))
                
                print(f"   Found {len(valid_datasets)} datasets with externalizing scores")
                
                # Create windows
                windows_created = 0
                for ds, externalizing in valid_datasets:
                    raw_data = ds.raw.get_data()  # (129, n_timepoints)
                    n_channels, n_timepoints = raw_data.shape
                    
                    # Sliding windows
                    start_idx = 0
                    while start_idx + self.window_size <= n_timepoints:
                        window_data = raw_data[:, start_idx:start_idx+self.window_size]
                        self.windows.append(window_data)
                        self.scores.append(externalizing)
                        windows_created += 1
                        start_idx += self.window_stride
                
                print(f"   Created {windows_created} windows")
                
            except Exception as e:
                print(f"   ⚠️  Error loading {release}: {e}")
                continue
        
        print(f"\n✅ Total: {len(self.windows)} windows")
        if len(self.scores) > 0:
            print(f"   Externalizing range: [{min(self.scores):.3f}, {max(self.scores):.3f}]")
            print(f"   Mean: {np.mean(self.scores):.3f}, Std: {np.std(self.scores):.3f}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        X = self.windows[idx]
        
        # Normalize per-channel
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
        
        y = self.scores[idx]
        
        return torch.FloatTensor(X), torch.FloatTensor([y])


# ============================================================================
# MODEL
# ============================================================================

class CompactExternalizingCNN(nn.Module):
    """Compact CNN for externalizing prediction (150K params)"""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Conv1: 129x200 -> 32x100
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3),

            # Conv2: 32x100 -> 64x50
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.4),

            # Conv3: 64x50 -> 96x25
            nn.Conv1d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(96),
            nn.ELU(),
            nn.Dropout(0.5),

            # Global pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.regressor = nn.Sequential(
            nn.Linear(96, 48),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(48, 24),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(24, 1)
        )

    def forward(self, x):
        features = self.features(x)
        return self.regressor(features)


# ============================================================================
# TRAINING
# ============================================================================

def compute_nrmse(y_true, y_pred):
    """Compute Normalized Root Mean Squared Error"""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    nrmse = rmse / (y_true.max() - y_true.min())
    return nrmse


def train_model(model, train_loader, val_loader, epochs=30, lr=0.001):
    """Train the model"""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    best_nrmse = float('inf')
    patience_counter = 0
    patience = 10
    
    print(f"\n🚀 Training for up to {epochs} epochs...")
    print(f"   Batch size: {train_loader.batch_size}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Train
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.detach().numpy().flatten())
            train_labels.extend(labels.numpy().flatten())
        
        # Validate
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for data, labels in val_loader:
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(outputs.numpy().flatten())
                val_labels.extend(labels.numpy().flatten())
        
        # Metrics
        train_nrmse = compute_nrmse(np.array(train_labels), np.array(train_preds))
        val_nrmse = compute_nrmse(np.array(val_labels), np.array(val_preds))
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch:2d}/{epochs} ({epoch_time:.1f}s)  |  "
              f"Train NRMSE: {train_nrmse:.4f}  |  "
              f"Val NRMSE: {val_nrmse:.4f}  |  "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        scheduler.step(val_nrmse)
        
        # Save best model
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_nrmse': val_nrmse,
                'epoch': epoch
            }, 'weights_challenge_2_new.pt')
            print(f"   ✅ Best model saved (NRMSE: {best_nrmse:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping (no improvement for {patience} epochs)")
                break
    
    return best_nrmse


# ============================================================================
# MAIN
# ============================================================================

def main():
    start_time = time.time()
    
    # Load data
    print("\n" + "="*80)
    print("PHASE 1: DATA LOADING")
    print("="*80)
    
    train_dataset = ExternalizingDataset(
        releases=['R1', 'R2', 'R3', 'R4'],
        mini=False,  # Use full data
        max_datasets=50  # Limit for speed
    )
    
    val_dataset = ExternalizingDataset(
        releases=['R5'],
        mini=False,
        max_datasets=20
    )
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("\n❌ ERROR: No data loaded!")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Create model
    print("\n" + "="*80)
    print("PHASE 2: MODEL CREATION")
    print("="*80)
    
    model = CompactExternalizingCNN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: CompactExternalizingCNN")
    print(f"Parameters: {n_params:,}")
    
    # Train
    print("\n" + "="*80)
    print("PHASE 3: TRAINING")
    print("="*80)
    
    best_nrmse = train_model(model, train_loader, val_loader, epochs=30, lr=0.001)
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Best validation NRMSE: {best_nrmse:.4f}")
    print(f"Model saved to: weights_challenge_2_new.pt")
    print()
    print("Next steps:")
    print("  1. Test the model: python test_submission_verbose.py")
    print("  2. Update weights: cp weights_challenge_2_new.pt weights_challenge_2.pt")
    print("  3. Recreate submission: zip -j submission.zip submission.py weights_*.pt")
    print("  4. Submit to competition!")
    print("="*80)


if __name__ == "__main__":
    main()
