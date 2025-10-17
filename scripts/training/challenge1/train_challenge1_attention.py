#!/usr/bin/env python3
"""
Challenge 1: Response Time Prediction with Sparse Attention
============================================================
Enhanced architecture with:
- Sparse Multi-Head Self-Attention (O(N) complexity)
- Channel Attention for spatial EEG features
- Temporal Attention for time-series patterns

Expected improvement: 10-15% over baseline (0.4523 → 0.38-0.42 NRMSE)
"""
import os
from pathlib import Path
import time
import numpy as np
import sys

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))

# Force CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HIP_VISIBLE_DEVICES'] = ''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import mne

# Import attention model
from challenge1_attention import LightweightResponseTimeCNNWithAttention

print("="*80, flush=True)
print("🎯 CHALLENGE 1: RESPONSE TIME WITH SPARSE ATTENTION", flush=True)
print("="*80, flush=True)
print("Model: LightweightResponseTimeCNNWithAttention (846K params)", flush=True)
print("Features: Sparse O(N) attention, Channel/Temporal attention", flush=True)
print("Expected: 10-15% improvement over baseline", flush=True)
print("="*80, flush=True)


class AugmentedResponseTimeDataset(Dataset):
    """EEG dataset with data augmentation"""
    
    def __init__(self, data_dir, segment_length=200, sampling_rate=100, augment=True):
        self.data_dir = Path(data_dir)
        self.segment_length = segment_length
        self.target_sr = sampling_rate
        self.augment = augment
        
        print("\n📦 Loading CCD data...", flush=True)
        
        self.segments = []
        self.response_times = []
        self.subject_ids = []
        
        subject_dirs = sorted(self.data_dir.glob("sub-*"))
        print(f"   Found {len(subject_dirs)} subjects", flush=True)
        
        for subject_dir in subject_dirs:
            subject_id = subject_dir.name
            eeg_dir = subject_dir / "eeg"
            
            if not eeg_dir.exists():
                continue
            
            ccd_files = list(eeg_dir.glob("*contrastChangeDetection*.bdf"))
            if not ccd_files:
                continue
            
            for eeg_file in ccd_files:
                try:
                    raw = mne.io.read_raw_bdf(eeg_file, preload=True, verbose=False)
                    
                    if raw.info['sfreq'] != self.target_sr:
                        raw.resample(self.target_sr, verbose=False)
                    
                    data = raw.get_data()
                    
                    if data.shape[0] != 129:
                        continue
                    
                    events_file = eeg_file.parent / eeg_file.name.replace('_eeg.bdf', '_events.tsv')
                    if not events_file.exists():
                        continue
                    
                    events_df = pd.read_csv(events_file, sep='\t')
                    
                    trial_start_events = events_df[events_df['value'].str.contains('contrastTrial_start', case=False, na=False)]
                    button_press_events = events_df[events_df['value'].str.contains('buttonPress', case=False, na=False)]
                    
                    if len(trial_start_events) == 0 or len(button_press_events) == 0:
                        continue
                    
                    # Standardize per channel
                    data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
                    
                    for _, trial_event in trial_start_events.iterrows():
                        trial_time = trial_event['onset']
                        
                        later_presses = button_press_events[button_press_events['onset'] > trial_time]
                        if len(later_presses) == 0:
                            continue
                        
                        press_event = later_presses.iloc[0]
                        response_time = press_event['onset'] - trial_time
                        
                        if response_time < 0.2 or response_time > 5.0:
                            continue
                        
                        start_sample = int(trial_time * self.target_sr)
                        end_sample = start_sample + self.segment_length
                        
                        if end_sample > data.shape[1]:
                            continue
                        
                        segment = data[:, start_sample:end_sample]
                        
                        self.segments.append(segment.copy())
                        self.response_times.append(response_time)
                        self.subject_ids.append(subject_id)
                        
                except Exception as e:
                    continue
        
        print(f"   ✅ Loaded {len(self.segments)} segments from {len(set(self.subject_ids))} subjects", flush=True)
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx].copy()
        response_time = self.response_times[idx]
        
        if self.augment and np.random.rand() < 0.5:
            noise_level = np.random.uniform(0.01, 0.05)
            segment += np.random.randn(*segment.shape) * noise_level
        
        return torch.FloatTensor(segment), torch.FloatTensor([response_time])


def compute_nrmse(y_true, y_pred):
    """Compute Normalized Root Mean Squared Error"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    return nrmse


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions.extend(output.detach().cpu().numpy())
        targets.extend(target.detach().cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    nrmse = compute_nrmse(targets, predictions)
    
    return avg_loss, nrmse


def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            predictions.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    
    nrmse = compute_nrmse(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    
    if len(targets) > 1 and np.std(targets) > 0 and np.std(predictions) > 0:
        corr, _ = pearsonr(targets, predictions)
    else:
        corr = 0.0
    
    return avg_loss, nrmse, mse, mae, corr


def train_model(model, train_loader, val_loader, epochs=50, device='cpu'):
    """Train the model with attention"""
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_nrmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    max_patience = 15
    
    print("\n🚀 Starting training...\n", flush=True)
    
    for epoch in range(epochs):
        train_loss, train_nrmse = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_nrmse, val_mse, val_mae, val_corr = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_nrmse)
        
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train NRMSE: {train_nrmse:.4f} | "
              f"Val NRMSE: {val_nrmse:.4f} | "
              f"Val Corr: {val_corr:.4f}", flush=True)
        
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            best_epoch = epoch + 1
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'nrmse': val_nrmse,
                'mse': val_mse,
                'mae': val_mae,
                'correlation': val_corr
            }, 'checkpoints/response_time_attention.pth')
            
            print(f"   ⭐ New best! NRMSE: {val_nrmse:.4f} (Epoch {best_epoch})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\n⚠️  Early stopping at epoch {epoch+1} (patience exceeded)", flush=True)
                break
    
    return best_nrmse


def main():
    start_time = time.time()
    device = 'cpu'
    
    # Create checkpoints directory
    Path("checkpoints").mkdir(exist_ok=True)
    
    # Load dataset
    data_dir = Path("data/raw/hbn_ccd_mini")
    dataset = AugmentedResponseTimeDataset(data_dir=data_dir, segment_length=200, sampling_rate=100, augment=True)
    
    # 5-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_nrmses = []
    
    print("\n📊 Starting 5-Fold Cross-Validation\n", flush=True)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset), 1):
        print(f"\n{'='*80}")
        print(f"FOLD {fold}/5")
        print(f"{'='*80}\n", flush=True)
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=2)
        
        print(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}", flush=True)
        
        # Create model with attention
        model = LightweightResponseTimeCNNWithAttention(
            num_channels=129,
            seq_length=200,
            dropout=0.4
        ).to(device)
        
        params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {params:,}", flush=True)
        
        # Train
        fold_nrmse = train_model(model, train_loader, val_loader, epochs=50, device=device)
        fold_nrmses.append(fold_nrmse)
        
        print(f"\n✅ Fold {fold} complete: NRMSE = {fold_nrmse:.4f}\n", flush=True)
    
    # Summary
    mean_nrmse = np.mean(fold_nrmses)
    std_nrmse = np.std(fold_nrmses)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("✅ CROSS-VALIDATION COMPLETE")
    print("="*80)
    print(f"Mean NRMSE: {mean_nrmse:.4f} ± {std_nrmse:.4f}")
    print(f"Fold NRMSEs: {[f'{x:.4f}' for x in fold_nrmses]}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Best model saved: checkpoints/response_time_attention.pth")
    print("\nComparison:")
    print(f"  Baseline (CNN only):        0.4523 NRMSE")
    print(f"  With Attention (this run):  {mean_nrmse:.4f} NRMSE")
    if mean_nrmse < 0.4523:
        improvement = ((0.4523 - mean_nrmse) / 0.4523) * 100
        print(f"  Improvement: {improvement:.1f}% better! 🎉")
    print("="*80)


if __name__ == "__main__":
    main()
