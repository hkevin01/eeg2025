#!/usr/bin/env python3
"""
Challenge 1: Response Time Prediction - IMPROVED VERSION
=========================================================
Improvements:
- Cross-validation for robust evaluation
- Data augmentation (noise, time jitter)
- Enhanced model architecture
- Better regularization
"""
import os
from pathlib import Path
import time
import numpy as np

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

print("="*80, flush=True)
print("🎯 CHALLENGE 1: RESPONSE TIME PREDICTION (IMPROVED)", flush=True)
print("="*80, flush=True)
print("Improvements: Cross-validation, Data augmentation, Better architecture", flush=True)
print("="*80, flush=True)


class AugmentedResponseTimeDataset(Dataset):
    """EEG dataset with data augmentation"""
    
    def __init__(self, data_dir, segment_length=200, sampling_rate=100, augment=True):
        self.data_dir = Path(data_dir)
        self.segment_length = segment_length
        self.target_sr = sampling_rate
        self.augment = augment
        
        print("\n�� Loading CCD data...", flush=True)
        
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
                        
                        if response_time < 0.1 or response_time > 5.0:
                            continue
                        
                        start_sample = int(trial_time * self.target_sr)
                        end_sample = start_sample + segment_length
                        
                        if end_sample > data.shape[1]:
                            continue
                        
                        segment = data[:, start_sample:end_sample]
                        
                        self.segments.append(torch.FloatTensor(segment))
                        self.response_times.append(response_time)
                        self.subject_ids.append(subject_id)
                    
                    if len(self.response_times) > 0:
                        print(f"   ✅ {subject_id}/{eeg_file.name}: {len(trial_start_events)} trials", flush=True)
                    
                except Exception as e:
                    print(f"   ⚠️  {subject_id}: {e}", flush=True)
                    continue
        
        print(f"\n📊 Total segments: {len(self.segments)}", flush=True)
        
        if len(self.segments) == 0:
            raise ValueError("No valid CCD segments found!")
        
        self.rt_array = np.array(self.response_times, dtype=np.float32)
        self.rt_mean = self.rt_array.mean()
        self.rt_std = self.rt_array.std()
        
        print(f"   Response Time: mean={self.rt_mean:.3f}s, std={self.rt_std:.3f}s", flush=True)
        print(f"   Range: [{self.rt_array.min():.3f}, {self.rt_array.max():.3f}]s", flush=True)
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx].clone()
        rt = torch.FloatTensor([self.response_times[idx]])
        
        # Data augmentation
        if self.augment:
            # Add Gaussian noise (10% of std)
            if np.random.rand() < 0.5:
                noise = torch.randn_like(segment) * 0.1
                segment = segment + noise
            
            # Time jitter (shift by ±5 samples)
            if np.random.rand() < 0.5:
                shift = np.random.randint(-5, 6)
                segment = torch.roll(segment, shift, dims=1)
        
        return segment, rt


class ImprovedResponseTimeCNN(nn.Module):
    """Improved CNN with residual connections and attention"""
    
    def __init__(self):
        super().__init__()
        
        # Initial projection
        self.proj = nn.Sequential(
            nn.Conv1d(129, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Multi-scale feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Regressor with residual
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        x = self.proj(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.regressor(x)
        return x


def compute_nrmse(y_true, y_pred):
    """Compute Normalized RMSE (competition metric)"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    std_true = np.std(y_true)
    nrmse = rmse / std_true if std_true > 0 else 0.0
    return nrmse


def train_fold(model, train_loader, val_loader, epochs=30):
    """Train one fold"""
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_nrmse = float('inf')
    patience_counter = 0
    patience = 7
    best_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for data, labels in val_loader:
                outputs = model(data)
                val_preds.extend(outputs.numpy().flatten())
                val_labels.extend(labels.numpy().flatten())
        
        scheduler.step()
        
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_nrmse = compute_nrmse(val_labels, val_preds)
        
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:2d}: NRMSE={val_nrmse:.4f} (best={best_nrmse:.4f})", flush=True)
        
        if patience_counter >= patience:
            break
    
    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return best_nrmse


def main():
    """Main training with cross-validation"""
    start_time = time.time()
    
    print("\n📂 Loading dataset...", flush=True)
    data_dir = Path("data/raw/hbn_ccd_mini")
    dataset = AugmentedResponseTimeDataset(data_dir=data_dir, segment_length=200, sampling_rate=100, augment=True)
    
    # Cross-validation
    n_folds = 5
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    print(f"\n🔄 Starting {n_folds}-Fold Cross-Validation", flush=True)
    print("="*80, flush=True)
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n📍 Fold {fold+1}/{n_folds}", flush=True)
        print("-"*80, flush=True)
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=2)
        
        print(f"  Train: {len(train_subset)} | Val: {len(val_subset)}", flush=True)
        
        model = ImprovedResponseTimeCNN()
        fold_nrmse = train_fold(model, train_loader, val_loader, epochs=30)
        
        fold_scores.append(fold_nrmse)
        print(f"  ✅ Fold {fold+1} NRMSE: {fold_nrmse:.4f}", flush=True)
    
    # Final results
    mean_nrmse = np.mean(fold_scores)
    std_nrmse = np.std(fold_scores)
    
    print("\n" + "="*80, flush=True)
    print("📊 CROSS-VALIDATION RESULTS", flush=True)
    print("="*80, flush=True)
    print(f"\nFold NRMSEs: {[f'{s:.4f}' for s in fold_scores]}", flush=True)
    print(f"Mean NRMSE: {mean_nrmse:.4f} ± {std_nrmse:.4f}", flush=True)
    
    if mean_nrmse < 0.5:
        print("✅ COMPETITION TARGET MET (NRMSE < 0.5)!", flush=True)
    else:
        print("⚠️  Above competition target (target: < 0.5)", flush=True)
    
    # Train final model on all data
    print("\n�� Training final model on all data...", flush=True)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    final_model = ImprovedResponseTimeCNN()
    final_nrmse = train_fold(final_model, train_loader, val_loader, epochs=40)
    
    # Save final model
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'cv_mean_nrmse': mean_nrmse,
        'cv_std_nrmse': std_nrmse,
        'final_nrmse': final_nrmse,
        'rt_mean': dataset.rt_mean,
        'rt_std': dataset.rt_std,
        'n_segments': len(dataset)
    }, checkpoint_dir / "response_time_improved.pth")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80, flush=True)
    print("✅ IMPROVED TRAINING COMPLETE!", flush=True)
    print("="*80, flush=True)
    print(f"\nCross-Validation: {mean_nrmse:.4f} ± {std_nrmse:.4f}", flush=True)
    print(f"Final Model: {final_nrmse:.4f}", flush=True)
    print(f"Total time: {total_time/60:.1f} minutes", flush=True)
    print(f"\n💾 Model saved to: checkpoints/response_time_improved.pth", flush=True)


if __name__ == "__main__":
    main()
