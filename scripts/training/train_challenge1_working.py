#!/usr/bin/env python3
"""
Challenge 1: Response Time Prediction - Working Version
Based on Challenge 2's proven approach (NRMSE 0.0918)
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
import json
import mne
import pandas as pd
from scipy.stats import pearsonr
from braindecode.models import EEGNeX

print("="*80)
print("🎯 CHALLENGE 1: RESPONSE TIME PREDICTION")
print("="*80)
print("Using Challenge 2's proven strategy (NRMSE 0.0918)")
print("Target: NRMSE < 0.5")
print("="*80)

# Configuration
CONFIG = {
    'data_dirs': ['data/ds005507-bdf', 'data/ds005506-bdf'],
    'batch_size': 32,
    'epochs': 100,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'early_stopping_patience': 15,
    'save_top_k': 5,
    'max_subjects': None,  # None = all subjects
}

print("\n⚙️  Configuration:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n��️  Device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print("\n" + "="*80)
print()


class ResponseTimeDataset(Dataset):
    """Load EEG windows with response times from BIDS events"""

    def __init__(self, data_dirs, max_subjects=None, augment=False):
        self.segments = []
        self.response_times = []
        self.augment = augment

        print(f"📁 Loading data (augment={augment})...")

        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            participants_file = data_dir / "participants.tsv"

            if not participants_file.exists():
                continue

            df = pd.read_csv(participants_file, sep='\t')

            if max_subjects:
                df = df.head(max_subjects)

            print(f"   📊 {data_dir.name}: {len(df)} subjects")

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"   {data_dir.name}", leave=False):
                subject_id = row['participant_id']
                subject_dir = data_dir / subject_id / "eeg"

                if not subject_dir.exists():
                    continue

                # Find contrastChangeDetection EEG files
                eeg_files = list(subject_dir.glob("*contrastChangeDetection*.bdf"))
                if not eeg_files:
                    continue

                for eeg_file in eeg_files:
                    try:
                        # Load EEG
                        raw = mne.io.read_raw_bdf(eeg_file, preload=True, verbose=False)

                        # Resample to 100Hz
                        if raw.info['sfreq'] != 100:
                            raw.resample(100, verbose=False)

                        data = raw.get_data()

                        # Ensure 129 channels
                        if data.shape[0] != 129:
                            continue

                        # Z-score normalize per channel
                        data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)

                        # Load events to get response times
                        events_file = eeg_file.with_name(eeg_file.name.replace('_eeg.bdf', '_events.tsv'))
                        if not events_file.exists():
                            continue

                        events_df = pd.read_csv(events_file, sep='\t')

                        # Find trials with response times
                        for _, event_row in events_df.iterrows():
                            if 'response_time' in events_df.columns:
                                rt = event_row.get('response_time', np.nan)
                            elif 'rt' in events_df.columns:
                                rt = event_row.get('rt', np.nan)
                            else:
                                continue

                            if pd.isna(rt) or rt <= 0:
                                continue

                            onset = event_row.get('onset', np.nan)
                            if pd.isna(onset):
                                continue

                            # Extract 2-second window starting 0.5s after stimulus
                            start_sample = int((onset + 0.5) * 100)
                            end_sample = start_sample + 200  # 2 seconds @ 100Hz

                            if end_sample > data.shape[1]:
                                continue

                            segment = data[:, start_sample:end_sample]

                            self.segments.append(segment)
                            self.response_times.append(rt)

                    except Exception as e:
                        continue

        self.segments = np.array(self.segments, dtype=np.float32)
        self.response_times = np.array(self.response_times, dtype=np.float32)

        print(f"\n   ✅ Loaded {len(self)} windows with response times")
        print(f"   RT range: {self.response_times.min():.3f} - {self.response_times.max():.3f} seconds")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        X = torch.FloatTensor(self.segments[idx])
        y = torch.FloatTensor([self.response_times[idx]])

        if self.augment:
            # Same augmentation as Challenge 2
            if torch.rand(1).item() < 0.5:
                scale = 0.8 + 0.4 * torch.rand(1).item()
                X = X * scale

            if torch.rand(1).item() < 0.3:
                n_channels = X.shape[0]
                n_drop = max(1, int(0.05 * n_channels))
                drop_channels = torch.randperm(n_channels)[:n_drop]
                X[drop_channels, :] = 0.0

            if torch.rand(1).item() < 0.2:
                noise = torch.randn_like(X) * 0.05
                X = X + noise

        return X, y


class EarlyStopping:
    def __init__(self, patience=15):
        self.patience = patience
        self.counter = 0
        self.best_score = None

    def __call__(self, val_nrmse):
        if self.best_score is None:
            self.best_score = val_nrmse
            return False

        if val_nrmse < self.best_score:
            self.best_score = val_nrmse
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def compute_nrmse(targets, predictions):
    targets = targets.cpu().numpy().flatten()
    predictions = predictions.cpu().numpy().flatten()
    mse = np.mean((targets - predictions) ** 2)
    rmse = np.sqrt(mse)
    target_range = targets.max() - targets.min()
    nrmse = rmse / target_range if target_range > 0 else float('inf')
    return nrmse


def pearson_correlation(targets, predictions):
    targets = targets.cpu().numpy().flatten()
    predictions = predictions.cpu().numpy().flatten()
    if len(targets) < 2:
        return 0.0
    r, _ = pearsonr(targets, predictions)
    return r


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc="Train", leave=False)
    for X, y in pbar:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = F.mse_loss(pred, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(pred.detach())
        all_targets.append(y.detach())

        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(loader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    nrmse = compute_nrmse(all_targets, all_preds)

    return avg_loss, nrmse


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc="Val", leave=False)
    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = F.mse_loss(pred, y)

        total_loss += loss.item()
        all_preds.append(pred.detach())
        all_targets.append(y.detach())

    avg_loss = total_loss / len(loader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    nrmse = compute_nrmse(all_targets, all_preds)
    pearson_r = pearson_correlation(all_targets, all_preds)

    return avg_loss, nrmse, pearson_r


def main():
    # Load data
    print("\n📦 Creating datasets...")
    train_dataset = ResponseTimeDataset(CONFIG['data_dirs'], CONFIG['max_subjects'], augment=True)
    val_dataset = ResponseTimeDataset(CONFIG['data_dirs'], CONFIG['max_subjects'], augment=False)

    # Split 80/20
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, _ = random_split(train_dataset, [train_size, val_size])

    val_split_size = int(0.2 * len(val_dataset))
    _, val_dataset = random_split(val_dataset, [len(val_dataset) - val_split_size, val_split_size])

    print(f"   Train: {len(train_dataset)} windows")
    print(f"   Val: {len(val_dataset)} windows")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0, pin_memory=True if device.type == 'cuda' else False)

    # Model
    print("\n🤖 Creating model...")
    model = EEGNeX(n_chans=129, n_outputs=1, n_times=200, sfreq=100).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")

    # Optimizer and schedulers
    optimizer = Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    criterion = nn.MSELoss()
    scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
    scheduler_cosine = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Early stopping
    early_stopping = EarlyStopping(patience=CONFIG['early_stopping_patience'])

    # Training loop
    print("\n" + "="*80)
    print(f"🚀 Starting training ({CONFIG['epochs']} epochs)")
    print("="*80)

    save_dir = Path('outputs/challenge1')
    save_dir.mkdir(parents=True, exist_ok=True)

    best_nrmse = float('inf')
    history = {'train_loss': [], 'val_nrmse': [], 'val_r': [], 'val_loss': [], 'lr': []}

    for epoch in range(CONFIG['epochs']):
        # Train
        train_loss, train_nrmse = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_nrmse, val_r = evaluate(model, val_loader, device)

        # Update schedulers
        scheduler_plateau.step(val_nrmse)
        scheduler_cosine.step()

        # Track history
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(float(train_loss))
        history['val_nrmse'].append(float(val_nrmse))
        history['val_r'].append(float(val_r))
        history['val_loss'].append(float(val_loss))
        history['lr'].append(float(current_lr))

        # Print progress
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']} | LR: {current_lr:.6f}")
        print(f"  Train Loss: {train_loss:.4f} | NRMSE: {train_nrmse:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | NRMSE: {val_nrmse:.4f} | r: {val_r:.3f}")

        # Save best model
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            torch.save(model.state_dict(), save_dir / 'challenge1_best.pt')
            torch.save(model.state_dict(), 'weights_challenge_1.pt')
            print(f"  ⭐ New best! Saved checkpoint")

        # Early stopping
        if early_stopping(val_nrmse):
            print(f"\n⏹️  Early stopping triggered (patience={CONFIG['early_stopping_patience']})")
            break

    # Save history
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print(f"   Best Val NRMSE: {best_nrmse:.4f}")
    print(f"   Target: < 0.5")
    print(f"   Status: {'✅ Achieved!' if best_nrmse < 0.5 else '🔴 Not yet'}")
    print("="*80)


if __name__ == '__main__':
    main()
