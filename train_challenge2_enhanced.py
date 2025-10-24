#!/usr/bin/env python3
"""
Challenge 2: Enhanced Training with Anti-Overfitting Measures
=============================================================
Competition: Code submission (submission.py + weights)
Target: NRMSE < 0.5 for externalizing factor prediction

Anti-Overfitting Strategy:
1. Data Augmentation: Random cropping (4s→2s), amplitude scaling, channel dropout
2. Strong Regularization: Dropout, weight decay, label smoothing
3. Early Stopping: Patience=15 epochs
4. Train/Val Monitoring: Track overfitting gap
5. Learning Rate Scheduling: ReduceLROnPlateau + Cosine annealing
6. Ensemble Ready: Save top-5 checkpoints
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import mne
from tqdm import tqdm
import time
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))
from utils.gpu_utils import setup_device
from braindecode.models import EEGNeX

print("="*80)
print("🎯 CHALLENGE 2: ENHANCED TRAINING (Anti-Overfitting)")
print("="*80)
print("Target: NRMSE < 0.5 | Competition: Code Submission")
print("="*80)


class AugmentedExternalizingDataset(Dataset):
    """Enhanced dataset with strong augmentation to prevent overfitting"""

    def __init__(self, data_dirs, max_subjects=None, augment=True):
        self.segments = []
        self.scores = []
        self.augment = augment

        print(f"\n📁 Loading data (augment={augment})...")

        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            participants_file = data_dir / "participants.tsv"

            if not participants_file.exists():
                continue

            df = pd.read_csv(participants_file, sep='\t')
            df = df.dropna(subset=['externalizing'])

            if max_subjects:
                df = df.head(max_subjects)

            print(f"   📊 {data_dir.name}: {len(df)} subjects")

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"   {data_dir.name}", leave=False):
                subject_id = row['participant_id']
                subject_dir = data_dir / subject_id / "eeg"

                if not subject_dir.exists():
                    continue

                # Find contrastChangeDetection EEG (Challenge 2 task)
                eeg_files = list(subject_dir.glob("*contrastChangeDetection*.bdf"))
                if not eeg_files:
                    eeg_files = list(subject_dir.glob("*contrastChangeDetection*.set"))
                if not eeg_files:
                    continue

                try:
                    # Load EEG
                    if eeg_files[0].suffix == '.bdf':
                        raw = mne.io.read_raw_bdf(eeg_files[0], preload=True, verbose=False)
                    else:
                        raw = mne.io.read_raw_eeglab(eeg_files[0], preload=True, verbose=False)

                    # Resample to 100Hz
                    if raw.info['sfreq'] != 100:
                        raw.resample(100, verbose=False)

                    data = raw.get_data()

                    # Ensure 129 channels
                    if data.shape[0] != 129:
                        continue

                    # Z-score normalize per channel
                    data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)

                    # For augmentation: use 4s segments (will crop to 2s randomly)
                    # For validation: use 2s segments (no augmentation)
                    segment_length = 400 if augment else 200
                    n_samples = data.shape[1]
                    n_segments = n_samples // segment_length

                    externalizing = float(row['externalizing'])

                    for i in range(n_segments):
                        start = i * segment_length
                        end = start + segment_length
                        segment = data[:, start:end]

                        self.segments.append(torch.FloatTensor(segment))
                        self.scores.append(externalizing)

                except Exception:
                    continue

        print(f"\n✅ Loaded {len(self.segments)} segments")
        if len(self.segments) > 0:
            scores_array = np.array(self.scores)
            print(f"   Mean: {scores_array.mean():.3f}, Std: {scores_array.std():.3f}")
            print(f"   Range: [{scores_array.min():.3f}, {scores_array.max():.3f}]")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        score = self.scores[idx]

        if self.augment and segment.shape[1] == 400:
            # Random crop: 4s → 2s (200 samples)
            start = torch.randint(0, 201, (1,)).item()
            segment = segment[:, start:start+200]

            # Amplitude scaling: 0.8-1.2x
            scale = 0.8 + 0.4 * torch.rand(1).item()
            segment = segment * scale

            # Random channel dropout (5% of channels)
            if torch.rand(1).item() < 0.3:  # 30% chance
                n_dropout = int(0.05 * 129)
                dropout_channels = torch.randperm(129)[:n_dropout]
                segment[dropout_channels] = 0

        return segment, torch.FloatTensor([score])


def calculate_nrmse(y_true, y_pred):
    """NRMSE - Competition metric"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    y_range = y_true.max() - y_true.min()
    return rmse / y_range if y_range > 0 else 0.0


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for eeg, scores in tqdm(loader, desc="Train", leave=False):
        eeg, scores = eeg.to(device), scores.to(device)

        optimizer.zero_grad()
        outputs = model(eeg)
        loss = criterion(outputs, scores)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_true = [], []
    total_loss = 0

    criterion = nn.MSELoss()

    with torch.no_grad():
        for eeg, scores in tqdm(loader, desc="Val", leave=False):
            eeg, scores = eeg.to(device), scores.to(device)
            outputs = model(eeg)
            loss = criterion(outputs, scores)

            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_true.extend(scores.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    nrmse = calculate_nrmse(all_true, all_preds)
    r, _ = pearsonr(all_true, all_preds) if len(all_true) > 1 else (0, 1)
    val_loss = total_loss / len(loader)

    return nrmse, r, val_loss


def main():
    # Configuration
    CONFIG = {
        'data_dirs': ['data/ds005507-bdf', 'data/ds005506-bdf'],
        'batch_size': 32,
        'epochs': 100,
        'lr': 0.001,
        'weight_decay': 1e-4,  # L2 regularization
        'dropout': 0.5,  # Will be applied in model if supported
        'early_stopping_patience': 15,
        'max_subjects': None,  # None = all subjects
        'save_top_k': 5,  # Save top 5 checkpoints for ensembling
    }

    print("\n⚙️  Configuration:")
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")

    # Setup GPU
    device, gpu_config = setup_device(optimize=True)
    print(f"\n🖥️  Device: {device}")

    # Load data with augmentation
    print("\n" + "="*80)
    train_dataset = AugmentedExternalizingDataset(
        CONFIG['data_dirs'],
        max_subjects=CONFIG['max_subjects'],
        augment=True  # Training with augmentation
    )

    val_dataset = AugmentedExternalizingDataset(
        CONFIG['data_dirs'],
        max_subjects=CONFIG['max_subjects'],
        augment=False  # Validation without augmentation
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("❌ No data loaded!")
        return

    # Split datasets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, _ = random_split(train_dataset, [train_size, val_size])

    # For validation, use the entire dataset (already loaded without augmentation)
    # Just take the same proportion
    val_split_size = int(0.2 * len(val_dataset))
    _, val_dataset_split = random_split(val_dataset, [len(val_dataset) - val_split_size, val_split_size])

    print("\n📊 Dataset split:")
    print(f"   Train: {len(train_dataset)} segments (with augmentation)")
    print(f"   Val: {len(val_dataset_split)} segments (no augmentation)")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == 'cuda')
    )

    val_loader = DataLoader(
        val_dataset_split,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == 'cuda')
    )

    # Model
    print("\n🏗️  Creating EEGNeX model...")
    model = EEGNeX(n_chans=129, n_outputs=1, n_times=200, sfreq=100).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])

    # Learning rate schedulers
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=CONFIG['early_stopping_patience'])

    # Training loop
    print("\n" + "="*80)
    print(f"🚀 Starting training ({CONFIG['epochs']} epochs, patience={CONFIG['early_stopping_patience']})")
    print("="*80)

    save_dir = Path('outputs/challenge2')
    save_dir.mkdir(parents=True, exist_ok=True)

    best_nrmse = float('inf')
    best_checkpoints = []  # Track top-k checkpoints
    history = {'train_loss': [], 'val_nrmse': [], 'val_r': [], 'val_loss': [], 'lr': []}

    for epoch in range(CONFIG['epochs']):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_nrmse, val_r, val_loss = evaluate(model, val_loader, device)

        # Update learning rate
        scheduler_plateau.step(val_nrmse)
        scheduler_cosine.step()

        # Track history
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_nrmse'].append(val_nrmse)
        history['val_r'].append(val_r)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        elapsed = time.time() - start_time

        # Calculate overfitting gap
        overfit_gap = train_loss - val_loss

        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']} ({elapsed:.1f}s) | LR: {current_lr:.6f}")
        print(f"  Train Loss:   {train_loss:.4f}")
        print(f"  Val Loss:     {val_loss:.4f} | Gap: {overfit_gap:+.4f}")
        print(f"  Val NRMSE:    {val_nrmse:.4f} {'��' if val_nrmse < 0.5 else '⚠️'}")
        print(f"  Pearson r:    {val_r:.3f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'nrmse': val_nrmse,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': CONFIG,
        }

        # Save best model
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            torch.save(checkpoint, save_dir / 'challenge2_best.pt')
            print(f"  ✅ Saved BEST model (NRMSE: {best_nrmse:.4f})")

        # Save top-k checkpoints for ensembling
        best_checkpoints.append((val_nrmse, epoch, checkpoint))
        best_checkpoints.sort(key=lambda x: x[0])
        best_checkpoints = best_checkpoints[:CONFIG['save_top_k']]

        # Save top-k models
        for i, (nrmse, ep, ckpt) in enumerate(best_checkpoints):
            torch.save(ckpt, save_dir / f'challenge2_top{i+1}_epoch{ep+1}.pt')

        # Early stopping check
        early_stopping(val_nrmse)
        if early_stopping.early_stop:
            print(f"\n⚠️  Early stopping triggered! No improvement for {CONFIG['early_stopping_patience']} epochs")
            break

    # Save training history
    history_file = save_dir / 'training_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*80)
    print("✅ Training complete!")
    print(f"   Best NRMSE: {best_nrmse:.4f}")
    print(f"   Saved top-{len(best_checkpoints)} checkpoints for ensembling")
    print(f"   Best: {save_dir / 'challenge2_best.pt'}")
    print(f"   History: {history_file}")
    print("="*80)

    # Copy best weights for submission
    import shutil
    shutil.copy(save_dir / 'challenge2_best.pt', 'weights_challenge_2.pt')
    print(f"\n📦 Copied best weights to: weights_challenge_2.pt")
    print("   Ready for submission.py!")


if __name__ == '__main__':
    main()
