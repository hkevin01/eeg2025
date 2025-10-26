#!/usr/bin/env python3
"""
Challenge 1: Improved Training with Anti-Overfitting Measures
==============================================================
Competition: Cross-Task Transfer Learning - Response Time Prediction
Target: Predict response time from Contrast Change Detection EEG

Anti-Overfitting Strategy (Same as Challenge 2):
1. Data Augmentation: Amplitude scaling (±20%), channel dropout (20%), Gaussian noise
2. Strong Regularization: Weight decay (1e-4), gradient clipping, dropout
3. Early Stopping: Patience=15 epochs
4. Dual LR Schedulers: CosineAnnealingWarmRestarts + ReduceLROnPlateau
5. Save top-5 checkpoints for ensembling
6. Train/Val split with careful monitoring
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import mne
from tqdm import tqdm
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))
from utils.gpu_utils import setup_device
from braindecode.models import EEGNeX

print("="*80)
print("🎯 CHALLENGE 1: IMPROVED TRAINING (Anti-Overfitting)")
print("="*80)
print("Task: Response Time Prediction | Contrast Change Detection")
print("="*80)


class ResponseTimeDataset(Dataset):
    """Dataset for Challenge 1 with strong augmentation"""

    def __init__(self, data_dirs, max_subjects=None, augment=True):
        self.segments = []
        self.response_times = []
        self.augment = augment

        print(f"\n📁 Loading data (augment={augment})...")

        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            print(f"\n   📊 Processing {data_dir.name}...")

            # Find all subjects
            subject_dirs = sorted([d for d in data_dir.glob("sub-*") if d.is_dir()])

            if max_subjects:
                subject_dirs = subject_dirs[:max_subjects]

            for subject_dir in tqdm(subject_dirs, desc=f"   Loading {data_dir.name}"):
                eeg_dir = subject_dir / "eeg"
                if not eeg_dir.exists():
                    continue

                # Find Contrast Change Detection task files
                eeg_files = list(eeg_dir.glob("*contrastChangeDetection*_eeg.bdf"))
                if not eeg_files:
                    continue

                try:
                    # Load EEG data
                    raw = mne.io.read_raw_bdf(eeg_files[0], preload=True, verbose=False)

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
                    events_file = eeg_files[0].with_name(eeg_files[0].name.replace('_eeg.bdf', '_events.tsv'))
                    if not events_file.exists():
                        continue

                    events_df = pd.read_csv(events_file, sep='\t')

                    # Process events - look for trials with responses
                    # Format: contrastTrial_start -> target -> buttonPress (with feedback)
                    trial_onsets = []
                    response_times_list = []

                    # Find trial start and response events
                    for idx, row in events_df.iterrows():
                        event_code = str(row.get('event_code', ''))
                        onset = row.get('onset', None)

                        # Look for trial start (contrastTrial_start)
                        if 'Trial_start' in event_code or event_code == '5':
                            trial_onset = onset

                            # Look ahead for buttonPress within 5 seconds
                            for next_idx in range(idx + 1, min(idx + 20, len(events_df))):
                                next_row = events_df.iloc[next_idx]
                                next_event = str(next_row.get('event_code', ''))
                                next_onset = next_row.get('onset', None)
                                feedback = str(next_row.get('feedback', ''))

                                # Button press events (left or right)
                                if next_onset is not None and ('buttonPress' in next_event or
                                                               next_event in ['12', '13']):
                                    rt = next_onset - trial_onset
                                    # Valid RT: 0.1s to 5.0s, with feedback (correct or incorrect)
                                    if 0.1 < rt < 5.0 and feedback != 'n/a':
                                        trial_onsets.append(trial_onset)
                                        response_times_list.append(rt)
                                        break
                                # Stop if we hit next trial start
                                elif 'Trial_start' in next_event:
                                    break

                    # Extract 2-second segments for training (or 4s for augmentation)
                    segment_length = 400 if augment else 200  # 4s or 2s at 100Hz

                    for trial_onset, rt in zip(trial_onsets, response_times_list):
                        # Start 0.5s after trial onset
                        start_sample = int((trial_onset + 0.5) * 100)
                        end_sample = start_sample + segment_length

                        if end_sample <= data.shape[1]:
                            segment = data[:, start_sample:end_sample]
                            self.segments.append(segment)
                            self.response_times.append(rt)

                except Exception as e:
                    continue

        self.segments = np.array(self.segments, dtype=np.float32)
        self.response_times = np.array(self.response_times, dtype=np.float32)

        print(f"\n   ✅ Loaded {len(self)} windows with response times")
        if len(self) > 0:
            print(f"   RT range: {self.response_times.min():.3f} - {self.response_times.max():.3f} seconds")
            print(f"   RT mean: {self.response_times.mean():.3f} ± {self.response_times.std():.3f} seconds")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = torch.FloatTensor(self.segments[idx])
        rt = self.response_times[idx]

        if self.augment and segment.shape[1] == 400:
            # Random crop: 4s → 2s (200 samples)
            start = torch.randint(0, 201, (1,)).item()
            segment = segment[:, start:start+200]

            # Amplitude scaling: ±20%
            scale = 0.8 + 0.4 * torch.rand(1).item()
            segment = segment * scale

            # Random channel dropout (20% probability, drop 5% of channels)
            if torch.rand(1).item() < 0.2:
                n_dropout = int(0.05 * 129)
                dropout_channels = torch.randperm(129)[:n_dropout]
                segment[dropout_channels] = 0

            # Add Gaussian noise
            noise = torch.randn_like(segment) * 0.02
            segment = segment + noise

        return segment, torch.FloatTensor([rt])


def calculate_nrmse(y_true, y_pred):
    """NRMSE - Normalized Root Mean Squared Error"""
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


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Training epoch with gradient clipping"""
    model.train()
    total_loss = 0

    for eeg, rts in tqdm(loader, desc="Train", leave=False):
        eeg, rts = eeg.to(device), rts.to(device)

        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(eeg)
                loss = criterion(outputs, rts)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(eeg)
            loss = criterion(outputs, rts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    """Evaluation with NRMSE and correlation"""
    model.eval()
    all_preds, all_true = [], []
    total_loss = 0

    criterion = nn.MSELoss()

    with torch.no_grad():
        for eeg, rts in tqdm(loader, desc="Val", leave=False):
            eeg, rts = eeg.to(device), rts.to(device)
            outputs = model(eeg)
            loss = criterion(outputs, rts)

            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_true.extend(rts.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    nrmse = calculate_nrmse(all_true, all_preds)
    mae = np.mean(np.abs(all_true - all_preds))
    r, _ = pearsonr(all_true, all_preds) if len(all_true) > 1 else (0, 1)
    val_loss = total_loss / len(loader)

    return nrmse, mae, r, val_loss


def main():
    # Configuration
    CONFIG = {
        'data_dirs': ['data/ds005507-bdf', 'data/ds005506-bdf'],
        'batch_size': 32,
        'epochs': 100,
        'lr': 0.001,
        'weight_decay': 1e-4,
        'early_stopping_patience': 15,
        'max_subjects': None,
        'save_top_k': 5,
    }

    print("\n⚙️  Configuration:")
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")

    # Setup device
    device, gpu_config = setup_device(optimize=True)
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create datasets
    print("\n📦 Creating datasets...")
    train_dataset = ResponseTimeDataset(CONFIG['data_dirs'], CONFIG['max_subjects'], augment=True)
    val_dataset = ResponseTimeDataset(CONFIG['data_dirs'], CONFIG['max_subjects'], augment=False)

    if len(train_dataset) == 0:
        print("❌ No training data loaded! Check data paths and event files.")
        return

    # Split datasets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, _ = random_split(train_dataset, [train_size, val_size])

    val_size = int(0.2 * len(val_dataset))
    train_val_size = len(val_dataset) - val_size
    _, val_dataset = random_split(val_dataset, [train_val_size, val_size])

    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                          shuffle=False, num_workers=4, pin_memory=True)

    # Create model
    print("\n🏗️  Building EEGNeX model...")
    model = EEGNeX(
        n_outputs=1,
        n_chans=129,
        n_times=200,
        sfreq=100
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'],
                           weight_decay=CONFIG['weight_decay'])

    # Dual LR schedulers
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=CONFIG['early_stopping_patience'])

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_nrmse': [],
        'val_mae': [],
        'val_r': [],
        'lr': []
    }

    best_nrmse = float('inf')
    best_checkpoints = []

    print("\n🚀 Starting training...")
    print("="*80)

    for epoch in range(CONFIG['epochs']):
        epoch_start = datetime.now()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)

        # Validate
        val_nrmse, val_mae, val_r, val_loss = evaluate(model, val_loader, device)

        # Update LR schedulers
        scheduler_cosine.step()
        scheduler_plateau.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_nrmse'].append(val_nrmse)
        history['val_mae'].append(val_mae)
        history['val_r'].append(val_r)
        history['lr'].append(current_lr)

        epoch_time = (datetime.now() - epoch_start).total_seconds()

        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f} | NRMSE: {val_nrmse:.4f} | MAE: {val_mae:.4f} | r: {val_r:.4f}")
        print(f"  LR: {current_lr:.2e}")

        # Save checkpoints
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': val_nrmse,
                'val_mae': val_mae,
                'val_r': val_r,
                'config': CONFIG
            }

            # Save best
            torch.save(checkpoint, 'checkpoints/challenge1_improved_best.pth')
            print(f"  ✅ New best NRMSE: {val_nrmse:.4f}")

            # Manage top-k checkpoints
            checkpoint_path = f"checkpoints/challenge1_improved_epoch{epoch+1}.pth"
            torch.save(checkpoint, checkpoint_path)
            best_checkpoints.append((val_nrmse, checkpoint_path))
            best_checkpoints.sort()

            if len(best_checkpoints) > CONFIG['save_top_k']:
                _, path_to_remove = best_checkpoints.pop()
                Path(path_to_remove).unlink(missing_ok=True)

        # Early stopping
        early_stopping(val_nrmse)
        if early_stopping.early_stop:
            print(f"\n⏹️  Early stopping triggered at epoch {epoch+1}")
            break

    # Save final model
    torch.save(model.state_dict(), 'weights_challenge_1_improved.pt')
    print("\n✅ Training complete!")
    print(f"   Best Val NRMSE: {best_nrmse:.4f}")
    print(f"   Weights saved: weights_challenge_1_improved.pt")

    # Save history
    with open('logs/challenge1/training_history_improved.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n📊 Training Summary:")
    print(f"   Final Train Loss: {history['train_loss'][-1]:.6f}")
    print(f"   Final Val NRMSE: {history['val_nrmse'][-1]:.4f}")
    print(f"   Best Val NRMSE: {best_nrmse:.4f}")
    print(f"   Final Correlation: {history['val_r'][-1]:.4f}")


if __name__ == '__main__':
    main()
