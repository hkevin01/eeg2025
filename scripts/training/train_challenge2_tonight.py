#!/usr/bin/env python3
"""
Challenge 2 Training - Tonight's Run
Uses starter kit approach: loads RestingState EEG directly from BIDS
Model: Standard braindecode EEGNeX (competition-compatible)
GPU: AMD RX 5600 XT via ROCm SDK
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

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))
from utils.gpu_utils import setup_device
from braindecode.models import EEGNeX

print("="*80)
print("🎯 CHALLENGE 2: EXTERNALIZING FACTOR - TONIGHT'S RUN")
print("="*80)

class QuickExternalizingDataset(Dataset):
    """Fast dataset loader - RestingState EEG + externalizing scores

    Features:
    - Data augmentation: random cropping from 4s windows
    - Amplitude scaling for robustness
    - Channel dropout simulation
    """

    def __init__(self, data_dirs, max_subjects=None, augment=False):
        self.segments = []
        self.scores = []
        self.augment = augment

        print("\n📁 Loading data from BIDS datasets...")
        if augment:
            print("   🔄 Data augmentation: ENABLED")

        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            participants_file = data_dir / "participants.tsv"

            if not participants_file.exists():
                print(f"   ⚠️  No participants.tsv in {data_dir.name}")
                continue

            df = pd.read_csv(participants_file, sep='\t')
            df = df.dropna(subset=['externalizing'])

            if max_subjects:
                df = df.head(max_subjects)

            print(f"   📊 {data_dir.name}: {len(df)} subjects with externalizing scores")

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"   Loading {data_dir.name}"):
                subject_id = row['participant_id']
                subject_dir = data_dir / subject_id / "eeg"

                if not subject_dir.exists():
                    continue

                # Find RestingState EEG
                eeg_files = list(subject_dir.glob("*RestingState*.bdf"))
                if not eeg_files:
                    eeg_files = list(subject_dir.glob("*RestingState*.set"))
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

                    # Create 4-second segments for augmentation (400 samples @ 100Hz)
                    # We'll randomly crop to 2s (200 samples) during training
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
                    # Skip problematic files
                    continue

        print(f"\n✅ Total segments loaded: {len(self.segments)}")
        scores_array = np.array(self.scores)
        print(f"   Externalizing: mean={scores_array.mean():.3f}, std={scores_array.std():.3f}")
        print(f"   Range: [{scores_array.min():.3f}, {scores_array.max():.3f}]")


    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        score = torch.FloatTensor([self.scores[idx]])

        if self.augment and segment.shape[1] == 400:
            # Random crop from 4s to 2s (data augmentation)
            start = torch.randint(0, 201, (1,)).item()  # Random start position
            segment = segment[:, start:start+200]

            # Random amplitude scaling (0.8 to 1.2)
            scale = 0.8 + 0.4 * torch.rand(1).item()
            segment = segment * scale

        return segment, score
def calculate_nrmse(y_true, y_pred):
    """NRMSE - Competition metric"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    y_range = y_true.max() - y_true.min()
    return rmse / y_range if y_range > 0 else 0.0


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for eeg, scores in tqdm(loader, desc="Training", leave=False):
        eeg, scores = eeg.to(device), scores.to(device)

        optimizer.zero_grad()
        outputs = model(eeg)
        loss = criterion(outputs, scores)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for eeg, scores in tqdm(loader, desc="Validating", leave=False):
            eeg = eeg.to(device)
            outputs = model(eeg)

            all_preds.extend(outputs.cpu().numpy().flatten())
            all_true.extend(scores.numpy().flatten())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    nrmse = calculate_nrmse(all_true, all_preds)
    r, _ = pearsonr(all_true, all_preds)

    return nrmse, r


def main():
    # Configuration
    DATA_DIRS = [
        'data/ds005507-bdf',
        'data/ds005506-bdf',
    ]

    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 0.001
    MAX_SUBJECTS = None  # None = all subjects, or set a number for quick test

    # Setup GPU
    device, gpu_config = setup_device(optimize=True)
    print(f"\n🖥️  Device: {device}")

    # Load data
    print("\n" + "="*80)
    dataset = QuickExternalizingDataset(DATA_DIRS, max_subjects=MAX_SUBJECTS)

    if len(dataset) == 0:
        print("❌ No data loaded! Check your data paths.")
        return

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"\n📊 Dataset split:")
    print(f"   Train: {len(train_dataset)} segments")
    print(f"   Val: {len(val_dataset)} segments")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Model - Standard EEGNeX (competition-compatible)
    print("\n🏗️  Creating model...")
    model = EEGNeX(n_chans=129, n_outputs=1, n_times=200, sfreq=100).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   EEGNeX: {n_params:,} parameters")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    print("\n" + "="*80)
    print(f"🚀 Starting training for {EPOCHS} epochs...")
    print("="*80)

    best_nrmse = float('inf')
    save_dir = Path('outputs/challenge2')
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_nrmse, val_r = evaluate(model, val_loader, device)

        # Update LR
        scheduler.step(val_nrmse)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{EPOCHS} ({elapsed:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val NRMSE:  {val_nrmse:.4f}")
        print(f"  Pearson r:  {val_r:.3f}")

        # Save best
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'nrmse': best_nrmse,
            }
            checkpoint_path = save_dir / 'challenge2_best.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✅ Saved best model (NRMSE: {best_nrmse:.4f})")

        print()

    print("="*80)
    print(f"✅ Training complete!")
    print(f"   Best NRMSE: {best_nrmse:.4f}")
    print(f"   Checkpoint: {checkpoint_path}")
    print("="*80)


if __name__ == '__main__':
    main()
