#!/usr/bin/env python3
"""
Challenge 2: Externalizing Factor Prediction
=============================================
Competition-specific trainer for predicting externalizing factor only.

Requirements:
- Input: (batch, 129, 200) - 129 channels, 200 samples @ 100Hz
- Output: (batch, 1) - externalizing score
- Metric: NRMSE (lower is better, target < 0.5)
"""
import os
import sys
from pathlib import Path
import time

# Force CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HIP_VISIBLE_DEVICES'] = ''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mne

print("="*80)
print("🎯 CHALLENGE 2: EXTERNALIZING FACTOR PREDICTION")
print("="*80)
print("Competition: https://eeg2025.github.io/")
print("Metric: NRMSE (target < 0.5)")
print("Device: CPU")
print("="*80)


class ExternalizingDataset(Dataset):
    """EEG dataset for externalizing factor prediction"""

    def __init__(self, data_dir, segment_length=200, sampling_rate=100):
        """
        Args:
            data_dir: Path to HBN data
            segment_length: 200 samples (2 seconds @ 100Hz)
            sampling_rate: Target sampling rate (100Hz for competition)
        """
        self.data_dir = Path(data_dir)
        self.segment_length = segment_length
        self.target_sr = sampling_rate

        # Load participants with externalizing scores
        print("\n📋 Loading externalizing scores...")
        participants_file = self.data_dir / "participants.tsv"
        self.participants_df = pd.read_csv(participants_file, sep='\t')

        # Filter for subjects with externalizing scores
        self.participants_df = self.participants_df.dropna(subset=['externalizing'])

        print(f"   Participants with externalizing scores: {len(self.participants_df)}")

        # Find subjects with RestingState EEG
        print("🔍 Finding RestingState EEG files...")
        self.segments = []
        self.externalizing_scores = []

        for _, row in self.participants_df.iterrows():
            subject_id = row['participant_id']
            subject_dir = self.data_dir / subject_id / "eeg"

            if not subject_dir.exists():
                continue

            # Find RestingState EEG
            eeg_files = list(subject_dir.glob("*RestingState*.set"))
            if not eeg_files:
                continue

            try:
                # Load EEG
                raw = mne.io.read_raw_eeglab(eeg_files[0], preload=True, verbose=False)

                # Resample to 100Hz if needed
                if raw.info['sfreq'] != self.target_sr:
                    raw.resample(self.target_sr, verbose=False)

                data = raw.get_data()

                # Ensure 129 channels
                if data.shape[0] != 129:
                    print(f"   ⚠️  {subject_id}: {data.shape[0]} channels (expected 129), skipping")
                    continue

                # Standardize per channel
                data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)

                # Create segments of 200 samples (2 seconds)
                n_samples = data.shape[1]
                n_segments = n_samples // segment_length

                externalizing = float(row['externalizing'])

                for i in range(n_segments):
                    start = i * segment_length
                    end = start + segment_length
                    segment = data[:, start:end]

                    self.segments.append(torch.FloatTensor(segment))
                    self.externalizing_scores.append(externalizing)

                print(f"   ✅ {subject_id}: {n_segments} segments")

            except Exception as e:
                print(f"   ⚠️  {subject_id}: {e}")
                continue

        print(f"\n📊 Total segments: {len(self.segments)}")

        # Normalize externalizing scores
        self.scores_array = np.array(self.externalizing_scores, dtype=np.float32)
        self.score_mean = self.scores_array.mean()
        self.score_std = self.scores_array.std()

        print(f"   Externalizing: mean={self.score_mean:.3f}, std={self.score_std:.3f}")

        # Don't normalize for competition - keep original scale
        # The metric is NRMSE which handles normalization

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        score = torch.FloatTensor([self.externalizing_scores[idx]])
        return segment, score


class ExternalizingCNN(nn.Module):
    """CNN for externalizing factor prediction

    Input: (batch, 129, 200)
    Output: (batch, 1)
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Conv1: 129x200 -> 64x100
            nn.Conv1d(129, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Conv2: 64x100 -> 128x50
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Conv3: 128x50 -> 256x25
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            # Global pooling: 256x25 -> 256x1
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 129, 200)
        Returns:
            (batch, 1)
        """
        features = self.features(x)
        output = self.regressor(features)
        return output


def compute_nrmse(y_true, y_pred):
    """Compute Normalized RMSE (competition metric)"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    std_true = np.std(y_true)
    nrmse = rmse / std_true if std_true > 0 else 0.0
    return nrmse


def train_model(model, train_loader, val_loader, dataset, epochs=40):
    """Train the externalizing prediction model"""
    print("\n" + "="*80, flush=True)
    print("🔥 Training Externalizing Prediction Model", flush=True)
    print("="*80, flush=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}", flush=True)
    print(f"   Input shape: (batch, 129, 200)", flush=True)
    print(f"   Output shape: (batch, 1)", flush=True)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_nrmse = float('inf')
    patience_counter = 0
    patience = 10

    print(f"\n🚀 Starting training for {epochs} epochs...", flush=True)
    print(f"   Batch size: {train_loader.batch_size}", flush=True)
    print(f"   Train batches: {len(train_loader)}", flush=True)
    print(f"   Val batches: {len(val_loader)}", flush=True)
    print("", flush=True)

    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"\n{'='*80}", flush=True)
        print(f"📍 Epoch {epoch+1:2d}/{epochs}", flush=True)
        print(f"{'='*80}", flush=True)

        # Train
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        print(f"🔄 Training...", end=' ', flush=True)
        batch_counter = 0
        for data, labels in train_loader:
            batch_counter += 1
            if batch_counter % 10 == 0:
                print(f"[{batch_counter}/{len(train_loader)}]", end=' ', flush=True)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(outputs.detach().numpy().flatten())
            train_labels.extend(labels.numpy().flatten())

        print(" ✓", flush=True)

        # Validate
        print("🔍 Validating...", end=' ', flush=True)
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

        print(" ✓", flush=True)

        scheduler.step()

        # Compute metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_preds = np.array(train_preds)
        train_labels = np.array(train_labels)
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)

        # Competition metrics
        train_nrmse = compute_nrmse(train_labels, train_preds)
        val_nrmse = compute_nrmse(val_labels, val_preds)

        # Additional metrics
        val_corr, _ = pearsonr(val_preds, val_labels)
        val_mae = mean_absolute_error(val_labels, val_preds)
        val_rmse = np.sqrt(mean_squared_error(val_labels, val_preds))

        print(f"\n📊 Results:", flush=True)
        print(f"  Train: Loss={train_loss:.4f}, NRMSE={train_nrmse:.4f}", flush=True)
        print(f"  Val:   Loss={val_loss:.4f}, NRMSE={val_nrmse:.4f}", flush=True)
        print(f"  Val:   Corr={val_corr:.4f}, MAE={val_mae:.3f}, RMSE={val_rmse:.3f}", flush=True)

        # Competition target check
        if val_nrmse < 0.5:
            print("  🎯 Below competition target (NRMSE < 0.5)!", flush=True)

        # Save best model
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            patience_counter = 0

            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)

            # Save in competition format
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'nrmse': best_nrmse,
                'score_mean': dataset.score_mean,
                'score_std': dataset.score_std
            }, checkpoint_dir / "externalizing_model.pth")

            print(f"  💾 New best model! NRMSE={best_nrmse:.4f}", flush=True)
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{patience}", flush=True)

        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping after {epoch+1} epochs", flush=True)
            break

    return best_nrmse


def main():
    """Main training function"""
    start_time = time.time()

    print("\n📂 Loading dataset...")
    data_dir = Path("data/raw/hbn")

    # Competition format: 200 samples @ 100Hz
    dataset = ExternalizingDataset(data_dir=data_dir, segment_length=200, sampling_rate=100)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f"   Train segments: {train_size}")
    print(f"   Val segments: {val_size}")

    # Train model
    model = ExternalizingCNN()
    best_nrmse = train_model(model, train_loader, val_loader, dataset, epochs=40)

    total_time = time.time() - start_time

    # Final results
    print("\n" + "="*80)
    print("📊 FINAL RESULTS - Challenge 2 (Externalizing)")
    print("="*80)
    print(f"\nBest Validation NRMSE: {best_nrmse:.4f}")

    if best_nrmse < 0.5:
        print(f"✅ COMPETITION TARGET MET (NRMSE < 0.5)!")
    else:
        print(f"⚠️  Above competition target (target: < 0.5)")

    print(f"\nTotal training time: {total_time/60:.1f} minutes")

    # Save metadata
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "challenge2_externalizing.txt", 'w') as f:
        f.write(f"Challenge 2: Externalizing Factor Prediction\n")
        f.write(f"="*50 + "\n")
        f.write(f"Best NRMSE: {best_nrmse:.4f}\n")
        f.write(f"Training time: {total_time/60:.1f} minutes\n")
        f.write(f"Model: externalizing_model.pth\n")
        f.write(f"Input shape: (batch, 129, 200)\n")
        f.write(f"Output shape: (batch, 1)\n")

    print("\n💾 Results saved to: results/challenge2_externalizing.txt")
    print("💾 Model saved to: checkpoints/externalizing_model.pth")

    print("\n" + "="*80)
    print("✅ CHALLENGE 2 TRAINING COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Test model with submission.py format")
    print("2. Download CCD data for Challenge 1")
    print("3. Create submission package")


if __name__ == "__main__":
    main()
