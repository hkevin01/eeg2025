#!/usr/bin/env python3
"""
Challenge 1: Response Time Prediction (CCD Task)
================================================
Competition-specific trainer for predicting response time from CCD task.

Requirements:
- Input: (batch, 129, 200) - 129 channels, 200 samples @ 100Hz
- Output: (batch, 1) - response time in seconds
- Metric: NRMSE (lower is better, target < 0.5)
"""
import os
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

print("="*80, flush=True)
print("🎯 CHALLENGE 1: RESPONSE TIME PREDICTION (CCD TASK)", flush=True)
print("="*80, flush=True)
print("Competition: https://eeg2025.github.io/", flush=True)
print("Metric: NRMSE (target < 0.5)", flush=True)
print("Device: CPU", flush=True)
print("="*80, flush=True)


class ResponseTimeDataset(Dataset):
    """EEG dataset for response time prediction from CCD task"""

    def __init__(self, data_dir, segment_length=200, sampling_rate=100):
        """
        Args:
            data_dir: Path to HBN data with CCD
            segment_length: 200 samples (2 seconds @ 100Hz)
            sampling_rate: Target sampling rate (100Hz for competition)
        """
        self.data_dir = Path(data_dir)
        self.segment_length = segment_length
        self.target_sr = sampling_rate

        print("\n📋 Loading CCD data...", flush=True)

        # Find subjects with CCD task
        self.segments = []
        self.response_times = []

        subject_dirs = sorted(self.data_dir.glob("sub-*"))
        print(f"   Found {len(subject_dirs)} subjects", flush=True)

        for subject_dir in subject_dirs:
            subject_id = subject_dir.name
            eeg_dir = subject_dir / "eeg"

            if not eeg_dir.exists():
                continue

            # Find CCD EEG files (.bdf format from competition data)
            ccd_files = list(eeg_dir.glob("*contrastChangeDetection*.bdf"))
            if not ccd_files:
                continue

            # Process each CCD run
            for eeg_file in ccd_files:
                try:
                    # Load EEG
                    raw = mne.io.read_raw_bdf(eeg_file, preload=True, verbose=False)

                    # Resample to 100Hz if needed
                    if raw.info['sfreq'] != self.target_sr:
                        raw.resample(self.target_sr, verbose=False)

                    data = raw.get_data()

                    # Ensure 129 channels
                    if data.shape[0] != 129:
                        print(f"   ⚠️  {subject_id}/{eeg_file.name}: {data.shape[0]} channels != 129, skipping", flush=True)
                        continue

                    # Load events to get response times
                    events_file = eeg_file.parent / eeg_file.name.replace('_eeg.bdf', '_events.tsv')
                    if not events_file.exists():
                        print(f"   ⚠️  {subject_id}: No events file, skipping", flush=True)
                        continue

                    events_df = pd.read_csv(events_file, sep='\t')

                    # Extract response times (trial start to button press)
                    # Look for trial start and button press events
                    trial_start_events = events_df[events_df['value'].str.contains('contrastTrial_start', case=False, na=False)]
                    button_press_events = events_df[events_df['value'].str.contains('buttonPress', case=False, na=False)]

                    if len(trial_start_events) == 0 or len(button_press_events) == 0:
                        print(f"   ⚠️  {subject_id}: No trial/button press events, skipping", flush=True)
                        continue

                    # Standardize per channel
                    data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)

                    # Create segments around trial start
                    for _, trial_event in trial_start_events.iterrows():
                        trial_time = trial_event['onset']

                        # Find corresponding button press
                        later_presses = button_press_events[button_press_events['onset'] > trial_time]
                        if len(later_presses) == 0:
                            continue

                        press_event = later_presses.iloc[0]
                        response_time = press_event['onset'] - trial_time

                        # Only use reasonable response times (0.1 to 5 seconds)
                        if response_time < 0.1 or response_time > 5.0:
                            continue

                        # Extract EEG segment starting from trial start
                        start_sample = int(trial_time * self.target_sr)
                        end_sample = start_sample + segment_length

                        if end_sample > data.shape[1]:
                            continue

                        segment = data[:, start_sample:end_sample]

                        self.segments.append(torch.FloatTensor(segment))
                        self.response_times.append(response_time)

                    if len(self.response_times) > 0:
                        print(f"   ✅ {subject_id}/{eeg_file.name}: {len(trial_start_events)} trials", flush=True)

                except Exception as e:
                    print(f"   ⚠️  {subject_id}: {e}", flush=True)
                    continue

        print(f"\n📊 Total segments: {len(self.segments)}", flush=True)

        if len(self.segments) == 0:
            raise ValueError("No valid CCD segments found! Check data directory and format.")

        # Compute statistics
        self.rt_array = np.array(self.response_times, dtype=np.float32)
        self.rt_mean = self.rt_array.mean()
        self.rt_std = self.rt_array.std()

        print(f"   Response Time: mean={self.rt_mean:.3f}s, std={self.rt_std:.3f}s", flush=True)
        print(f"   Range: [{self.rt_array.min():.3f}, {self.rt_array.max():.3f}]s", flush=True)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        rt = torch.FloatTensor([self.response_times[idx]])
        return segment, rt


class ResponseTimeCNN(nn.Module):
    """CNN for response time prediction (Challenge 1)

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
    """Train the response time prediction model"""
    print("\n" + "="*80, flush=True)
    print("🔥 Training Response Time Prediction Model", flush=True)
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
        print(f"\n{'='*80}", flush=True)
        print(f"📍 Epoch {epoch+1:2d}/{epochs}", flush=True)
        print(f"{'='*80}", flush=True)

        # Train
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        print("🔄 Training...", end=' ', flush=True)
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
        print(f"  Val:   Corr={val_corr:.4f}, MAE={val_mae:.3f}s, RMSE={val_rmse:.3f}s", flush=True)

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
                'rt_mean': dataset.rt_mean,
                'rt_std': dataset.rt_std
            }, checkpoint_dir / "response_time_model.pth")

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

    print("\n📂 Loading dataset...", flush=True)
    data_dir = Path("data/raw/hbn_ccd_mini")

    # Competition format: 200 samples @ 100Hz
    dataset = ResponseTimeDataset(data_dir=data_dir, segment_length=200, sampling_rate=100)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f"   Train segments: {train_size}", flush=True)
    print(f"   Val segments: {val_size}", flush=True)

    # Train model
    model = ResponseTimeCNN()
    best_nrmse = train_model(model, train_loader, val_loader, dataset, epochs=40)

    total_time = time.time() - start_time

    # Final results
    print("\n" + "="*80, flush=True)
    print("📊 FINAL RESULTS - Challenge 1 (Response Time)", flush=True)
    print("="*80, flush=True)
    print(f"\nBest Validation NRMSE: {best_nrmse:.4f}", flush=True)

    if best_nrmse < 0.5:
        print("✅ COMPETITION TARGET MET (NRMSE < 0.5)!", flush=True)
    else:
        print("⚠️  Above competition target (target: < 0.5)", flush=True)

    print(f"\nTotal training time: {total_time/60:.1f} minutes", flush=True)

    # Save metadata
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "challenge1_response_time.txt", 'w') as f:
        f.write("Challenge 1: Response Time Prediction\n")
        f.write("="*50 + "\n")
        f.write(f"Best NRMSE: {best_nrmse:.4f}\n")
        f.write(f"Training time: {total_time/60:.1f} minutes\n")
        f.write(f"Model: response_time_model.pth\n")
        f.write(f"Input shape: (batch, 129, 200)\n")
        f.write(f"Output shape: (batch, 1)\n")

    print("\n💾 Results saved to: results/challenge1_response_time.txt", flush=True)
    print("💾 Model saved to: checkpoints/response_time_model.pth", flush=True)

    print("\n" + "="*80, flush=True)
    print("✅ CHALLENGE 1 TRAINING COMPLETE!", flush=True)
    print("="*80, flush=True)
    print("\nNext steps:", flush=True)
    print("1. Convert checkpoint to competition format:", flush=True)
    print("   python3 -c \"import torch; cp=torch.load('checkpoints/response_time_model.pth', map_location='cpu'); torch.save(cp['model_state_dict'], 'weights_challenge_1.pt'); print('✅ Created weights_challenge_1.pt')\"", flush=True)
    print("2. Test both models: python3 scripts/test_submission_quick.py", flush=True)
    print("3. Create submission package and upload to Codabench!", flush=True)


if __name__ == "__main__":
    main()
