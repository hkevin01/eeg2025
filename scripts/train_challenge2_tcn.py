#!/usr/bin/env python3
"""
Challenge 2: TCN for Externalizing Prediction
Train on R1-R3, validate on R4 (Competition data)
"""
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append('src')
sys.path.append('improvements')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from eegdash import EEGChallengeDataset
from braindecode.preprocessing import Preprocessor, preprocess
from braindecode.preprocessing import create_windows_from_events
from eegdash.hbn.windows import annotate_trials_with_target, add_aux_anchors, add_extras_columns

# Import TCN from improvements
from all_improvements import TCN_EEG

print("="*80)
print("🎯 CHALLENGE 2: TCN FOR EXTERNALIZING PREDICTION")
print("="*80)
print("Training: R1, R2, R3 | Validation: R4")
print("Task: RestingState EEG → Externalizing Score")
print("="*80)
print()

# Configuration
CONFIG = {
    'data': {
        'train_releases': ['R1', 'R2', 'R3'],
        'val_releases': ['R4'],
        'task': 'RestingState',  # Challenge 2 uses resting state
        'mini': False,
        'epoch_len_s': 2.0,
        'sfreq': 100,
        'max_datasets_per_release': 50
    },
    'model': {
        'num_filters': 48,
        'kernel_size': 7,
        'dropout': 0.3,
        'num_levels': 5
    },
    'training': {
        'batch_size': 16,
        'epochs': 100,
        'lr': 0.001,
        'patience': 15
    }
}


class Challenge2Dataset(Dataset):
    """Dataset for Challenge 2: Externalizing prediction from RestingState EEG"""

    def __init__(self, releases, config):
        self.samples = []

        print(f"\n📂 Loading Challenge 2 data from releases: {releases}")

        for release in releases:
            print(f"\n📦 Processing release: {release}")
            release_start = time.time()

            try:
                # Load RestingState data
                dataset = EEGChallengeDataset(
                    release=release,
                    mini=config['data']['mini'],
                    query=dict(task="RestingState"),
                    cache_dir=Path('data/raw')
                )

                print(f"   Datasets loaded: {len(dataset.datasets)}")

                # For RestingState, we just need to extract windows
                # The externalizing score is in the dataset metadata
                for ds_idx, ds in enumerate(dataset.datasets):
                    try:
                        raw = ds.raw

                        # Get externalizing score from metadata
                        if hasattr(ds, 'description') and 'externalizing' in ds.description:
                            externalizing = ds.description['externalizing']
                            if externalizing is None or np.isnan(externalizing):
                                continue
                        else:
                            continue

                        # Extract 2-second windows
                        n_samples = int(config['data']['epoch_len_s'] * config['data']['sfreq'])
                        data = raw.get_data()

                        # Ensure 129 channels
                        if data.shape[0] < 129:
                            padding = np.zeros((129 - data.shape[0], data.shape[1]))
                            data = np.vstack([data, padding])
                        elif data.shape[0] > 129:
                            data = data[:129, :]

                        # Create non-overlapping windows
                        for start_idx in range(0, data.shape[1] - n_samples + 1, n_samples):
                            window = data[:, start_idx:start_idx + n_samples]

                            # Normalize
                            window = (window - window.mean(axis=1, keepdims=True)) / (window.std(axis=1, keepdims=True) + 1e-8)

                            self.samples.append((window.astype(np.float32), float(externalizing)))

                    except Exception:
                        continue

                samples_added = len(self.samples)
                release_time = time.time() - release_start
                print(f"   ✅ Extracted {samples_added} samples from {release} ({release_time:.1f}s)")

            except Exception as e:
                print(f"   ❌ Error loading {release}: {e}")
                continue

        print(f"\n✅ Total samples loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        # Ensure Float32 for both X and y
        return torch.from_numpy(X).float(), torch.tensor(y, dtype=torch.float32)


def train_challenge2():
    """Train Challenge 2 model"""

    # Load data
    print("\n📊 Loading training data...")
    train_dataset = Challenge2Dataset(CONFIG['data']['train_releases'], CONFIG)

    print("\n📊 Loading validation data...")
    val_dataset = Challenge2Dataset(CONFIG['data']['val_releases'], CONFIG)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("❌ No data loaded! Exiting.")
        return

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['training']['batch_size'],
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['training']['batch_size'],
                           shuffle=False, num_workers=4)

    # Create model
    print("\n🔨 Creating TCN model...")
    model = TCN_EEG(
        num_channels=129,
        num_outputs=1,
        num_filters=CONFIG['model']['num_filters'],
        kernel_size=CONFIG['model']['kernel_size'],
        dropout=CONFIG['model']['dropout'],
        num_levels=CONFIG['model']['num_levels']
    )

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    device = torch.device('cpu')
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['training']['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    print("\n" + "="*80)
    print("🚀 Starting Training")
    print("="*80)

    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    Path('checkpoints').mkdir(exist_ok=True)

    for epoch in range(1, CONFIG['training']['epochs'] + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{CONFIG['training']['epochs']}")
        print(f"{'='*80}")

        # Training
        model.train()
        train_loss = 0.0

        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device).unsqueeze(1)

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.6f}")

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device).unsqueeze(1)
                output = model(X)
                loss = criterion(output, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Print metrics
        print(f"\n📈 Train Loss: {train_loss:.6f}")
        print(f"📉 Val Loss:   {val_loss:.6f}")
        print(f"🎓 Learning rate: {current_lr:.6f}")
        print(f"⏳ Patience: {patience_counter}/{CONFIG['training']['patience']}")

        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': current_lr
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }, 'checkpoints/challenge2_tcn_competition_best.pth')

            print("✅ New best model!")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= CONFIG['training']['patience']:
            print("\n⛔ Early stopping triggered!")
            break

        # Periodic checkpoints
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss
            }, f'checkpoints/challenge2_tcn_competition_epoch{epoch}.pth')

    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss
    }, 'checkpoints/challenge2_tcn_competition_final.pth')

    # Save history
    import json
    with open('checkpoints/challenge2_tcn_competition_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*80)
    print("🎉 Training Complete!")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Total epochs: {epoch}")
    print("="*80)


if __name__ == '__main__':
    train_challenge2()
