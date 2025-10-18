#!/usr/bin/env python3
"""
Robust TCN Training on ACTUAL Competition Data (R1-R5)
Survives crashes, auto-saves, uses competition dataset format
"""

import os
import sys
import json
import time
import signal
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# Force CPU for stability
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HIP_VISIBLE_DEVICES'] = ''

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import competition data loader
from eegdash import EEGChallengeDataset
from braindecode.preprocessing import Preprocessor, preprocess
from braindecode.preprocessing import create_windows_from_events
from eegdash.hbn.windows import annotate_trials_with_target, add_aux_anchors, add_extras_columns

# Import TCN model
from improvements.all_improvements import TCN_EEG

print("="*80)
print("🧠 ROBUST TCN TRAINING ON COMPETITION DATA (R1-R5)")
print("="*80)
print(f"Start time: {datetime.now()}")
print(f"Project root: {PROJECT_ROOT}")
print()

# Configuration
CONFIG = {
    'model': {
        'num_channels': 129,
        'num_outputs': 1,
        'num_filters': 48,
        'kernel_size': 7,
        'num_levels': 5,
        'dropout': 0.3
    },
    'training': {
        'batch_size': 16,
        'accumulation_steps': 2,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'epochs': 100,
        'patience': 15,
        'num_workers': 0
    },
    'data': {
        'train_releases': ['R1', 'R2', 'R3'],  # Competition training data
        'val_releases': ['R4'],                 # Competition validation data
        'task': 'contrastChangeDetection',
        'mini': False,  # Use FULL dataset, not mini
        'cache_dir': 'data/raw',
        'epoch_len_s': 2.0,
        'shift_after_stim': 0.5,
        'sfreq': 100,
        'max_datasets_per_release': 50  # Limit for faster iteration
    },
    'checkpoint': {
        'save_dir': 'checkpoints',
        'save_every': 5
    }
}

# Setup checkpoint directory
CHECKPOINT_DIR = PROJECT_ROOT / CONFIG['checkpoint']['save_dir']
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Global state for graceful shutdown
SHOULD_STOP = False

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    global SHOULD_STOP
    print(f"\n⚠️  Received signal {signum}, will stop after current epoch...")
    SHOULD_STOP = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class CompetitionEEGDataset(Dataset):
    """Dataset for competition EEG data with proper preprocessing"""

    def __init__(self, releases, task='contrastChangeDetection', mini=False,
                 cache_dir='data/raw', max_datasets_per_release=None):
        self.releases = releases
        self.samples = []

        print(f"📂 Loading competition data from releases: {releases}")
        print(f"   Task: {task}")
        print(f"   Mini: {mini}")
        print(f"   Max datasets per release: {max_datasets_per_release or 'unlimited'}")
        print()

        for release in releases:
            print(f"📦 Processing release: {release}")
            release_start = time.time()

            try:
                # Load dataset using competition format
                dataset = EEGChallengeDataset(
                    release=release,
                    mini=mini,
                    query=dict(task=task),
                    cache_dir=Path(cache_dir)
                )

                print(f"   Found {len(dataset.datasets)} datasets")

                # Limit datasets if specified
                if max_datasets_per_release and len(dataset.datasets) > max_datasets_per_release:
                    dataset.datasets = dataset.datasets[:max_datasets_per_release]
                    print(f"   Limited to {max_datasets_per_release} datasets")

                # Filter corrupted datasets
                valid_datasets = []
                for idx, ds in enumerate(dataset.datasets):
                    try:
                        _ = ds.raw.n_times
                        valid_datasets.append(ds)
                    except Exception:
                        continue

                print(f"   Valid datasets: {len(valid_datasets)}")

                if len(valid_datasets) == 0:
                    print(f"   ⚠️  No valid datasets in {release}, skipping...")
                    continue

                # Create new dataset with valid datasets only
                from braindecode.datasets import BaseConcatDataset
                dataset = BaseConcatDataset(valid_datasets)

                # Preprocess: annotate trials with response times
                print(f"   Preprocessing...")
                preprocessors = [
                    Preprocessor(
                        annotate_trials_with_target,
                        apply_on_array=False,
                        target_field="rt_from_stimulus",
                        epoch_length=CONFIG['data']['epoch_len_s'],
                        require_stimulus=True,
                        require_response=True,
                    ),
                    Preprocessor(add_aux_anchors, apply_on_array=False),
                ]

                preprocess(dataset, preprocessors, n_jobs=-1)

                # Create windows from events
                print("   Creating windows...")
                windows_dataset = create_windows_from_events(
                    dataset,
                    mapping={"contrast_trial_start": 0},
                    trial_start_offset_samples=int(CONFIG['data']['shift_after_stim'] * CONFIG['data']['sfreq']),
                    trial_stop_offset_samples=int((CONFIG['data']['shift_after_stim'] + CONFIG['data']['epoch_len_s']) * CONFIG['data']['sfreq']),
                    window_size_samples=int(CONFIG['data']['epoch_len_s'] * CONFIG['data']['sfreq']),
                    window_stride_samples=CONFIG['data']['sfreq'],
                    preload=True,
                )

                print(f"   Total windows: {len(windows_dataset)}")

                if len(windows_dataset) == 0:
                    print(f"   ⚠️  No windows created for {release}, skipping...")
                    continue

                # CRITICAL: Use add_extras_columns to inject trial metadata
                print("   Injecting trial metadata (response times)...")
                try:
                    windows_dataset = add_extras_columns(
                        windows_dataset,
                        dataset,
                        desc="contrast_trial_start",
                        keys=("rt_from_stimulus", "target", "correct")
                    )
                    print("   ✓ Metadata injected successfully")
                except Exception as e:
                    print(f"   ⚠️  Metadata injection failed, trying minimal keys: {e}")
                    try:
                        windows_dataset = add_extras_columns(
                            windows_dataset,
                            dataset,
                            desc="contrast_trial_start",
                            keys=("rt_from_stimulus",)
                        )
                        print("   ✓ Metadata injected (minimal)")
                    except Exception as e2:
                        print(f"   ❌ Failed to inject metadata: {e2}")
                        continue

                # Extract samples with response times
                print("   Extracting samples with response times...")
                metadata = windows_dataset.get_metadata()
                initial_sample_count = len(self.samples)

                # Debug: Check metadata columns
                print(f"   Metadata columns: {metadata.columns.tolist()}")
                if 'rt_from_stimulus' in metadata.columns:
                    rt_stats = metadata['rt_from_stimulus'].describe()
                    print(f"   RT stats: count={rt_stats['count']}, mean={rt_stats['mean']:.3f}s, std={rt_stats['std']:.3f}s")
                    valid_rts = metadata['rt_from_stimulus'].notna().sum()
                    print(f"   Valid RTs: {valid_rts}/{len(metadata)}")
                else:
                    print(f"   ❌ rt_from_stimulus NOT in metadata!")
                    print(f"   Available: {metadata.columns.tolist()}")
                    continue

                # Debug first sample
                if len(windows_dataset) > 0:
                    try:
                        X_test, y_test, idx_test = windows_dataset[0]
                        print(f"   First sample shape: X={X_test.shape}, y={y_test}, window_ind={idx_test}")
                        # window_ind is actually [i_trial, i_start, i_stop]
                        i_trial = idx_test[0] if isinstance(idx_test, (list, np.ndarray)) else idx_test
                        print(f"   First RT (trial {i_trial}): {metadata.iloc[i_trial]['rt_from_stimulus']}")
                    except Exception as e:
                        print(f"   ⚠️  Error accessing first sample: {e}")

                error_count = 0
                for i in range(len(windows_dataset)):
                    try:
                        X, y, window_ind = windows_dataset[i]

                        # Get response time from metadata
                        # window_ind is [i_trial, i_start_sample, i_stop_sample]
                        # We need the trial index (first element)
                        i_trial = window_ind[0] if isinstance(window_ind, (list, np.ndarray)) else window_ind
                        rt = metadata.iloc[i_trial]['rt_from_stimulus']

                        if rt is None or np.isnan(rt) or rt <= 0:
                            continue

                        # Ensure correct shape (129 channels, 200 samples)
                        if X.shape[0] < 129:
                            padding = np.zeros((129 - X.shape[0], X.shape[1]))
                            X = np.vstack([X, padding])
                        elif X.shape[0] > 129:
                            X = X[:129, :]

                        # Normalize
                        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

                        self.samples.append((X.astype(np.float32), float(rt)))

                    except Exception as e:
                        error_count += 1
                        if error_count <= 3:  # Show first 3 errors
                            print(f"   ⚠️  Sample {i} error: {e}")
                        continue

                samples_added = len(self.samples) - initial_sample_count
                release_time = time.time() - release_start
                print(f"   ✅ Extracted {samples_added} samples from {release} ({release_time:.1f}s)")
                print()

            except Exception as e:
                print(f"   ❌ Error loading {release}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"✅ Total samples loaded: {len(self.samples)}")
        print()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, target = self.samples[idx]
        return torch.FloatTensor(data), torch.FloatTensor([target])


def save_checkpoint(epoch, model, optimizer, scheduler, val_loss, history, config, filepath):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'history': history,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }

    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint saved: {filepath}")


def train():
    """Main training function"""
    global SHOULD_STOP

    print("📦 Initializing training...")
    print()

    # Device selection
    device = torch.device('cpu')
    print(f"Device: {device}")
    print()

    # Create datasets
    print("📊 Loading training data...")
    train_dataset = CompetitionEEGDataset(
        releases=CONFIG['data']['train_releases'],
        task=CONFIG['data']['task'],
        mini=CONFIG['data']['mini'],
        cache_dir=CONFIG['data']['cache_dir'],
        max_datasets_per_release=CONFIG['data']['max_datasets_per_release']
    )

    print("📊 Loading validation data...")
    val_dataset = CompetitionEEGDataset(
        releases=CONFIG['data']['val_releases'],
        task=CONFIG['data']['task'],
        mini=CONFIG['data']['mini'],
        cache_dir=CONFIG['data']['cache_dir'],
        max_datasets_per_release=CONFIG['data']['max_datasets_per_release']
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("❌ No data loaded! Check data directory and releases.")
        return

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['training']['batch_size'],
        shuffle=True,
        num_workers=CONFIG['training']['num_workers']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['training']['batch_size'],
        shuffle=False,
        num_workers=CONFIG['training']['num_workers']
    )

    # Create model
    print("🔨 Creating TCN model...")
    model = TCN_EEG(
        num_channels=CONFIG['model']['num_channels'],
        num_outputs=CONFIG['model']['num_outputs'],
        num_filters=CONFIG['model']['num_filters'],
        kernel_size=CONFIG['model']['kernel_size'],
        num_levels=CONFIG['model']['num_levels'],
        dropout=CONFIG['model']['dropout']
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print()

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['training']['learning_rate'],
        weight_decay=CONFIG['training']['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    criterion = nn.MSELoss()

    # Training state
    history = []
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    print("="*80)
    print("🚀 Starting Training on Competition Data (R1-R3 → R4)")
    print("="*80)
    print()

    try:
        for epoch in range(1, CONFIG['training']['epochs'] + 1):
            if SHOULD_STOP:
                print("\n⚠️  Stopping training (signal received)")
                break

            print(f"{'='*80}")
            print(f"Epoch {epoch}/{CONFIG['training']['epochs']}")
            print(f"{'='*80}")

            # Training
            model.train()
            train_loss = 0.0
            train_batches = 0

            optimizer.zero_grad()

            for batch_idx, (data, target) in enumerate(train_loader, 1):
                data = data.to(device)
                target = target.to(device)

                output = model(data)
                loss = criterion(output, target)
                loss = loss / CONFIG['training']['accumulation_steps']
                loss.backward()

                if batch_idx % CONFIG['training']['accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss += loss.item() * CONFIG['training']['accumulation_steps']
                train_batches += 1

                if batch_idx % 20 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item() * CONFIG['training']['accumulation_steps']:.6f}")

            train_loss /= train_batches

            # Validation
            model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for data, target in val_loader:
                    data = data.to(device)
                    target = target.to(device)

                    output = model(data)
                    loss = criterion(output, target)

                    val_loss += loss.item()
                    val_batches += 1

            val_loss /= val_batches

            # Update scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Log metrics
            print(f"\n📈 Train Loss: {train_loss:.6f}")
            print(f"📉 Val Loss:   {val_loss:.6f}")
            print(f"🎓 Learning rate: {current_lr:.6f}")

            # Save history
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': current_lr
            })

            # Save periodic checkpoint
            if epoch % CONFIG['checkpoint']['save_every'] == 0:
                checkpoint_path = CHECKPOINT_DIR / f"challenge1_tcn_competition_epoch{epoch}.pth"
                save_checkpoint(epoch, model, optimizer, scheduler, val_loss, history, CONFIG, checkpoint_path)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                best_path = CHECKPOINT_DIR / "challenge1_tcn_competition_best.pth"
                save_checkpoint(epoch, model, optimizer, scheduler, val_loss, history, CONFIG, best_path)
                print(f"✅ New best model! (Val Loss: {val_loss:.6f})")
            else:
                patience_counter += 1
                print(f"⏳ Patience: {patience_counter}/{CONFIG['training']['patience']}")

            # Early stopping
            if patience_counter >= CONFIG['training']['patience']:
                print("\n⛔ Early stopping triggered!")
                break

            print()

    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Save final checkpoint
        final_path = CHECKPOINT_DIR / "challenge1_tcn_competition_final.pth"
        save_checkpoint(epoch, model, optimizer, scheduler, val_loss, history, CONFIG, final_path)

        # Save history
        history_path = CHECKPOINT_DIR / "challenge1_tcn_competition_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"📊 History saved: {history_path}")

    print("\n" + "="*80)
    print("🎉 Training Complete!")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Total epochs: {len(history)}")
    print(f"Training data: R1, R2, R3 (Competition releases)")
    print(f"Validation data: R4 (Competition release)")
    print(f"End time: {datetime.now()}")
    print()


if __name__ == "__main__":
    train()
