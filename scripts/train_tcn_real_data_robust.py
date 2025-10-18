#!/usr/bin/env python3
"""
Robust TCN Training on Real EEG Data
Survives crashes, auto-saves, can resume from checkpoint
"""

import os
import sys
import json
import time
import signal
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mne

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

# Import TCN model from improvements
from improvements.all_improvements import TCN_EEG

print("="*80)
print("🧠 ROBUST TCN TRAINING ON REAL EEG DATA")
print("="*80)
print(f"Start time: {datetime.now()}")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
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
        'batch_size': 8,
        'accumulation_steps': 4,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'epochs': 50,
        'patience': 10,
        'num_workers': 0,
        'resume_from': None  # Set to checkpoint path to resume
    },
    'data': {
        'data_dir': 'data/raw',
        'seq_len': 500,
        'max_samples_per_subject': 20,  # Limit samples for faster training
        'train_split': 0.8
    },
    'checkpoint': {
        'save_dir': 'checkpoints',
        'save_every': 5,  # Save every N epochs
        'keep_last': 3    # Keep last N checkpoints
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


class RealEEGDataset(Dataset):
    """Dataset for real EEG data with robust loading"""

    def __init__(self, data_dir, seq_len=500, max_samples_per_subject=20):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.max_samples_per_subject = max_samples_per_subject
        self.samples = []

        print(f"📁 Loading data from: {self.data_dir}")
        self._load_data()
        print(f"✅ Loaded {len(self.samples)} samples")

    def _load_data(self):
        """Load and preprocess EEG data"""

        # Look for BDF files
        bdf_files = list(self.data_dir.rglob("*.bdf"))
        print(f"Found {len(bdf_files)} BDF files")

        if not bdf_files:
            # Fallback to synthetic data
            print("⚠️  No BDF files found, generating synthetic data...")
            self._generate_synthetic_data()
            return

        # Process each file
        for bdf_file in bdf_files[:10]:  # Limit to first 10 subjects for speed
            try:
                print(f"  Processing: {bdf_file.name}...")

                # Load with MNE
                raw = mne.io.read_raw_bdf(str(bdf_file), preload=True, verbose=False)

                # Get data
                data = raw.get_data()  # Shape: (n_channels, n_times)
                sfreq = raw.info['sfreq']

                # Ensure we have enough channels
                if data.shape[0] < 64:
                    print(f"    Skipping (only {data.shape[0]} channels)")
                    continue

                # Pad to 129 channels if needed
                if data.shape[0] < 129:
                    padding = np.zeros((129 - data.shape[0], data.shape[1]))
                    data = np.vstack([data, padding])
                elif data.shape[0] > 129:
                    data = data[:129, :]

                # Normalize
                data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)

                # Create segments
                n_samples = min(data.shape[1] // self.seq_len, self.max_samples_per_subject)

                for i in range(n_samples):
                    start = i * self.seq_len
                    end = start + self.seq_len

                    segment = data[:, start:end].astype(np.float32)

                    # Generate synthetic response time (will be replaced with real labels)
                    response_time = np.random.uniform(0.3, 0.8)

                    self.samples.append((segment, response_time))

                print(f"    Added {n_samples} samples")

            except Exception as e:
                print(f"    Error loading {bdf_file.name}: {e}")
                continue

    def _generate_synthetic_data(self):
        """Generate synthetic data as fallback"""
        print("Generating 500 synthetic samples...")

        for i in range(500):
            # Create realistic EEG patterns
            t = np.linspace(0, 2, self.seq_len)
            data = np.zeros((129, self.seq_len), dtype=np.float32)

            for ch in range(129):
                # Alpha wave (8-13 Hz)
                alpha = np.sin(2 * np.pi * np.random.uniform(8, 13) * t)
                # Beta wave (13-30 Hz)
                beta = 0.5 * np.sin(2 * np.pi * np.random.uniform(13, 30) * t)
                # Theta wave (4-8 Hz)
                theta = 0.3 * np.sin(2 * np.pi * np.random.uniform(4, 8) * t)
                # Noise
                noise = 0.1 * np.random.randn(self.seq_len)

                data[ch] = alpha + beta + theta + noise

            # Normalize
            data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)

            # Synthetic response time
            response_time = 0.5 + 0.3 * np.random.randn()
            response_time = np.clip(response_time, 0.2, 1.0)

            self.samples.append((data, response_time))

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


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """Load training checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"✅ Checkpoint loaded from: {filepath}")
    print(f"   Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.6f}")

    return checkpoint


def train():
    """Main training function"""
    global SHOULD_STOP

    print("📦 Initializing training...")

    # Device selection
    device = torch.device('cpu')
    print(f"Device: {device}")
    print()

    # Create dataset
    print("📊 Creating dataset...")
    full_dataset = RealEEGDataset(
        data_dir=PROJECT_ROOT / CONFIG['data']['data_dir'],
        seq_len=CONFIG['data']['seq_len'],
        max_samples_per_subject=CONFIG['data']['max_samples_per_subject']
    )

    # Split into train/val
    train_size = int(CONFIG['data']['train_split'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['training']['batch_size'],
        shuffle=True,
        num_workers=CONFIG['training']['num_workers'],
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['training']['batch_size'],
        shuffle=False,
        num_workers=CONFIG['training']['num_workers'],
        pin_memory=False
    )

    # Create model
    print("🔨 Creating model...")
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

    # Resume from checkpoint if specified
    start_epoch = 1
    history = []
    best_val_loss = float('inf')
    patience_counter = 0

    if CONFIG['training']['resume_from']:
        checkpoint = load_checkpoint(
            CONFIG['training']['resume_from'],
            model, optimizer, scheduler
        )
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint['history']
        best_val_loss = checkpoint['val_loss']
        print()

    # Training loop
    print("="*80)
    print("🚀 Starting Training")
    print("="*80)
    print()

    try:
        for epoch in range(start_epoch, CONFIG['training']['epochs'] + 1):
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

                if batch_idx % 10 == 0:
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

            # Save checkpoint
            if epoch % CONFIG['checkpoint']['save_every'] == 0:
                checkpoint_path = CHECKPOINT_DIR / f"challenge1_tcn_real_epoch{epoch}.pth"
                save_checkpoint(epoch, model, optimizer, scheduler, val_loss, history, CONFIG, checkpoint_path)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                best_path = CHECKPOINT_DIR / "challenge1_tcn_real_best.pth"
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
        traceback.print_exc()

        # Save emergency checkpoint
        emergency_path = CHECKPOINT_DIR / f"challenge1_tcn_real_emergency_{int(time.time())}.pth"
        try:
            save_checkpoint(epoch, model, optimizer, scheduler, val_loss, history, CONFIG, emergency_path)
            print(f"💾 Emergency checkpoint saved: {emergency_path}")
        except:
            print("Failed to save emergency checkpoint")

    finally:
        # Save final checkpoint
        final_path = CHECKPOINT_DIR / "challenge1_tcn_real_final.pth"
        save_checkpoint(epoch, model, optimizer, scheduler, val_loss, history, CONFIG, final_path)

        # Save history
        history_path = CHECKPOINT_DIR / "challenge1_tcn_real_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"📊 History saved: {history_path}")

    print("\n" + "="*80)
    print("🎉 Training Complete!")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Total epochs: {len(history)}")
    print(f"End time: {datetime.now()}")
    print()


if __name__ == "__main__":
    train()
