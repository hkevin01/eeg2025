#!/usr/bin/env python3
"""
Simple, working training script for foundation model
Focuses on getting training running and completing successfully
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import time
import json
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("🚀 Simple Foundation Model Training")
print("=" * 80)
print(f"PyTorch version: {torch.__version__}")
print(f"Device: CPU (stable)")
print("=" * 80)

# Import dataset
try:
    from scripts.models.eeg_dataset_simple import SimpleEEGDataset
    print("✅ Dataset loader imported")
except Exception as e:
    print(f"❌ Failed to import dataset: {e}")
    sys.exit(1)

# Configuration
CONFIG = {
    'data_dir': Path(__file__).parent.parent / "data" / "raw" / "hbn",
    'output_dir': Path(__file__).parent.parent / "outputs",
    'checkpoint_dir': Path(__file__).parent.parent / "checkpoints",
    'log_dir': Path(__file__).parent.parent / "logs",

    # Model - Production size
    'hidden_dim': 128,
    'n_heads': 8,
    'n_layers': 4,
    'dropout': 0.1,

    # Training
    'batch_size': 16,
    'epochs': 10,  # Full training
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'save_every': 2,
}

# Create directories
for dir_path in [CONFIG['output_dir'], CONFIG['checkpoint_dir'], CONFIG['log_dir']]:
    dir_path.mkdir(exist_ok=True, parents=True)

print("\n📋 Configuration:")
for key, val in CONFIG.items():
    if not isinstance(val, Path):
        print(f"   {key}: {val}")

# Model
class SimpleTransformer(nn.Module):
    def __init__(self, n_channels=129, seq_len=1000,
                 hidden_dim=64, n_heads=4, n_layers=2,
                 n_classes=2, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(n_channels, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        # x: [batch, channels, time]
        x = x.transpose(1, 2)  # [batch, time, channels]
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.classifier(x)
        return x

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if (batch_idx + 1) % 50 == 0:
            acc = 100. * correct / total
            print(f"   [{batch_idx+1}/{len(loader)}] loss={loss.item():.4f}, acc={acc:.1f}%")

    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return total_loss / len(loader), 100. * correct / total

def main():
    start_time = time.time()
    device = torch.device('cpu')

    print("\n📂 Loading Dataset...")
    try:
        dataset = SimpleEEGDataset(data_dir=CONFIG['data_dir'], max_subjects=None)
        print(f"✅ Loaded {len(dataset)} windows")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                           shuffle=False, num_workers=2)

    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"   Batches: {len(train_loader)} train, {len(val_loader)} val")

    # Model
    print("\n🧠 Creating Model...")
    sample_data, _ = dataset[0]
    n_channels, seq_len = sample_data.shape
    print(f"   Input shape: {n_channels} channels × {seq_len} timepoints")

    model = SimpleTransformer(
        n_channels=n_channels,
        seq_len=seq_len,
        hidden_dim=CONFIG['hidden_dim'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        dropout=CONFIG['dropout']
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,} (~{n_params*4/(1024**2):.1f} MB)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CONFIG['learning_rate'],
                                  weight_decay=CONFIG['weight_decay'])

    # Training
    print("\n" + "=" * 80)
    print("🏋️  Training Started")
    print("=" * 80)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(CONFIG['epochs']):
        epoch_start = time.time()

        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"{'='*80}")

        # Train
        print(f"\n📚 Training...")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        print(f"\n🔍 Validating...")
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        # Record
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train: loss={train_loss:.4f}, acc={train_acc:.1f}%")
        print(f"   Val:   loss={val_loss:.4f}, acc={val_acc:.1f}%")
        print(f"   Time:  {epoch_time:.1f}s")

        # Save checkpoint
        if (epoch + 1) % CONFIG['save_every'] == 0 or val_loss < best_val_loss:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'config': CONFIG
            }

            path = CONFIG['checkpoint_dir'] / f"foundation_epoch_{epoch+1}.pth"
            torch.save(checkpoint, path)
            print(f"   💾 Saved: {path.name}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = CONFIG['checkpoint_dir'] / "foundation_best.pth"
                torch.save(checkpoint, best_path)
                print(f"   ⭐ New best model!")

    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("✅ Training Complete!")
    print(f"{'='*80}")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Final train acc: {train_acc:.1f}%")
    print(f"   Final val acc: {val_acc:.1f}%")
    print(f"   Checkpoints: {CONFIG['checkpoint_dir']}")

    # Save history
    history_path = CONFIG['log_dir'] / f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"   History: {history_path}")

    return model, history

if __name__ == "__main__":
    try:
        model, history = main()
        print("\n✅ Script completed successfully!")
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
