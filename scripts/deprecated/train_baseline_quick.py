#!/usr/bin/env python3
"""
Quick Baseline Training
=======================
Train simple baseline models to establish performance benchmarks.
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
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.models.eeg_dataset_simple import SimpleEEGDataset

print("="*80)
print("🚀 BASELINE MODEL TRAINING")
print("="*80)
print(f"Device: CPU")
print("="*80)


class SimpleBaseline(nn.Module):
    """Simple baseline CNN"""
    def __init__(self, n_channels=129):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_data(max_samples=500):
    """Load dataset"""
    print("\n📂 Loading data...")

    data_dir = Path(__file__).parent.parent / "data" / "raw" / "hbn"
    participants_file = data_dir / "participants.tsv"

    if not participants_file.exists():
        raise FileNotFoundError("participants.tsv not found!")

    participants_df = pd.read_csv(participants_file, sep='\t')
    print(f"   Found {len(participants_df)} participants in metadata")

    # Load dataset
    dataset = SimpleEEGDataset(data_dir=data_dir, max_subjects=None)

    # Sample data
    valid_indices = list(range(min(len(dataset), max_samples)))
    print(f"   Using {len(valid_indices)} samples")

    # Split
    split_idx = int(0.8 * len(valid_indices))
    train_indices = valid_indices[:split_idx]
    val_indices = valid_indices[split_idx:]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

    print(f"   Train: {len(train_subset)}, Val: {len(val_subset)}")

    return train_loader, val_loader, participants_df


def train_pytorch_baseline(train_loader, val_loader, epochs=5):
    """Train PyTorch CNN baseline"""
    print("\n" + "="*80)
    print("🔥 Training PyTorch CNN Baseline")
    print("="*80)

    model = SimpleBaseline()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_corr = -1

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            labels = labels.float()  # Convert to float
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(outputs.detach().numpy())
            train_labels.extend(labels.numpy())

        # Validate
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for data, labels in val_loader:
                outputs = model(data)
                labels = labels.float()  # Convert to float
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(outputs.numpy())
                val_labels.extend(labels.numpy())

        # Metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_corr, _ = pearsonr(train_preds, train_labels)
        val_corr, _ = pearsonr(val_preds, val_labels)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train: Loss={train_loss:.4f}, Corr={train_corr:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Corr={val_corr:.4f}")

        if val_corr > best_corr:
            best_corr = val_corr
            # Save model
            checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_corr': best_corr
            }, checkpoint_dir / "baseline_cnn.pth")

    print(f"\n✅ Best validation correlation: {best_corr:.4f}")
    return best_corr


def train_sklearn_baselines(train_loader, val_loader):
    """Train sklearn baselines"""
    print("\n" + "="*80)
    print("🌲 Training Sklearn Baselines")
    print("="*80)

    # Collect data
    print("   Collecting training data...")
    X_train = []
    y_train = []
    for data, labels in train_loader:
        # Flatten EEG data
        X_train.append(data.numpy().reshape(data.shape[0], -1))
        y_train.append(labels.numpy())

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)

    print("   Collecting validation data...")
    X_val = []
    y_val = []
    for data, labels in val_loader:
        X_val.append(data.numpy().reshape(data.shape[0], -1))
        y_val.append(labels.numpy())

    X_val = np.vstack(X_val)
    y_val = np.concatenate(y_val)

    print(f"   Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # Take subset of features (too many features)
    if X_train.shape[1] > 1000:
        print(f"   Reducing features from {X_train.shape[1]} to 1000...")
        # Take every Nth feature
        step = X_train.shape[1] // 1000
        X_train = X_train[:, ::step][:, :1000]
        X_val = X_val[:, ::step][:, :1000]

    results = {}

    # 1. Linear Regression
    print("\n1️⃣  Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_val)
    lr_corr, _ = pearsonr(lr_preds, y_val)
    results['Linear Regression'] = lr_corr
    print(f"   Correlation: {lr_corr:.4f}")

    # 2. Random Forest
    print("\n2️⃣  Random Forest (n_estimators=50)...")
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_val)
    rf_corr, _ = pearsonr(rf_preds, y_val)
    results['Random Forest'] = rf_corr
    print(f"   Correlation: {rf_corr:.4f}")

    return results


def main():
    """Main training function"""
    start_time = time.time()

    # Load data
    train_loader, val_loader, participants_df = load_data(max_samples=500)

    # Train baselines
    print("\n" + "="*80)
    print("📊 TRAINING ALL BASELINES")
    print("="*80)

    results = {}

    # 1. PyTorch CNN
    cnn_corr = train_pytorch_baseline(train_loader, val_loader, epochs=5)
    results['Simple CNN'] = cnn_corr

    # 2. Sklearn models
    sklearn_results = train_sklearn_baselines(train_loader, val_loader)
    results.update(sklearn_results)

    # Summary
    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("📊 BASELINE RESULTS SUMMARY")
    print("="*80)

    print("\nModel Performance (Validation Correlation):")
    for model_name, corr in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model_name:20s}: {corr:.4f}")

    print(f"\nTotal training time: {total_time:.1f}s")

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    results_df = pd.DataFrame([
        {'model': name, 'validation_correlation': corr}
        for name, corr in results.items()
    ])
    results_df.to_csv(results_dir / "baseline_results.csv", index=False)

    print(f"\n💾 Results saved to results/baseline_results.csv")

    print("\n" + "="*80)
    print("✅ BASELINE TRAINING COMPLETE")
    print("="*80)

    # Mark as complete
    print("\n📝 Update your TODO:")
    print("  [x] Train baseline models")
    print("  [ ] Train improved models")


if __name__ == "__main__":
    main()
