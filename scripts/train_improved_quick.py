#!/usr/bin/env python3
"""
Quick Improved Training
=======================
Train improved model with multi-scale architecture on small dataset.
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.models.eeg_dataset_simple import SimpleEEGDataset

print("="*80)
print("🚀 IMPROVED MODEL TRAINING (Quick)")
print("="*80)
print("Device: CPU")
print("="*80)


class ImprovedModel(nn.Module):
    """Improved multi-scale CNN"""
    def __init__(self, n_channels=129):
        super().__init__()
        
        # Multi-scale branches
        self.short_branch = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.medium_branch = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Combine and process
        self.combine = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # Multi-scale processing
        short = self.short_branch(x)
        medium = self.medium_branch(x)
        
        # Combine
        combined = torch.cat([short, medium], dim=1)
        features = self.combine(combined)
        
        # Classify
        output = self.classifier(features)
        return output.squeeze(-1)


def load_data(max_samples=300):
    """Load dataset quickly"""
    print("\n📂 Loading data...")
    
    data_dir = Path(__file__).parent.parent / "data" / "raw" / "hbn"
    dataset = SimpleEEGDataset(data_dir=data_dir, max_subjects=None)
    
    # Sample data
    valid_indices = list(range(min(len(dataset), max_samples)))
    print(f"   Using {len(valid_indices)} samples")
    
    # Split
    split_idx = int(0.8 * len(valid_indices))
    train_subset = Subset(dataset, valid_indices[:split_idx])
    val_subset = Subset(dataset, valid_indices[split_idx:])
    
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=2)
    
    print(f"   Train: {len(train_subset)}, Val: {len(val_subset)}")
    
    return train_loader, val_loader


def train_model(train_loader, val_loader, epochs=10):
    """Train improved model"""
    print("\n" + "="*80)
    print("🔥 Training Improved Model")
    print("="*80)
    
    model = ImprovedModel()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Learning rate scheduler with warmup
    num_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * num_steps)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, num_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"   Optimizer: AdamW (lr=1e-4, wd=1e-5)")
    print(f"   Scheduler: Warmup + Cosine")
    print()
    
    best_corr = -1
    patience_counter = 0
    patience = 5
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            labels = labels.float()
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
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
                labels = labels.float()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(outputs.numpy())
                val_labels.extend(labels.numpy())
        
        # Metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_corr, _ = pearsonr(train_preds, train_labels)
        val_corr, _ = pearsonr(val_preds, val_labels)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1:2d}/{epochs}")
        print(f"  Train: Loss={train_loss:.4f}, Corr={train_corr:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Corr={val_corr:.4f}")
        print(f"  LR:    {current_lr:.6f}")
        
        # Save best model
        if val_corr > best_corr:
            best_corr = val_corr
            patience_counter = 0
            checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_corr': best_corr
            }, checkpoint_dir / "improved_model.pth")
            print(f"  💾 New best model saved!")
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{patience}")
        
        print()
        
        # Early stopping
        if patience_counter >= patience:
            print(f"⏹️  Early stopping after {epoch+1} epochs")
            break
    
    print(f"✅ Best validation correlation: {best_corr:.4f}")
    return best_corr


def main():
    """Main training function"""
    start_time = time.time()
    
    # Load data
    train_loader, val_loader = load_data(max_samples=300)
    
    # Train model
    improved_corr = train_model(train_loader, val_loader, epochs=10)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("📊 TRAINING COMPLETE")
    print("="*80)
    print(f"Improved Model Correlation: {improved_corr:.4f}")
    print(f"Total training time: {total_time:.1f}s")
    
    # Compare with baseline
    print("\n📈 Comparison with Baseline:")
    try:
        results_file = Path(__file__).parent.parent / "results" / "baseline_results.csv"
        if results_file.exists():
            baseline_df = pd.read_csv(results_file)
            baseline_best = baseline_df['validation_correlation'].max()
            print(f"  Baseline best: {baseline_best:.4f}")
            print(f"  Improved:      {improved_corr:.4f}")
            improvement = ((improved_corr - baseline_best) / abs(baseline_best)) * 100
            print(f"  Change:        {improvement:+.1f}%")
    except:
        pass
    
    print("\n💾 Model saved to: checkpoints/improved_model.pth")
    
    print("\n📝 Update your TODO:")
    print("  [x] Train baseline models")
    print("  [x] Train improved models ← COMPLETE")
    print("  [ ] Apply preprocessing")
    print("  [ ] Use test-time augmentation")


if __name__ == "__main__":
    main()
