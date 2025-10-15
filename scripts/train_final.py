#!/usr/bin/env python3
"""
Final Training with Real Age Labels
====================================
Train models with real age data from participants.tsv
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
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.models.eeg_dataset_age import AgeEEGDataset

print("="*80)
print("🎯 FINAL TRAINING WITH REAL AGE LABELS")
print("="*80)
print("Device: CPU")
print("="*80)


class SimpleCNN(nn.Module):
    """Simple CNN baseline"""
    def __init__(self, n_channels=129):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output.squeeze(-1)


class ImprovedCNN(nn.Module):
    """Improved multi-scale CNN"""
    def __init__(self, n_channels=129):
        super().__init__()
        
        # Multi-scale branches
        self.short = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.medium = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.long = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=31, padding=15),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Combine
        self.combine = nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=7, padding=3),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        short = self.short(x)
        medium = self.medium(x)
        long = self.long(x)
        
        combined = torch.cat([short, medium, long], dim=1)
        features = self.combine(combined)
        output = self.classifier(features)
        return output.squeeze(-1)


def train_model(model, train_loader, val_loader, dataset, epochs=15, lr=1e-3, model_name="model"):
    """Train a model and evaluate"""
    print(f"\n{'='*80}")
    print(f"🔥 Training {model_name}")
    print(f"{'='*80}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_mae = float('inf')
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
            labels = labels.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                labels = labels.squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(outputs.numpy())
                val_labels.extend(labels.numpy())
        
        scheduler.step()
        
        # Metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_corr, _ = pearsonr(train_preds, train_labels)
        val_corr, _ = pearsonr(val_preds, val_labels)
        
        # Denormalize for MAE in years
        train_preds_years = [dataset.denormalize_age(p) for p in train_preds]
        train_labels_years = [dataset.denormalize_age(l) for l in train_labels]
        val_preds_years = [dataset.denormalize_age(p) for p in val_preds]
        val_labels_years = [dataset.denormalize_age(l) for l in val_labels]
        
        train_mae = mean_absolute_error(train_labels_years, train_preds_years)
        val_mae = mean_absolute_error(val_labels_years, val_preds_years)
        
        print(f"Epoch {epoch+1:2d}/{epochs}")
        print(f"  Train: Loss={train_loss:.4f}, Corr={train_corr:.4f}, MAE={train_mae:.2f}yr")
        print(f"  Val:   Loss={val_loss:.4f}, Corr={val_corr:.4f}, MAE={val_mae:.2f}yr")
        
        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            best_corr = val_corr
            patience_counter = 0
            
            checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'mae': best_mae,
                'corr': best_corr
            }, checkpoint_dir / f"{model_name}.pth")
            print(f"  💾 New best model! MAE={best_mae:.2f}yr, Corr={best_corr:.4f}")
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{patience}")
        
        print()
        
        if patience_counter >= patience:
            print(f"⏹️  Early stopping after {epoch+1} epochs")
            break
    
    return best_mae, best_corr


def main():
    """Main training function"""
    start_time = time.time()
    
    print("\n📂 Loading dataset with real age labels...")
    data_dir = Path(__file__).parent.parent / "data" / "raw" / "hbn"
    
    # Load all available subjects
    dataset = AgeEEGDataset(data_dir=data_dir, max_subjects=None)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total segments: {len(dataset)}")
    print(f"   Age range: {dataset.age_min:.1f} - {dataset.age_max:.1f} years")
    print(f"   Mean age: {dataset.ages_array.mean():.1f} years")
    print(f"   Std age: {dataset.ages_array.std():.1f} years")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"   Train segments: {train_size}")
    print(f"   Val segments: {val_size}")
    
    # Train Simple CNN
    simple_model = SimpleCNN()
    simple_mae, simple_corr = train_model(
        simple_model, train_loader, val_loader, dataset,
        epochs=15, lr=1e-3, model_name="simple_cnn_age"
    )
    
    # Train Improved CNN
    improved_model = ImprovedCNN()
    improved_mae, improved_corr = train_model(
        improved_model, train_loader, val_loader, dataset,
        epochs=15, lr=5e-4, model_name="improved_cnn_age"
    )
    
    total_time = time.time() - start_time
    
    # Final results
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    print(f"\nSimple CNN:")
    print(f"  MAE: {simple_mae:.2f} years")
    print(f"  Correlation: {simple_corr:.4f}")
    
    print(f"\nImproved CNN:")
    print(f"  MAE: {improved_mae:.2f} years")
    print(f"  Correlation: {improved_corr:.4f}")
    
    improvement_mae = ((simple_mae - improved_mae) / simple_mae) * 100
    improvement_corr = ((improved_corr - simple_corr) / abs(simple_corr)) * 100 if simple_corr != 0 else 0
    
    print(f"\nImprovement:")
    print(f"  MAE: {improvement_mae:+.1f}%")
    print(f"  Correlation: {improvement_corr:+.1f}%")
    
    print(f"\nTotal training time: {total_time/60:.1f} minutes")
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    results_df = pd.DataFrame({
        'model': ['Simple CNN', 'Improved CNN'],
        'mae_years': [simple_mae, improved_mae],
        'correlation': [simple_corr, improved_corr]
    })
    results_df.to_csv(results_dir / "final_results.csv", index=False)
    
    print("\n💾 Results saved to: results/final_results.csv")
    print("💾 Models saved to: checkpoints/")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
