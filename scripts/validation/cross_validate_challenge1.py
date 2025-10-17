#!/usr/bin/env python3
"""
5-Fold Cross-Validation for Challenge 1
========================================
Tests model robustness across different data splits.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HIP_VISIBLE_DEVICES'] = ''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import time

print("="*80, flush=True)
print("🔄 5-FOLD CROSS-VALIDATION - CHALLENGE 1", flush=True)
print("="*80, flush=True)

# Import dataset and model from training script
import sys
sys.path.insert(0, str(Path(__file__).parent))

from train_challenge1_response_time import (
    ResponseTimeDataset,
    ResponseTimeCNN,
    compute_nrmse
)

def train_fold(model, train_loader, val_loader, epochs=30):
    """Train model for one fold"""
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_nrmse = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for data, labels in val_loader:
                outputs = model(data)
                val_preds.extend(outputs.numpy().flatten())
                val_labels.extend(labels.numpy().flatten())

        val_nrmse = compute_nrmse(np.array(val_labels), np.array(val_preds))

        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        scheduler.step()

    return best_nrmse

def main():
    """Run 5-fold cross-validation"""
    start_time = time.time()

    print("\n📂 Loading dataset...", flush=True)
    data_dir = Path("data/raw/hbn_ccd_mini")
    dataset = ResponseTimeDataset(data_dir=data_dir, segment_length=200, sampling_rate=100)

    print(f"   Total samples: {len(dataset)}", flush=True)

    # 5-fold cross-validation
    n_folds = 5
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []

    print(f"\n🔄 Running {n_folds}-fold cross-validation...", flush=True)
    print("="*80, flush=True)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(range(len(dataset)))):
        print(f"\n📍 Fold {fold+1}/{n_folds}", flush=True)
        print(f"   Train samples: {len(train_ids)}", flush=True)
        print(f"   Val samples: {len(val_ids)}", flush=True)

        # Create data loaders
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)

        train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)

        # Create fresh model
        model = ResponseTimeCNN()

        # Train
        print("   🔥 Training...", end=' ', flush=True)
        fold_start = time.time()
        best_nrmse = train_fold(model, train_loader, val_loader, epochs=30)
        fold_time = time.time() - fold_start
        print(f"✓ ({fold_time:.1f}s)", flush=True)

        print(f"   📊 Best NRMSE: {best_nrmse:.4f}", flush=True)

        if best_nrmse < 0.5:
            print(f"   ✅ Below target (0.5)", flush=True)
        else:
            print(f"   ⚠️  Above target (0.5)", flush=True)

        fold_results.append(best_nrmse)

    # Summary
    total_time = time.time() - start_time
    fold_results = np.array(fold_results)

    print("\n" + "="*80, flush=True)
    print("📊 CROSS-VALIDATION RESULTS", flush=True)
    print("="*80, flush=True)

    print(f"\nFold Results:", flush=True)
    for i, nrmse in enumerate(fold_results):
        status = "✅" if nrmse < 0.5 else "⚠️"
        print(f"   Fold {i+1}: {nrmse:.4f} {status}", flush=True)

    print(f"\n📈 Statistics:", flush=True)
    print(f"   Mean NRMSE: {fold_results.mean():.4f}", flush=True)
    print(f"   Std NRMSE:  {fold_results.std():.4f}", flush=True)
    print(f"   Min NRMSE:  {fold_results.min():.4f}", flush=True)
    print(f"   Max NRMSE:  {fold_results.max():.4f}", flush=True)

    print(f"\n⏱️  Total time: {total_time/60:.1f} minutes", flush=True)

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "challenge1_crossval.txt", 'w') as f:
        f.write("Challenge 1: 5-Fold Cross-Validation\n")
        f.write("="*50 + "\n\n")
        for i, nrmse in enumerate(fold_results):
            f.write(f"Fold {i+1}: {nrmse:.4f}\n")
        f.write(f"\nMean: {fold_results.mean():.4f}\n")
        f.write(f"Std:  {fold_results.std():.4f}\n")
        f.write(f"Min:  {fold_results.min():.4f}\n")
        f.write(f"Max:  {fold_results.max():.4f}\n")

    print(f"\n💾 Results saved to: results/challenge1_crossval.txt", flush=True)

    # Final assessment
    mean_nrmse = fold_results.mean()
    if mean_nrmse < 0.5:
        print(f"\n✅ ROBUST MODEL: Mean NRMSE {mean_nrmse:.4f} < 0.5", flush=True)
    else:
        print(f"\n⚠️  Model needs improvement: Mean NRMSE {mean_nrmse:.4f} > 0.5", flush=True)

    print("\n" + "="*80, flush=True)
    print("✅ CROSS-VALIDATION COMPLETE!", flush=True)
    print("="*80, flush=True)

if __name__ == "__main__":
    main()
