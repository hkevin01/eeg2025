#!/usr/bin/env python3
"""
Ensemble Training for Challenge 1
==================================
Train 3 models with different random seeds for ensemble prediction.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HIP_VISIBLE_DEVICES'] = ''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import numpy as np
import time

print("="*80, flush=True)
print("🎲 ENSEMBLE TRAINING - CHALLENGE 1 (3 Models)", flush=True)
print("="*80, flush=True)

from train_challenge1_improved import (
    AugmentedResponseTimeDataset,
    ImprovedResponseTimeCNN,
    compute_nrmse
)

def train_model(model, train_loader, val_loader, dataset, seed, epochs=40):
    """Train single model with specific seed"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_nrmse = float('inf')
    best_model_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

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
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        scheduler.step()

    return best_nrmse, best_model_state

def main():
    """Train ensemble of 3 models"""
    start_time = time.time()

    print("\n📂 Loading dataset...", flush=True)
    data_dir = Path("data/raw/hbn_ccd_mini")
    dataset = AugmentedResponseTimeDataset(data_dir=data_dir, segment_length=200, sampling_rate=100)

    print(f"   Total samples: {len(dataset)}", flush=True)

    # Random seeds for ensemble
    seeds = [42, 123, 456]
    ensemble_results = []
    model_states = []

    print(f"\n🎲 Training {len(seeds)} models with different seeds...", flush=True)
    print("="*80, flush=True)

    for i, seed in enumerate(seeds):
        print(f"\n📍 Model {i+1}/{len(seeds)} (seed={seed})", flush=True)

        # Set seed for data split
        torch.manual_seed(seed)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Create model
        model = ImprovedResponseTimeCNN()

        # Train
        print("   🔥 Training...", end=' ', flush=True)
        model_start = time.time()
        best_nrmse, best_state = train_model(model, train_loader, val_loader, dataset, seed, epochs=40)
        model_time = time.time() - model_start
        print(f"✓ ({model_time:.1f}s)", flush=True)

        print(f"   📊 Best NRMSE: {best_nrmse:.4f}", flush=True)

        if best_nrmse < 0.5:
            print(f"   ✅ Below target", flush=True)
        else:
            print(f"   ⚠️  Above target", flush=True)

        ensemble_results.append(best_nrmse)
        model_states.append(best_state)

        # Save individual model
        checkpoint_dir = Path("checkpoints/ensemble")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': best_state,
            'nrmse': best_nrmse,
            'seed': seed
        }, checkpoint_dir / f"response_time_model_seed{seed}.pth")
        print(f"   💾 Saved: checkpoints/ensemble/response_time_model_seed{seed}.pth", flush=True)

    # Test ensemble prediction
    print("\n" + "="*80, flush=True)
    print("📊 ENSEMBLE RESULTS", flush=True)
    print("="*80, flush=True)

    print(f"\nIndividual Model Performance:", flush=True)
    for i, (seed, nrmse) in enumerate(zip(seeds, ensemble_results)):
        status = "✅" if nrmse < 0.5 else "⚠️"
        print(f"   Model {i+1} (seed={seed}): {nrmse:.4f} {status}", flush=True)

    print(f"\n📈 Ensemble Statistics:", flush=True)
    mean_nrmse = np.mean(ensemble_results)
    std_nrmse = np.std(ensemble_results)
    print(f"   Mean NRMSE: {mean_nrmse:.4f}", flush=True)
    print(f"   Std NRMSE:  {std_nrmse:.4f}", flush=True)
    print(f"   Best NRMSE: {np.min(ensemble_results):.4f}", flush=True)

    total_time = time.time() - start_time
    print(f"\n⏱️  Total time: {total_time/60:.1f} minutes", flush=True)

    # Save ensemble info
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "challenge1_ensemble.txt", 'w') as f:
        f.write("Challenge 1: Ensemble Training (3 models)\n")
        f.write("="*50 + "\n\n")
        for i, (seed, nrmse) in enumerate(zip(seeds, ensemble_results)):
            f.write(f"Model {i+1} (seed={seed}): {nrmse:.4f}\n")
        f.write(f"\nMean: {mean_nrmse:.4f}\n")
        f.write(f"Std:  {std_nrmse:.4f}\n")
        f.write(f"Best: {np.min(ensemble_results):.4f}\n")
        f.write(f"\nEnsemble prediction: Average of 3 models\n")

    print(f"\n💾 Results saved to: results/challenge1_ensemble.txt", flush=True)

    # Select best single model for submission
    best_idx = np.argmin(ensemble_results)
    best_seed = seeds[best_idx]
    best_nrmse = ensemble_results[best_idx]

    print(f"\n🏆 BEST MODEL: seed={best_seed}, NRMSE={best_nrmse:.4f}", flush=True)
    print(f"\nℹ️  For ensemble inference, load all 3 models and average predictions", flush=True)

    print("\n" + "="*80, flush=True)
    print("✅ ENSEMBLE TRAINING COMPLETE!", flush=True)
    print("="*80, flush=True)

if __name__ == "__main__":
    main()
