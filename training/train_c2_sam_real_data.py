#!/usr/bin/env python3
"""
Challenge 2: SAM Training - Real EEG Data
==========================================
Predicting externalizing factor (p_factor) with SAM optimizer.

Key points:
- Task: contrastChangeDetection
- Target: p_factor (externalizing)
- Model: EEGNeX
- Optimizer: SAM (Sharpness-Aware Minimization)
- Data: Real EEG from R1-R5 releases
"""
import os
import sys
import time
import warnings
import math
import random
import argparse
from pathlib import Path

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch import optim
from torch.nn.functional import l1_loss
from torch.utils.data import DataLoader
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.datasets.base import BaseDataset, BaseConcatDataset
from braindecode.models import EEGNeX
from eegdash import EEGChallengeDataset
import numpy as np

print("="*80)
print("🎯 CHALLENGE 2: SAM TRAINING WITH REAL EEG DATA")
print("="*80)
print("Task: contrastChangeDetection")
print("Target: p_factor (externalizing factor)")
print("Model: EEGNeX")
print("Optimizer: SAM (Sharpness-Aware Minimization)")
print("="*80)
print()


# ============================================================================
# SAM OPTIMIZER
# ============================================================================

class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer"""

    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def step(self, closure=None):
        raise NotImplementedError("Use first_step and second_step")

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("data/raw")
SFREQ = 100

# Subjects to remove (from starter kit)
SUB_RM = ["NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
          "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV",
          "NDARBA381JGH"]

# ============================================================================
# DATASET WRAPPER (from starter kit)
# ============================================================================

class DatasetWrapper(BaseDataset):
    """Wrapper to extract random 2-second crops from 4-second windows"""

    def __init__(self, dataset, crop_size_samples, target_name="p_factor", seed=42):
        self.dataset = dataset
        self.crop_size_samples = crop_size_samples
        self.target_name = target_name
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, _, crop_inds = self.dataset[index]

        target = self.dataset.description[self.target_name]
        target = float(target)

        # Additional information
        infos = {
            "subject": self.dataset.description["subject"],
            "sex": self.dataset.description["sex"],
            "age": float(self.dataset.description["age"]),
            "task": self.dataset.description["task"],
            "session": self.dataset.description.get("session", None) or "",
            "run": self.dataset.description.get("run", None) or "",
        }

        # Randomly crop to desired length
        i_window_in_trial, i_start, i_stop = crop_inds
        if i_stop - i_start >= self.crop_size_samples:
            start_offset = self.rng.randint(0, i_stop - i_start - self.crop_size_samples)
            X = X[:, start_offset : start_offset + self.crop_size_samples]

        return X, target, crop_inds, infos


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(model, train_loader, val_loader, device, epochs=20, lr=0.002, rho=0.05):
    """Train with SAM optimizer and L1 loss"""

    # Use SAM optimizer with Adamax base
    base_optimizer = torch.optim.Adamax
    optimizer = SAM(params=model.parameters(), base_optimizer=base_optimizer, lr=lr, rho=rho)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5

    print(f"\n🚀 Training for up to {epochs} epochs with SAM...")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Loss: L1 (robust to outliers)")
    print(f"   Optimizer: SAM (rho={rho})")
    print()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ===== TRAIN WITH SAM =====
        model.train()
        train_loss = 0
        train_count = 0

        for idx, batch in enumerate(train_loader):
            X, y, crop_inds, infos = batch
            X = X.to(dtype=torch.float32, device=device)
            y = y.to(dtype=torch.float32, device=device).unsqueeze(1)

            # First forward-backward pass (ascent step)
            y_pred = model(X)
            loss = l1_loss(y_pred, y)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # Second forward-backward pass (descent step)
            y_pred = model(X)
            loss = l1_loss(y_pred, y)
            loss.backward()
            optimizer.second_step(zero_grad=True)

            train_loss += loss.item()
            train_count += 1

            if idx % 20 == 0:
                print(f"  Epoch {epoch}/{epochs} - Batch {idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / train_count

        # ===== VALIDATE =====
        model.eval()
        val_loss = 0
        val_count = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                X, y, crop_inds, infos = batch
                X = X.to(dtype=torch.float32, device=device)
                y = y.to(dtype=torch.float32, device=device).unsqueeze(1)

                y_pred = model(X)
                loss = l1_loss(y_pred, y)

                val_loss += loss.item()
                val_count += 1
                val_preds.extend(y_pred.cpu().numpy().flatten())
                val_targets.extend(y.cpu().numpy().flatten())

        avg_val_loss = val_loss / val_count

        # Compute NRMSE
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        rmse = np.sqrt(np.mean((val_targets - val_preds) ** 2))
        nrmse = rmse / (val_targets.max() - val_targets.min() + 1e-8)

        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch:2d}/{epochs} ({epoch_time:.1f}s):")
        print(f"  Train L1 Loss: {avg_train_loss:.4f}")
        print(f"  Val L1 Loss:   {avg_val_loss:.4f}")
        print(f"  Val NRMSE:     {nrmse:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val_loss,
                'val_nrmse': nrmse,
                'epoch': epoch
            }, 'weights_challenge_2_correct.pt')
            print(f"  ✅ Best model saved (Val L1: {best_val_loss:.4f}, NRMSE: {nrmse:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping (no improvement for {patience} epochs)")
                break
        print()

    return best_val_loss


# ============================================================================
# MAIN
# ============================================================================

def main():
    start_time = time.time()

    # Use GPU with ROCm SDK (has gfx1010 support)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "="*80)
    print("PHASE 1: DATA LOADING")
    print("="*80)
    print(f"🖥️  Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("="*80)

    # Load R1-R4 for training, R5 for validation
    print("\nLoading training data (R1-R4)...")
    train_releases = ["R1", "R2", "R3", "R4"]

    train_datasets_list = []
    for release in train_releases:
        print(f"  Loading {release}...")
        try:
            ds = EEGChallengeDataset(
                release=release,
                task="contrastChangeDetection",  # CORRECT TASK
                mini=False,  # Use full data
                description_fields=["subject", "session", "run", "task", "age", "sex", "p_factor"],
                cache_dir=DATA_DIR,
            )
            train_datasets_list.append(ds)
            print(f"    ✅ {release} loaded")
        except Exception as e:
            print(f"    ⚠️  {release} failed: {e}")

    print("\nLoading validation data (R5)...")
    try:
        val_ds = EEGChallengeDataset(
            release="R5",
            task="contrastChangeDetection",
            mini=False,
            description_fields=["subject", "session", "run", "task", "age", "sex", "p_factor"],
            cache_dir=DATA_DIR,
        )
        print("  ✅ R5 loaded")
    except Exception as e:
        print(f"  ❌ R5 failed: {e}")
        return

    # Combine and filter
    print("\nCombining and filtering datasets...")
    train_datasets = BaseConcatDataset(train_datasets_list)

    # Filter out bad subjects and recordings
    train_datasets = BaseConcatDataset([
        ds for ds in train_datasets.datasets
        if (not ds.description.subject in SUB_RM and
            ds.raw.n_times >= 4 * SFREQ and
            len(ds.raw.ch_names) == 129 and
            not math.isnan(ds.description["p_factor"]))
    ])

    val_datasets = BaseConcatDataset([
        ds for ds in val_ds.datasets
        if (not ds.description.subject in SUB_RM and
            ds.raw.n_times >= 4 * SFREQ and
            len(ds.raw.ch_names) == 129 and
            not math.isnan(ds.description["p_factor"]))
    ])

    print(f"  Train: {len(train_datasets.datasets)} recordings")
    print(f"  Val:   {len(val_datasets.datasets)} recordings")

    # Create windows
    print("\nCreating windows (4s window, 2s stride)...")
    train_windows = create_fixed_length_windows(
        train_datasets,
        window_size_samples=4 * SFREQ,
        window_stride_samples=2 * SFREQ,
        drop_last_window=True,
    )

    val_windows = create_fixed_length_windows(
        val_datasets,
        window_size_samples=4 * SFREQ,
        window_stride_samples=2 * SFREQ,
        drop_last_window=True,
    )

    # Wrap with random cropping
    train_windows = BaseConcatDataset([
        DatasetWrapper(ds, crop_size_samples=2 * SFREQ)
        for ds in train_windows.datasets
    ])

    val_windows = BaseConcatDataset([
        DatasetWrapper(ds, crop_size_samples=2 * SFREQ)
        for ds in val_windows.datasets
    ])

    print(f"  Train windows: {len(train_windows)}")
    print(f"  Val windows:   {len(val_windows)}")

    # Create dataloaders
    train_loader = DataLoader(train_windows, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_windows, batch_size=64, shuffle=False, num_workers=0)

    # ===== CREATE MODEL =====
    print("\n" + "="*80)
    print("PHASE 2: MODEL CREATION")
    print("="*80)

    model = EEGNeX(n_chans=129, n_outputs=1, n_times=2 * SFREQ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: EEGNeX")
    print(f"Parameters: {n_params:,}")
    print(f"Designed for: Generalization and robustness")

    # ===== TRAIN =====
    print("\n" + "="*80)
    print("PHASE 3: TRAINING")
    print("="*80)

    best_loss = train_model(model, train_loader, val_loader, device, epochs=20, lr=0.002)

    # ===== SUMMARY =====
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Best validation L1 loss: {best_loss:.4f}")
    print(f"Model saved to: weights_challenge_2_correct.pt")
    print()
    print("Next steps:")
    print("  1. Copy to submission: cp weights_challenge_2_correct.pt weights_challenge_2.pt")
    print("  2. Update submission.py to use EEGNeX")
    print("  3. Recreate zip: zip -j submission.zip submission.py weights_*.pt")
    print("  4. Submit!")
    print("="*80)


if __name__ == "__main__":
    main()
