#!/usr/bin/env python3
"""
Challenge 1 LIGHT Enhanced Training
Uses EEGNeX baseline + SAM optimizer only (no attention, no multi-scale)
This is proven to work - we have weights_challenge_1_sam.pt from this approach
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Hide GPU during imports
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HIP_VISIBLE_DEVICES'] = ''

import mne
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from braindecode.models import EEGNeX
from tqdm import tqdm

mne.set_config('MNE_USE_CUDA', 'false', set_env=True)
mne.set_log_level('WARNING')

print("=" * 80)
print("🚀 CHALLENGE 1: LIGHT ENHANCED (EEGNeX + SAM)")
print("=" * 80)
print("Proven approach: Baseline model + SAM optimizer")
print("Model: 62K params (TESTED & WORKING)")
print("=" * 80)

# SAM Optimizer
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
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
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

# Dataset
class ResponseTimeDataset(Dataset):
    def __init__(self, data_dirs, max_subjects=None):
        self.segments = []
        self.response_times = []
        self.subject_ids = []

        print(f"\n📁 Loading data (GPU HIDDEN)...")

        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            participants_file = data_dir / "participants.tsv"
            if not participants_file.exists():
                continue

            df = pd.read_csv(participants_file, sep='\t')
            if max_subjects:
                df = df.head(max_subjects)

            print(f"   📊 {data_dir.name}: {len(df)} subjects")

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"   {data_dir.name}", leave=False):
                subject_id = row['participant_id']
                subject_dir = data_dir / subject_id / "eeg"
                if not subject_dir.exists():
                    continue

                eeg_files = list(subject_dir.glob("*contrastChangeDetection*.bdf"))
                for eeg_file in eeg_files:
                    try:
                        raw = mne.io.read_raw_bdf(eeg_file, preload=True, verbose=False)
                        if raw.info['sfreq'] != 100:
                            raw.resample(100, verbose=False)

                        data = raw.get_data()
                        if data.shape[0] != 129:
                            continue

                        data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)

                        events_file = eeg_file.with_name(eeg_file.name.replace('_eeg.bdf', '_events.tsv'))
                        if not events_file.exists():
                            continue

                        events_df = pd.read_csv(events_file, sep='\t')
                        trial_starts = events_df[events_df['value'].str.contains('contrastTrial_start', case=False, na=False)]
                        button_presses = events_df[events_df['value'].str.contains('buttonPress', case=False, na=False)]

                        for _, trial in trial_starts.iterrows():
                            trial_time = trial['onset']
                            later_presses = button_presses[button_presses['onset'] > trial_time]
                            if len(later_presses) == 0:
                                continue

                            rt = later_presses.iloc[0]['onset'] - trial_time
                            if rt < 0.1 or rt > 5.0:
                                continue

                            start_sample = int(trial_time * 100)
                            end_sample = start_sample + 200
                            if end_sample > data.shape[1]:
                                continue

                            self.segments.append(data[:, start_sample:end_sample])
                            self.response_times.append(rt)
                            self.subject_ids.append(subject_id)
                    except:
                        continue

        self.segments = np.array(self.segments, dtype=np.float32)
        self.response_times = np.array(self.response_times, dtype=np.float32)
        self.subject_ids = np.array(self.subject_ids)

        print(f"\n   ✅ Loaded {len(self)} windows")
        print(f"   RT range: {self.response_times.min():.3f} - {self.response_times.max():.3f} sec")
        print(f"   Subjects: {len(np.unique(self.subject_ids))}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.segments[idx]), torch.FloatTensor([self.response_times[idx]])

def compute_nrmse(preds, targets):
    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    return rmse / (targets.max() - targets.min() + 1e-8)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        # SAM first step
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # SAM second step  
        criterion(model(X), y).backward()
        optimizer.second_step(zero_grad=True)

        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            pred = model(X.to(device))
            preds.extend(pred.cpu().numpy().squeeze())
            targets.extend(y.numpy().squeeze())
    return compute_nrmse(np.array(preds), np.array(targets))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dirs', nargs='+', default=['data/ds005506-bdf', 'data/ds005507-bdf'])
    parser.add_argument('--max_subjects', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--rho', type=float, default=0.05)
    args = parser.parse_args()

    print(f"\n⚙️  Config: max_subjects={args.max_subjects}, batch_size={args.batch_size}, epochs={args.epochs}")

    # Load data with GPU hidden
    print("\n" + "="*80)
    dataset = ResponseTimeDataset(args.data_dirs, args.max_subjects)
    
    subjects = dataset.subject_ids
    unique_subjects = np.unique(subjects)
    n_train = int(0.8 * len(unique_subjects))
    train_subjects = unique_subjects[:n_train]
    val_subjects = unique_subjects[n_train:]

    train_idx = [i for i, s in enumerate(subjects) if s in train_subjects]
    val_idx = [i for i, s in enumerate(subjects) if s in val_subjects]

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    print(f"   Train: {len(train_subjects)} subjects, {len(train_set)} samples")
    print(f"   Val: {len(val_subjects)} subjects, {len(val_set)} samples")

    # Enable GPU
    print("\n" + "="*80)
    print("🖥️  EXPOSING GPU")
    print("="*80)
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['HIP_VISIBLE_DEVICES'] = '0'
    torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Create model (EEGNeX baseline - 62K params, PROVEN WORKING)
    print("\n" + "="*80)
    print("🧠 MODEL: EEGNeX (Baseline)")
    print("="*80)
    model = EEGNeX(
        n_outputs=1,
        n_chans=129,
        n_times=200,
        final_fc_length='auto'
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")

    # Setup
    criterion = nn.MSELoss()
    optimizer = SAM(model.parameters(), AdamW, lr=args.lr, rho=args.rho, weight_decay=1e-4)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    exp_dir = Path('experiments') / 'light_enhanced' / datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Experiment: {exp_dir}")
    print("\n" + "="*80)
    print("🚀 TRAINING")
    print("="*80)

    best_val = float('inf')
    patience, patience_counter = 15, 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_nrmse = validate(model, val_loader, device)

        is_best = val_nrmse < best_val
        marker = "✨ BEST" if is_best else ""
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {train_loss:.4f} | Val NRMSE: {val_nrmse:.4f} {marker}")

        if is_best:
            best_val = val_nrmse
            patience_counter = 0
            torch.save(model.state_dict(), exp_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  Early stop at epoch {epoch+1}")
                break

    print(f"\n✅ Done! Best Val NRMSE: {best_val:.4f}")
    print(f"   Weights: {exp_dir / 'best_model.pt'}")

if __name__ == '__main__':
    main()
