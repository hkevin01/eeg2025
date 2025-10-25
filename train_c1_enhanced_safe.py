#!/usr/bin/env python3
"""
Challenge 1 Enhanced Training - GPU Safe Version
Loads data with GPU hidden, then exposes GPU for training
"""

import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from datetime import datetime

# CRITICAL: Hide GPU during imports and data loading
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HIP_VISIBLE_DEVICES'] = ''
os.environ['ROCR_VISIBLE_DEVICES'] = ''

import mne
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import GroupKFold
from braindecode.models import EEGNeX
from tqdm import tqdm

# Configure MNE
mne.set_config('MNE_USE_CUDA', 'false', set_env=True)
mne.set_log_level('WARNING')

print("=" * 80)
print("🚀 CHALLENGE 1: ENHANCED TRAINING - GPU SAFE")
print("=" * 80)
print("Strategy: Load data on CPU, then move to GPU for training")
print("=" * 80)

# SAM Optimizer
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
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

# Temporal Attention
class TemporalAttention(nn.Module):
    def __init__(self, embed_dim=96, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x: (batch, channels, time) -> (batch, time, channels)
        x = x.transpose(1, 2)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        return x.transpose(1, 2)  # Back to (batch, channels, time)

# Multi-Scale Features
class MultiScaleFeaturesExtractor(nn.Module):
    def __init__(self, in_channels=129, out_channels=32):
        super().__init__()
        self.conv_small = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv_medium = nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=7)
        self.conv_large = nn.Conv1d(in_channels, out_channels, kernel_size=31, padding=15)
        self.bn = nn.BatchNorm1d(out_channels * 3)
        
    def forward(self, x):
        small = self.conv_small(x)
        medium = self.conv_medium(x)
        large = self.conv_large(x)
        combined = torch.cat([small, medium, large], dim=1)
        return self.bn(combined)

# Enhanced Model
class EnhancedEEGNeX(nn.Module):
    def __init__(self, n_channels=129, n_times=200, n_outputs=1):
        super().__init__()
        self.multiscale = MultiScaleFeaturesExtractor(n_channels, out_channels=32)
        self.temporal_attention = TemporalAttention(embed_dim=96, num_heads=4)
        self.fusion = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_outputs)
        )

    def forward(self, x):
        multiscale_features = self.multiscale(x)
        attended_features = self.temporal_attention(multiscale_features)
        output = self.fusion(attended_features)
        return output

# Dataset
class ResponseTimeDataset(Dataset):
    def __init__(self, data_dirs, max_subjects=None, augment=False):
        self.segments = []
        self.response_times = []
        self.subject_ids = []
        self.augment = augment

        print(f"\n📁 Loading data (augment={augment}, GPU HIDDEN)...")

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
                if not eeg_files:
                    continue

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
                        trial_start_events = events_df[events_df['value'].str.contains('contrastTrial_start', case=False, na=False)]
                        button_press_events = events_df[events_df['value'].str.contains('buttonPress', case=False, na=False)]

                        if len(trial_start_events) == 0 or len(button_press_events) == 0:
                            continue

                        for _, trial_event in trial_start_events.iterrows():
                            trial_time = trial_event['onset']
                            later_presses = button_press_events[button_press_events['onset'] > trial_time]
                            if len(later_presses) == 0:
                                continue

                            press_event = later_presses.iloc[0]
                            response_time = press_event['onset'] - trial_time

                            if response_time < 0.1 or response_time > 5.0:
                                continue

                            start_sample = int(trial_time * 100)
                            end_sample = start_sample + 200

                            if end_sample > data.shape[1]:
                                continue

                            segment = data[:, start_sample:end_sample]
                            self.segments.append(segment)
                            self.response_times.append(response_time)
                            self.subject_ids.append(subject_id)

                    except Exception as e:
                        continue

        self.segments = np.array(self.segments, dtype=np.float32)
        self.response_times = np.array(self.response_times, dtype=np.float32)
        self.subject_ids = np.array(self.subject_ids)

        print(f"\n   ✅ Loaded {len(self)} windows with response times")
        if len(self) > 0:
            print(f"   RT range: {self.response_times.min():.3f} - {self.response_times.max():.3f} seconds")
            print(f"   Unique subjects: {len(np.unique(self.subject_ids))}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        X = torch.FloatTensor(self.segments[idx])
        y = torch.FloatTensor([self.response_times[idx]])
        return X, y

def compute_nrmse(predictions, targets):
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    nrmse = rmse / (targets.max() - targets.min() + 1e-8)
    return nrmse

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # SAM first step
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # SAM second step
        predictions = model(batch_X)
        criterion(predictions, batch_y).backward()
        optimizer.second_step(zero_grad=True)

        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            predictions = model(batch_X)
            all_preds.extend(predictions.cpu().numpy().squeeze())
            all_targets.extend(batch_y.numpy().squeeze())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    nrmse = compute_nrmse(all_preds, all_targets)
    return nrmse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dirs', nargs='+', default=['data/ds005506-bdf', 'data/ds005507-bdf'])
    parser.add_argument('--max_subjects', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--rho', type=float, default=0.05)
    parser.add_argument('--exp_name', type=str, default='enhanced_safe')
    parser.add_argument('--early_stopping', type=int, default=15)
    args = parser.parse_args()

    print(f"\n⚙️  Configuration:")
    for arg, value in vars(args).items():
        print(f"   {arg}: {value}")

    # Load data with GPU HIDDEN
    print("\n" + "="*80)
    print("📁 LOADING DATA (GPU HIDDEN)")
    print("="*80)
    
    full_dataset = ResponseTimeDataset(
        data_dirs=args.data_dirs,
        max_subjects=args.max_subjects,
        augment=False
    )

    # Split data
    subjects = full_dataset.subject_ids
    unique_subjects = np.unique(subjects)
    n_train = int(0.8 * len(unique_subjects))
    train_subjects = unique_subjects[:n_train]
    val_subjects = unique_subjects[n_train:]

    train_indices = [i for i, s in enumerate(subjects) if s in train_subjects]
    val_indices = [i for i, s in enumerate(subjects) if s in val_subjects]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    print(f"\n   Total subjects: {len(unique_subjects)}")
    print(f"   Train subjects: {len(train_subjects)}")
    print(f"   Val subjects: {len(val_subjects)}")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")

    # NOW EXPOSE GPU FOR TRAINING
    print("\n" + "="*80)
    print("🖥️  EXPOSING GPU FOR TRAINING")
    print("="*80)
    
    # Re-enable GPU
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['HIP_VISIBLE_DEVICES'] = '0'
    os.environ['ROCR_VISIBLE_DEVICES'] = '0'
    
    # Force torch to reinitialize CUDA
    torch.cuda.empty_cache()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # Create model
    print("\n" + "="*80)
    print("🧠 CREATING MODEL")
    print("="*80)
    
    model = EnhancedEEGNeX(n_channels=129, n_times=200, n_outputs=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")

    # Setup training
    criterion = nn.MSELoss()
    optimizer = SAM(model.parameters(), AdamW, lr=args.lr, rho=args.rho, weight_decay=1e-4)

    # Experiment directory
    exp_dir = Path('experiments') / args.exp_name / datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = exp_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)

    print(f"\n📁 Experiment: {exp_dir}")

    # Training loop
    print("\n" + "="*80)
    print("🚀 TRAINING STARTED")
    print("="*80)

    best_val_nrmse = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_nrmse': []}

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_nrmse = validate(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['val_nrmse'].append(val_nrmse)

        is_best = val_nrmse < best_val_nrmse
        best_marker = "✨ BEST!" if is_best else ""

        print(f"Epoch {epoch+1:3d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val NRMSE: {val_nrmse:.4f} | {best_marker}")

        if is_best:
            best_val_nrmse = val_nrmse
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.base_optimizer.state_dict(),
                'val_nrmse': val_nrmse,
                'train_loss': train_loss,
            }
            torch.save(checkpoint, checkpoints_dir / 'best_model.pt')
            torch.save(model.state_dict(), checkpoints_dir / 'best_model_weights.pt')
        else:
            patience_counter += 1

        if patience_counter >= args.early_stopping:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            break

    print(f"\n{'='*80}")
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"   Best Val NRMSE: {best_val_nrmse:.4f}")
    print(f"   Experiment: {exp_dir}")
    print(f"   Best model: {checkpoints_dir / 'best_model.pt'}")

    with open(exp_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == '__main__':
    main()
