#!/usr/bin/env python3
"""
Challenge 1: Training with Pre-cached H5 Data (FAST!)
Uses cached windows from data/cached/challenge1_*.h5
"""

import sys
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from scipy.stats import pearsonr
from scipy.signal import welch
from braindecode.models import EEGNeX
from tqdm import tqdm
from pathlib import Path

print("\n" + "=" * 80)
print("🚀 CHALLENGE 1: CACHED DATA TRAINING → Target Pearson r ≥ 0.91")
print("=" * 80)


# ============================================================================
# MODEL
# ============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, n_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(n_channels, max(1, n_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, n_channels // reduction), n_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        attention = self.sigmoid(avg_out + max_out).unsqueeze(-1)
        return x * attention


class ImprovedEEGModel(nn.Module):
    def __init__(self, n_channels=129, n_times=200, n_outputs=1):
        super().__init__()
        
        self.backbone = EEGNeX(
            n_outputs=64,
            n_chans=n_channels,
            n_times=n_times,
            drop_prob=0.3
        )
        
        self.channel_attention = ChannelAttention(n_channels, reduction=8)
        
        self.freq_encoder = nn.Sequential(
            nn.Linear(n_channels * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.head = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_outputs)
        )
    
    def extract_freq_features(self, x):
        batch_size, n_channels, n_times = x.shape
        x_np = x.detach().cpu().numpy()
        
        features = []
        for i in range(batch_size):
            band_powers = []
            for ch in range(n_channels):
                freqs, psd = welch(x_np[i, ch], fs=100, nperseg=min(64, n_times))
                
                delta = np.mean(psd[(freqs >= 0.5) & (freqs < 4)])
                theta = np.mean(psd[(freqs >= 4) & (freqs < 8)])
                alpha = np.mean(psd[(freqs >= 8) & (freqs < 13)])
                beta = np.mean(psd[(freqs >= 13) & (freqs < 30)])
                
                band_powers.extend([delta, theta, alpha, beta])
            
            features.append(band_powers)
        
        return torch.tensor(features, device=x.device, dtype=x.dtype)
    
    def forward(self, x):
        x_attended = self.channel_attention(x)
        
        time_features = self.backbone(x_attended)
        freq_features = self.freq_encoder(self.extract_freq_features(x))
        
        combined = torch.cat([time_features, freq_features], dim=1)
        return self.head(combined)


# ============================================================================
# DATASET (using cached H5 files)
# ============================================================================

class CachedResponseTimeDataset(Dataset):
    """Load from pre-cached H5 files"""
    
    def __init__(self, h5_files, max_samples=None):
        print("\n📁 Loading cached data...")
        
        self.segments = []
        self.response_times = []
        
        for h5_file in h5_files:
            h5_file = Path(h5_file)
            if not h5_file.exists():
                print(f"   ⚠️  Skipping {h5_file.name} (not found)")
                continue
            
            print(f"   📦 Loading {h5_file.name}...")
            with h5py.File(h5_file, 'r') as f:
                segments = f['eeg'][:]
                rts = f['labels'][:]
                
                # Filter valid response times (0.1 to 5.0 seconds)
                valid_mask = (rts >= 0.1) & (rts <= 5.0)
                segments = segments[valid_mask]
                rts = rts[valid_mask]
                
                self.segments.append(segments)
                self.response_times.append(rts)
                
                print(f"      ✅ Loaded {len(rts):,} windows")
        
        if self.segments:
            self.segments = np.concatenate(self.segments, axis=0)
            self.response_times = np.concatenate(self.response_times, axis=0)
        
        if max_samples and len(self.segments) > max_samples:
            indices = np.random.choice(len(self.segments), max_samples, replace=False)
            self.segments = self.segments[indices]
            self.response_times = self.response_times[indices]
        
        print(f"\n   ✅ Total: {len(self.segments):,} windows")
        print(f"   RT range: {self.response_times.min():.3f} - {self.response_times.max():.3f}s")
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = torch.from_numpy(self.segments[idx]).float()
        rt = torch.tensor(self.response_times[idx], dtype=torch.float32)
        return segment, rt


# ============================================================================
# TRAINING
# ============================================================================

def compute_nrmse(y_true, y_pred):
    y_true_np = y_true.detach().cpu().numpy().flatten()
    y_pred_np = y_pred.detach().cpu().numpy().flatten()
    rmse = np.sqrt(np.mean((y_true_np - y_pred_np) ** 2))
    std = np.std(y_true_np)
    return rmse / (std + 1e-8)


def compute_pearson(y_true, y_pred):
    y_true_np = y_true.detach().cpu().numpy().flatten()
    y_pred_np = y_pred.detach().cpu().numpy().flatten()
    if len(y_true_np) < 2:
        return 0.0
    corr, _ = pearsonr(y_true_np, y_pred_np)
    return corr if not np.isnan(corr) else 0.0


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for X, y in tqdm(loader, desc="   Training", leave=False):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X).squeeze()
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for X, y in tqdm(loader, desc="   Validating", leave=False):
            X, y = X.to(device), y.to(device)
            y_pred = model(X).squeeze()
            
            all_preds.append(y_pred)
            all_targets.append(y)
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    nrmse = compute_nrmse(all_targets, all_preds)
    pearson = compute_pearson(all_targets, all_preds)
    
    return nrmse, pearson


def main():
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_epochs = 50
    batch_size = 32
    learning_rate = 0.001
    
    print(f"\n⚙️  Configuration:")
    print(f"   Device: {device}")
    print(f"   Epochs: {n_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    
    # Load cached data
    h5_files = [
        'data/cached/challenge1_R1_windows.h5',
        'data/cached/challenge1_R2_windows.h5',
        'data/cached/challenge1_R3_windows.h5',
        'data/cached/challenge1_R4_windows.h5'
    ]
    
    dataset = CachedResponseTimeDataset(h5_files)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"\n📊 Dataset split:")
    print(f"   Train: {len(train_dataset):,} samples")
    print(f"   Val: {len(val_dataset):,} samples")
    
    # Model
    model = ImprovedEEGModel(n_channels=129, n_times=200, n_outputs=1).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model: {total_params:,} parameters")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training loop
    print("\n" + "=" * 80)
    print("🎯 TRAINING")
    print("=" * 80)
    
    best_pearson = -1.0
    best_nrmse = float('inf')
    
    for epoch in range(n_epochs):
        print(f"\n📍 Epoch {epoch+1}/{n_epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_nrmse, val_pearson = validate(model, val_loader, device)
        
        scheduler.step()
        
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val NRMSE: {val_nrmse:.4f}")
        print(f"   Val Pearson: {val_pearson:.4f}")
        
        # Save best model
        if val_pearson > best_pearson:
            best_pearson = val_pearson
            best_nrmse = val_nrmse
            torch.save(model.state_dict(), 'checkpoints/c1_improved_best.pt')
            print(f"   💾 Saved best model (r={val_pearson:.4f})")
        
        # Early stopping if target reached
        if val_pearson >= 0.91:
            print(f"\n🎉 TARGET REACHED! Pearson r = {val_pearson:.4f} ≥ 0.91")
            break
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Val NRMSE: {best_nrmse:.4f}")
    print(f"Best Val Pearson: {best_pearson:.4f}")
    print(f"Target (r ≥ 0.91): {'✅ ACHIEVED' if best_pearson >= 0.91 else '❌ NOT YET'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

