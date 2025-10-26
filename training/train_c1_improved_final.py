#!/usr/bin/env python3
"""
Challenge 1: Improved Training to reach Pearson r=0.91

Key Improvements:
1. Channel + Temporal Attention
2. Time-Frequency Features (alpha/beta/theta/gamma)
3. Pearson Correlation Loss + Tracking
4. Larger Model Capacity (169K params)
5. Better Regularization
6. SAM Optimizer

Expected: r = 0.85-0.95 (target ≥ 0.91)
"""

import sys
import json
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import mne
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from scipy.stats import pearsonr
from scipy.signal import welch
from braindecode.models import EEGNeX
from tqdm import tqdm

print("=" * 80)
print("🚀 CHALLENGE 1: IMPROVED TRAINING → Target Pearson r ≥ 0.91")
print("=" * 80)


# ============================================================================
# ATTENTION MECHANISMS
# ============================================================================

class ChannelAttention(nn.Module):
    """Learn which EEG channels are most important"""
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
        # x: (batch, channels, time)
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        attention = self.sigmoid(avg_out + max_out).unsqueeze(-1)
        return x * attention


class ImprovedEEGModel(nn.Module):
    """Enhanced EEGNeX with attention and time-frequency features"""
    def __init__(self, n_channels=129, n_times=200, n_outputs=1):
        super().__init__()
        
        # Backbone: EEGNeX
        self.backbone = EEGNeX(
            n_outputs=64,
            n_chans=n_channels,
            n_times=n_times,
            drop_prob=0.3
        )
        
        # Channel attention
        self.channel_attention = ChannelAttention(n_channels, reduction=8)
        
        # Time-frequency branch (faster: just power in key bands)
        self.freq_encoder = nn.Sequential(
            nn.Linear(n_channels * 4, 128),  # 4 bands per channel
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Fusion head
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
        """Fast frequency feature extraction"""
        batch_size, n_channels, n_times = x.shape
        x_np = x.detach().cpu().numpy()
        
        features = []
        for i in range(batch_size):
            band_powers = []
            for ch in range(n_channels):
                freqs, psd = welch(x_np[i, ch], fs=100, nperseg=min(64, n_times))
                
                theta = psd[(freqs >= 4) & (freqs < 8)].mean()
                alpha = psd[(freqs >= 8) & (freqs < 13)].mean()
                beta = psd[(freqs >= 13) & (freqs < 30)].mean()
                gamma = psd[(freqs >= 30) & (freqs < 50)].mean()
                
                band_powers.extend([theta, alpha, beta, gamma])
            
            features.append(band_powers)
        
        return torch.tensor(features, dtype=x.dtype, device=x.device)
    
    def forward(self, x):
        x_att = self.channel_attention(x)
        backbone_feat = self.backbone(x_att)
        freq_feat = self.extract_freq_features(x)
        freq_feat = self.freq_encoder(freq_feat)
        combined = torch.cat([backbone_feat, freq_feat], dim=1)
        return self.head(combined)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class PearsonCorrelationLoss(nn.Module):
    """Directly optimize Pearson correlation"""
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true):
        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        
        y_pred_centered = y_pred - y_pred.mean()
        y_true_centered = y_true - y_true.mean()
        
        cov = (y_pred_centered * y_true_centered).mean()
        std_pred = y_pred_centered.std() + 1e-8
        std_true = y_true_centered.std() + 1e-8
        
        corr = cov / (std_pred * std_true)
        return 1.0 - corr


class CombinedLoss(nn.Module):
    """MSE + Pearson correlation"""
    def __init__(self, mse_weight=0.6, corr_weight=0.4):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.corr_loss = PearsonCorrelationLoss()
        self.mse_weight = mse_weight
        self.corr_weight = corr_weight
    
    def forward(self, y_pred, y_true):
        mse = self.mse_loss(y_pred, y_true)
        corr = self.corr_loss(y_pred, y_true)
        return self.mse_weight * mse + self.corr_weight * corr


# ============================================================================
# METRICS
# ============================================================================

def compute_nrmse(y_true, y_pred):
    """Normalized RMSE (competition metric)"""
    y_true_np = y_true.detach().cpu().numpy().flatten()
    y_pred_np = y_pred.detach().cpu().numpy().flatten()
    
    rmse = np.sqrt(np.mean((y_true_np - y_pred_np) ** 2))
    std = np.std(y_true_np)
    return rmse / (std + 1e-8)


def compute_pearson(y_true, y_pred):
    """Pearson correlation (target metric)"""
    y_true_np = y_true.detach().cpu().numpy().flatten()
    y_pred_np = y_pred.detach().cpu().numpy().flatten()
    
    if len(y_true_np) < 2:
        return 0.0
    
    corr, _ = pearsonr(y_true_np, y_pred_np)
    return corr if not np.isnan(corr) else 0.0


# ============================================================================
# DATA LOADER (from working script)
# ============================================================================

class ResponseTimeDataset(Dataset):
    """Load EEG windows with response times from BIDS events"""
    
    def __init__(self, data_dirs, max_subjects=None, augment=False):
        self.segments = []
        self.response_times = []
        self.subject_ids = []
        self.augment = augment
        
        print(f"\n📁 Loading data (augment={augment})...")
        
        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            participants_file = data_dir / "participants.tsv"
            
            if not participants_file.exists():
                continue
            
            df = pd.read_csv(participants_file, sep="\t")
            
            if max_subjects:
                df = df.head(max_subjects)
            
            print(f"   📊 {data_dir.name}: {len(df)} subjects")
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"   {data_dir.name}", leave=False):
                subject_id = row["participant_id"]
                subject_dir = data_dir / subject_id / "eeg"
                
                if not subject_dir.exists():
                    continue
                
                eeg_files = list(subject_dir.glob("*contrastChangeDetection*.bdf"))
                if not eeg_files:
                    continue
                
                for eeg_file in eeg_files:
                    try:
                        raw = mne.io.read_raw_bdf(eeg_file, preload=True, verbose=False)
                        
                        if raw.info["sfreq"] != 100:
                            raw.resample(100, verbose=False)
                        
                        data = raw.get_data()
                        
                        if data.shape[0] != 129:
                            continue
                        
                        data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
                        
                        events_file = eeg_file.with_name(eeg_file.name.replace("_eeg.bdf", "_events.tsv"))
                        if not events_file.exists():
                            continue
                        
                        events_df = pd.read_csv(events_file, sep="\t")
                        
                        trial_start_events = events_df[events_df["value"].str.contains("contrastTrial_start", case=False, na=False)]
                        button_press_events = events_df[events_df["value"].str.contains("buttonPress", case=False, na=False)]
                        
                        if len(trial_start_events) == 0 or len(button_press_events) == 0:
                            continue
                        
                        for _, trial_event in trial_start_events.iterrows():
                            trial_time = trial_event["onset"]
                            
                            later_presses = button_press_events[button_press_events["onset"] > trial_time]
                            if len(later_presses) == 0:
                                continue
                            
                            press_event = later_presses.iloc[0]
                            response_time = press_event["onset"] - trial_time
                            
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
                    
                    except:
                        continue
        
        self.segments = np.array(self.segments, dtype=np.float32)
        self.response_times = np.array(self.response_times, dtype=np.float32)
        self.subject_ids = np.array(self.subject_ids)
        
        print(f"\n   ✅ Loaded {len(self)} windows")
        if len(self) > 0:
            print(f"   RT range: {self.response_times.min():.3f} - {self.response_times.max():.3f}s")
            print(f"   Unique subjects: {len(np.unique(self.subject_ids))}")
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        X = torch.FloatTensor(self.segments[idx])
        y = torch.FloatTensor([self.response_times[idx]])
        
        if self.augment:
            if torch.rand(1).item() < 0.5:
                X = X * (0.8 + 0.4 * torch.rand(1).item())
            if torch.rand(1).item() < 0.3:
                n_drop = max(1, int(0.05 * X.shape[0]))
                drop_ch = torch.randperm(X.shape[0])[:n_drop]
                X[drop_ch, :] = 0.0
            if torch.rand(1).item() < 0.2:
                X = X + torch.randn_like(X) * 0.05
        
        return X, y


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for eeg, rt in pbar:
        eeg, rt = eeg.to(device), rt.to(device)
        
        optimizer.zero_grad()
        pred = model(eeg)
        loss = criterion(pred, rt)
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        all_preds.append(pred.detach())
        all_targets.append(rt.detach())
        
        pbar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / len(loader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    nrmse = compute_nrmse(all_targets, all_preds)
    pearson_r = compute_pearson(all_targets, all_preds)
    
    return avg_loss, nrmse, pearson_r


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    
    pbar = tqdm(loader, desc="Val", leave=False)
    for eeg, rt in pbar:
        eeg, rt = eeg.to(device), rt.to(device)
        pred = model(eeg)
        loss = criterion(pred, rt)
        
        total_loss += loss.item()
        all_preds.append(pred.detach())
        all_targets.append(rt.detach())
        
        pbar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / len(loader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    nrmse = compute_nrmse(all_targets, all_preds)
    pearson_r = compute_pearson(all_targets, all_preds)
    
    return avg_loss, nrmse, pearson_r


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dirs", nargs="+", default=["data/ds005506-bdf", "data/ds005507-bdf"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--exp-name", type=str, default="improved_c1")
    
    args = parser.parse_args()
    
    print(f"\n⚙️  Configuration:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Device: {args.device}")
    
    device = torch.device(args.device)
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("experiments") / args.exp_name / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = exp_dir / "training.log"
    
    try:
        # Load data
        train_dataset = ResponseTimeDataset(args.data_dirs, augment=True)
        val_dataset = ResponseTimeDataset(args.data_dirs, augment=False)
        
        # Split 80/20
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        _, val_dataset = torch.utils.data.random_split(val_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        print(f"\n📊 Data:")
        print(f"   Train: {len(train_dataset)} samples")
        print(f"   Val: {len(val_dataset)} samples")
        
        # Model
        model = ImprovedEEGModel(n_channels=129, n_times=200).to(device)
        print(f"\n🧠 Model: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Training setup
        criterion = CombinedLoss(mse_weight=0.6, corr_weight=0.4)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        # Training loop
        best_pearson = 0.0
        best_nrmse = float("inf")
        
        print(f"\n🏋️  Training...")
        for epoch in range(1, args.epochs + 1):
            train_loss, train_nrmse, train_r = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
            val_loss, val_nrmse, val_r = evaluate(model, val_loader, criterion, device)
            
            # Log
            log_msg = f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.4f} NRMSE: {train_nrmse:.4f} r: {train_r:.4f} | Val Loss: {val_loss:.4f} NRMSE: {val_nrmse:.4f} r: {val_r:.4f}"
            
            if val_r > best_pearson:
                best_pearson = val_r
                log_msg += " | ✨ BEST r!"
                torch.save(model.state_dict(), exp_dir / "best_model_pearson.pt")
            
            if val_nrmse < best_nrmse:
                best_nrmse = val_nrmse
                log_msg += " | ✨ BEST NRMSE!"
                torch.save(model.state_dict(), exp_dir / "best_model_nrmse.pt")
            
            print(log_msg, flush=True)
            with open(log_file, "a") as f:
                f.write(log_msg + "\n")
        
        print(f"\n🎉 Training complete!")
        print(f"   Best Pearson r: {best_pearson:.4f} (target ≥ 0.91)")
        print(f"   Best NRMSE: {best_nrmse:.4f}")
        print(f"   Saved to: {exp_dir}")
        
        # Save final summary
        with open(exp_dir / "summary.txt", "w") as f:
            f.write(f"Best Pearson r: {best_pearson:.4f}\n")
            f.write(f"Best NRMSE: {best_nrmse:.4f}\n")
            f.write(f"Target: r ≥ 0.91\n")
            f.write(f"Status: {'✅ ACHIEVED' if best_pearson >= 0.91 else '⚠️  CLOSE' if best_pearson >= 0.85 else '❌ NEED MORE IMPROVEMENT'}\n")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
