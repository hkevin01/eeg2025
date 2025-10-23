#!/usr/bin/env python3
"""
Challenge 2: Externalizing Factor Prediction (GPU-Compatible)
==============================================================
Universal training script for NVIDIA CUDA and AMD ROCm

Requirements:
- Input: (batch, 129, 200) - 129 channels, 200 samples @ 100Hz
- Output: (batch, 1) - externalizing score  
- Metric: NRMSE (target < 0.5)
- GPU: Automatically detects CUDA or ROCm
"""

import os
import sys
from pathlib import Path
import time
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import GPU utilities
from utils.gpu_utils import setup_device

# Import model
try:
    from models.baseline.tcn import TemporalConvNet
except ImportError:
    print("⚠️  Could not import TCN, using simplified model")
    TemporalConvNet = None


class SimplifiedTCN(nn.Module):
    """Simplified TCN for Challenge 2 if full model unavailable"""
    
    def __init__(self, n_channels=129, n_outputs=1):
        super().__init__()
        
        # Temporal feature extraction
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Regression head
        self.fc = nn.Linear(32, n_outputs)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: (batch, channels, time)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x


class ExternalizingDataset(Dataset):
    """Simple EEG dataset for externalizing prediction"""
    
    def __init__(self, data_dir, max_samples=None):
        """
        Args:
            data_dir: Path to HBN data
            max_samples: Maximum number of samples (for quick testing)
        """
        self.data_dir = Path(data_dir)
        
        print("\n📋 Loading externalizing scores...")
        participants_file = self.data_dir / "participants.tsv"
        
        if not participants_file.exists():
            print(f"❌ participants.tsv not found at {participants_file}")
            print("   Using dummy data for testing...")
            self.use_dummy = True
            self.n_samples = max_samples if max_samples else 100
            return
        
        self.use_dummy = False
        self.participants_df = pd.read_csv(participants_file, sep='\t')
        
        # Filter for subjects with externalizing scores
        self.participants_df = self.participants_df.dropna(subset=['externalizing'])
        
        if max_samples:
            self.participants_df = self.participants_df.head(max_samples)
        
        print(f"   Found {len(self.participants_df)} participants with externalizing scores")
        
        # TODO: Load actual EEG data
        # For now, generate dummy data to test GPU pipeline
        self.n_samples = len(self.participants_df)
    
    def __len__(self):
        if self.use_dummy:
            return self.n_samples
        return len(self.participants_df)
    
    def __getitem__(self, idx):
        if self.use_dummy:
            # Generate dummy data for testing
            eeg = torch.randn(129, 200)
            score = torch.randn(1) * 10 + 50  # Random score ~50
        else:
            # TODO: Load real EEG data
            # For now, use dummy data
            eeg = torch.randn(129, 200)
            score = torch.tensor([self.participants_df.iloc[idx]['externalizing']], dtype=torch.float32)
        
        return eeg, score


def calculate_nrmse(y_true, y_pred):
    """Calculate Normalized RMSE"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Normalize by range
    y_range = y_true.max() - y_true.min()
    if y_range == 0:
        return 0.0
    
    nrmse = rmse / y_range
    return nrmse


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for eeg, scores in dataloader:
        eeg = eeg.to(device)
        scores = scores.to(device)
        
        optimizer.zero_grad()
        outputs = model(eeg)
        loss = criterion(outputs, scores)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for eeg, scores in dataloader:
            eeg = eeg.to(device)
            outputs = model(eeg)
            
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_true.extend(scores.numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    # Calculate metrics
    nrmse = calculate_nrmse(all_true, all_preds)
    mae = mean_absolute_error(all_true, all_preds)
    
    try:
        r, p_value = pearsonr(all_true, all_preds)
    except:
        r, p_value = 0.0, 1.0
    
    return {
        'nrmse': nrmse,
        'mae': mae,
        'pearson_r': r,
        'p_value': p_value
    }


def main(args):
    print("="*80)
    print("🎯 CHALLENGE 2: EXTERNALIZING FACTOR PREDICTION")
    print("="*80)
    print("Competition: https://eeg2025.github.io/")
    print("Metric: NRMSE (target < 0.5)")
    print("="*80)
    
    # Setup device with GPU detection
    if args.cpu_only:
        device = torch.device('cpu')
        print("\n🖥️  Using CPU")
    else:
        device, gpu_config = setup_device(
            gpu_id=args.gpu,
            force_sdk=args.use_sdk,
            optimize=True
        )
    
    # Create dataset
    print(f"\n📁 Loading data from: {args.data_dir}")
    dataset = ExternalizingDataset(args.data_dir, max_samples=args.max_samples)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\n📊 Dataset split:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == 'cuda')
    )
    
    # Create model
    print(f"\n🏗️  Creating model...")
    if TemporalConvNet is not None:
        model = TemporalConvNet(n_channels=129, n_outputs=1).to(device)
        print("   Using TemporalConvNet")
    else:
        model = SimplifiedTCN(n_channels=129, n_outputs=1).to(device)
        print("   Using SimplifiedTCN")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {n_params:,}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print("="*80)
    
    best_nrmse = float('inf')
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        
        epoch_time = time.time() - start_time
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val NRMSE:  {val_metrics['nrmse']:.4f}")
        print(f"  Val MAE:    {val_metrics['mae']:.2f}")
        print(f"  Pearson r:  {val_metrics['pearson_r']:.3f}")
        
        # Update learning rate
        scheduler.step(val_metrics['nrmse'])
        
        # Save best model
        if val_metrics['nrmse'] < best_nrmse:
            best_nrmse = val_metrics['nrmse']
            checkpoint_path = Path(args.output_dir) / "challenge2_best.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'nrmse': best_nrmse,
                'metrics': val_metrics
            }, checkpoint_path)
            print(f"  ✅ Saved best model (NRMSE: {best_nrmse:.4f})")
        
        print()
    
    print("="*80)
    print(f"✅ Training complete!")
    print(f"   Best NRMSE: {best_nrmse:.4f}")
    print(f"   Checkpoint: {checkpoint_path}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Challenge 2 GPU Training")
    parser.add_argument("--data-dir", type=str, default="data/hbn", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="outputs/challenge2", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU")
    parser.add_argument("--use-sdk", action="store_true", help="Use custom ROCm SDK")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples (for testing)")
    
    args = parser.parse_args()
    main(args)
