#!/usr/bin/env python3
"""
CPU-ONLY Safe Training Script
==============================

COMPLETELY DISABLES GPU to prevent system crashes.
Uses only CPU for all operations.
"""
import os
import sys
from pathlib import Path

# FORCE CPU ONLY - DISABLE ALL GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HIP_VISIBLE_DEVICES'] = ''
os.environ['ROCR_VISIBLE_DEVICES'] = ''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.models.eeg_dataset_simple import SimpleEEGDataset

# Verify CPU only
print("="*80)
print("🛡️  CPU-ONLY SAFE MODE")
print("="*80)
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: CPU (GPU completely disabled)")
print("="*80)

class SimpleCPUModel(nn.Module):
    """Simple CPU-only model"""
    def __init__(self, n_channels=129):
        super().__init__()
        self.conv = nn.Conv1d(n_channels, 64, kernel_size=7, padding=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x.squeeze(-1)

def train_cpu_only():
    """Train on CPU only"""
    print("\n🔥 Starting CPU-Only Training")
    
    # Config
    batch_size = 8  # Small for CPU
    epochs = 3
    max_samples = 500
    
    # Load data
    data_dir = Path(__file__).parent.parent / "data" / "raw" / "hbn"
    participants_file = data_dir / "participants.tsv"
    
    if not participants_file.exists():
        print("❌ No data found")
        return
        
    participants_df = pd.read_csv(participants_file, sep='\t')
    label_dict = dict(zip(participants_df['participant_id'], participants_df['age']))
    
    print(f"📊 Loaded {len(participants_df)} participants")
    
    # Load dataset
    full_dataset = SimpleEEGDataset(data_dir=data_dir, max_subjects=None)
    
    # Get valid samples
    valid_indices = []
    valid_labels = []
    
    for i, (data, window_label) in enumerate(full_dataset):
        if i >= max_samples:
            break
        # Use the window label directly since it's from labeled data
        valid_indices.append(i)
        valid_labels.append(window_label)
    
    print(f"✅ Found {len(valid_indices)} samples")
    
    # Split
    split_idx = int(0.8 * len(valid_indices))
    train_subset = Subset(full_dataset, valid_indices[:split_idx])
    val_subset = Subset(full_dataset, valid_indices[split_idx:])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    print(f"Train: {len(train_subset)}, Val: {len(val_subset)}")
    
    # Create model - CPU ONLY
    model = SimpleCPUModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    for epoch in range(epochs):
        print(f"\n📍 Epoch {epoch+1}/{epochs}")
        
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            # All on CPU
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.detach().numpy())
            train_labels.extend(labels.numpy())
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")
        
        # Validate
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for data, labels in val_loader:
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(outputs.numpy())
                val_labels.extend(labels.numpy())
        
        # Metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_corr, _ = pearsonr(train_preds, train_labels)
        val_corr, _ = pearsonr(val_preds, val_labels)
        
        print(f"✅ Train Loss: {train_loss:.4f}, Corr: {train_corr:.4f}")
        print(f"✅ Val Loss: {val_loss:.4f}, Corr: {val_corr:.4f}")
    
    print("\n🎉 Training completed safely on CPU!")

if __name__ == "__main__":
    train_cpu_only()
