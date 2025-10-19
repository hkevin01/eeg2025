"""
Simple HDF5-based Challenge 1 training script.
Uses cached HDF5 files instead of loading full dataset into RAM.
"""
import sys
import os
sys.path.append('src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Import HDF5Dataset
from utils.hdf5_dataset import HDF5Dataset

# Setup logging
log_dir = Path("logs/training_comparison")
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"challenge1_hdf5_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Simple CNN Model (from existing script)
class SimpleCNN(nn.Module):
    def __init__(self, n_channels=21, n_timepoints=200):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool1d(2)
        
        self.fc1 = nn.Linear(256 * (n_timepoints // 8), 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def normalize_batch(X):
    """Normalize each channel"""
    mean = X.mean(dim=2, keepdim=True)
    std = X.std(dim=2, keepdim=True)
    return (X - mean) / (std + 1e-8)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (X, y) in enumerate(loader):
        # Normalize
        X = normalize_batch(X)
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output.squeeze(), y.squeeze())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 100 == 0:
            logger.info(f"  Batch {batch_idx+1}/{len(loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for X, y in loader:
            X = normalize_batch(X)
            X, y = X.to(device), y.to(device)
            
            output = model(X)
            loss = criterion(output.squeeze(), y.squeeze())
            total_loss += loss.item()
            
            predictions.append(output.cpu().numpy())
            targets.append(y.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    # Calculate NRMSE
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    target_std = np.std(targets)
    nrmse = rmse / target_std if target_std > 0 else float('inf')
    
    return total_loss / len(loader), nrmse

def main():
    logger.info("="*80)
    logger.info("Challenge 1 Training - HDF5 Memory-Efficient Version")
    logger.info("="*80)
    
    # Load HDF5 files
    hdf5_files = [
        "data/cached/challenge1_R1_windows.h5",
        "data/cached/challenge1_R2_windows.h5",
        "data/cached/challenge1_R3_windows.h5",
        "data/cached/challenge1_R4_windows.h5",
    ]
    
    logger.info("\nLoading HDF5 dataset...")
    dataset = HDF5Dataset(hdf5_files)
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"Train size: {train_size}, Val size: {val_size}")
    
    # DataLoaders (num_workers=0 to avoid multiprocessing issues with HDF5)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = SimpleCNN(n_channels=21, n_timepoints=200).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    # Training loop
    num_epochs = 30
    best_nrmse = float('inf')
    
    logger.info("\nStarting training...")
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info("-" * 40)
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_nrmse = validate(model, val_loader, criterion, device)
        
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val NRMSE: {val_nrmse:.4f}")
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            torch.save(model.state_dict(), "weights_challenge_1_hdf5.pt")
            logger.info(f"✓ Saved best model (NRMSE: {best_nrmse:.4f})")
    
    logger.info("\n" + "="*80)
    logger.info(f"Training complete! Best Val NRMSE: {best_nrmse:.4f}")
    logger.info(f"Model saved to: weights_challenge_1_hdf5.pt")
    logger.info("="*80)

if __name__ == "__main__":
    main()
