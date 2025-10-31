"""
Self-Supervised Pre-training for Challenge 1
Uses SimCLR (Simple Framework for Contrastive Learning) adapted for EEG signals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import random
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_ssl_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


class ContrastiveAugmentation:
    """Strong augmentation pipeline for contrastive learning"""
    
    def __init__(self, 
                 time_shift_prob=0.8, 
                 channel_dropout_prob=0.5,
                 temporal_mask_prob=0.5,
                 amplitude_scale_prob=0.7,
                 noise_prob=0.6):
        self.time_shift_prob = time_shift_prob
        self.channel_dropout_prob = channel_dropout_prob
        self.temporal_mask_prob = temporal_mask_prob
        self.amplitude_scale_prob = amplitude_scale_prob
        self.noise_prob = noise_prob
    
    def __call__(self, x):
        """Apply random augmentation to EEG signal
        
        Args:
            x: Tensor of shape (channels, timepoints)
        
        Returns:
            Augmented tensor of same shape
        """
        x = x.clone()
        
        # 1. Time shift: ±20 samples
        if random.random() < self.time_shift_prob:
            shift = random.randint(-20, 20)
            x = torch.roll(x, shifts=shift, dims=-1)
        
        # 2. Channel dropout: Drop 20-40% channels
        if random.random() < self.channel_dropout_prob:
            n_channels = x.shape[0]
            n_drop = random.randint(int(0.2 * n_channels), int(0.4 * n_channels))
            drop_idx = random.sample(range(n_channels), n_drop)
            x[drop_idx, :] = 0
        
        # 3. Temporal masking: Mask 20-40% timepoints
        if random.random() < self.temporal_mask_prob:
            n_timepoints = x.shape[1]
            mask_len = random.randint(int(0.2 * n_timepoints), int(0.4 * n_timepoints))
            start = random.randint(0, n_timepoints - mask_len)
            x[:, start:start+mask_len] = 0
        
        # 4. Amplitude scaling: 0.7-1.3x
        if random.random() < self.amplitude_scale_prob:
            scale = random.uniform(0.7, 1.3)
            x = x * scale
        
        # 5. Gaussian noise: σ=0.03
        if random.random() < self.noise_prob:
            noise = torch.randn_like(x) * 0.03
            x = x + noise
        
        return x


class ContrastiveEEGDataset(Dataset):
    """Dataset for contrastive learning - returns two augmented views of each sample"""
    
    def __init__(self, h5_paths, augmentation):
        self.data = []
        self.augmentation = augmentation
        
        logger.info(f"Loading data from {len(h5_paths)} files...")
        for h5_path in h5_paths:
            with h5py.File(h5_path, 'r') as f:
                eeg_data = f['eeg'][:]
                self.data.append(torch.FloatTensor(eeg_data))
        
        self.data = torch.cat(self.data, dim=0)
        logger.info(f"Loaded {len(self.data)} samples, shape: {self.data.shape}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        
        # Create two different augmented views
        x1 = self.augmentation(x)
        x2 = self.augmentation(x)
        
        return x1, x2


class CompactCNNEncoder(nn.Module):
    """Encoder based on V8 CompactCNN architecture"""
    
    def __init__(self, in_channels=129):
        super().__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(0.5)
        
        # Conv Block 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(0.6)
        
        # Conv Block 3
        self.conv3 = nn.Conv1d(64, 96, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(96)
        self.pool3 = nn.MaxPool1d(2)
        self.drop3 = nn.Dropout(0.7)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.drop3(x)
        
        # Global pooling
        x = self.adaptive_pool(x)
        x = x.flatten(1)  # (batch, 96)
        
        return x


class SimCLR_EEG(nn.Module):
    """SimCLR model for EEG with projection head"""
    
    def __init__(self, encoder, projection_dim=64):
        super().__init__()
        self.encoder = encoder
        
        # Projection head (MLP with one hidden layer)
        self.projector = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Linear(128, projection_dim)
        )
    
    def forward(self, x):
        # Encode
        features = self.encoder(x)
        
        # Project
        projection = self.projector(features)
        
        # L2 normalize
        projection = F.normalize(projection, dim=1)
        
        return projection


def nt_xent_loss(z1, z2, temperature=0.5):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss
    
    Args:
        z1: Projections from first view (batch_size, projection_dim)
        z2: Projections from second view (batch_size, projection_dim)
        temperature: Temperature parameter
    
    Returns:
        Contrastive loss
    """
    batch_size = z1.shape[0]
    
    # Concatenate z1 and z2
    z = torch.cat([z1, z2], dim=0)  # (2*batch_size, projection_dim)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(z, z.t()) / temperature  # (2*batch_size, 2*batch_size)
    
    # Create mask for positive pairs
    pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
    for i in range(batch_size):
        pos_mask[i, batch_size + i] = True
        pos_mask[batch_size + i, i] = True
    
    # Create mask for negative pairs (exclude diagonal and positive pairs)
    neg_mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    neg_mask = neg_mask & ~pos_mask
    
    # Compute loss using LogSumExp trick for numerical stability
    exp_sim = torch.exp(sim_matrix)
    
    # For each sample, compute: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
    loss = 0
    for i in range(2 * batch_size):
        pos_sim = exp_sim[i][pos_mask[i]].sum()
        neg_sim = exp_sim[i][neg_mask[i]].sum()
        loss += -torch.log(pos_sim / (pos_sim + neg_sim))
    
    return loss / (2 * batch_size)


def linear_probe_eval(encoder, train_loader, val_loader, epochs=10):
    """Evaluate learned representations with linear probe
    
    Freeze encoder and train only a linear classifier on top
    """
    logger.info("Running linear probe evaluation...")
    
    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False
    
    # Linear probe
    probe = nn.Linear(96, 1).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        probe.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Extract features
            with torch.no_grad():
                features = encoder(x)
            
            # Predict
            pred = probe(features).squeeze()
            loss = F.mse_loss(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        probe.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                features = encoder(x)
                pred = probe(features).squeeze()
                loss = F.mse_loss(pred, y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        logger.info(f"  Probe Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Unfreeze encoder
    for param in encoder.parameters():
        param.requires_grad = True
    
    val_nrmse = np.sqrt(best_val_loss)
    logger.info(f"Linear probe best Val NRMSE: {val_nrmse:.6f}")
    return val_nrmse


class SupervisedDataset(Dataset):
    """Standard supervised dataset for linear probe"""
    
    def __init__(self, h5_paths):
        self.data = []
        self.labels = []
        
        for h5_path in h5_paths:
            with h5py.File(h5_path, 'r') as f:
                self.data.append(torch.FloatTensor(f['eeg'][:]))
                self.labels.append(torch.FloatTensor(f['labels'][:]))
        
        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train_ssl(model, train_loader, optimizer, scheduler, epochs=50, 
              probe_train_loader=None, probe_val_loader=None):
    """Train SSL model with contrastive learning"""
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for x1, x2 in progress_bar:
            x1, x2 = x1.to(device), x2.to(device)
            
            # Forward pass
            z1 = model(x1)
            z2 = model(x2)
            
            # Compute contrastive loss
            loss = nt_xent_loss(z1, z2, temperature=0.5)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{epochs} - Contrastive Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'checkpoints/ssl_pretrained_best.pt')
            logger.info(f"  💾 Saved best model (loss: {best_loss:.4f})")
        
        # Linear probe evaluation every 5 epochs
        if (epoch + 1) % 5 == 0 and probe_train_loader is not None:
            probe_nrmse = linear_probe_eval(
                model.encoder, 
                probe_train_loader, 
                probe_val_loader, 
                epochs=10
            )
            logger.info(f"  🔬 Linear probe Val NRMSE: {probe_nrmse:.6f}")
        
        # Step scheduler
        scheduler.step()
    
    logger.info(f"\n✅ Training complete! Best contrastive loss: {best_loss:.4f}")
    return model


def main():
    logger.info("="*80)
    logger.info("Self-Supervised Pre-training for Challenge 1 (SimCLR)")
    logger.info("="*80)
    
    # Hyperparameters
    batch_size = 256
    epochs = 50
    learning_rate = 0.001
    projection_dim = 64
    
    # Data paths
    data_dir = Path('data/processed_tuab_challenge_1/')
    train_files = [
        data_dir / 'train_r1.h5',
        data_dir / 'train_r2.h5',
        data_dir / 'train_r3.h5',
    ]
    val_files = [data_dir / 'train_r4.h5']
    
    # Create checkpoint directory
    Path('checkpoints').mkdir(exist_ok=True)
    
    # Create datasets
    logger.info("\n📦 Creating contrastive datasets...")
    augmentation = ContrastiveAugmentation()
    ssl_train_dataset = ContrastiveEEGDataset(train_files, augmentation)
    ssl_train_loader = DataLoader(
        ssl_train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"SSL training samples: {len(ssl_train_dataset)}")
    
    # Create supervised datasets for linear probe evaluation
    logger.info("\n📦 Creating supervised datasets for linear probe...")
    probe_train_dataset = SupervisedDataset(train_files)
    probe_val_dataset = SupervisedDataset(val_files)
    probe_train_loader = DataLoader(
        probe_train_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4
    )
    probe_val_loader = DataLoader(
        probe_val_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4
    )
    
    logger.info(f"Probe train: {len(probe_train_dataset)}, val: {len(probe_val_dataset)}")
    
    # Create model
    logger.info("\n🏗️ Building SimCLR model...")
    encoder = CompactCNNEncoder(in_channels=129)
    model = SimCLR_EEG(encoder, projection_dim=projection_dim).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=0.0001
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs
    )
    
    # Train
    logger.info("\n🚀 Starting self-supervised pre-training...")
    trained_model = train_ssl(
        model, 
        ssl_train_loader, 
        optimizer, 
        scheduler, 
        epochs=epochs,
        probe_train_loader=probe_train_loader,
        probe_val_loader=probe_val_loader
    )
    
    # Save final encoder
    torch.save(
        trained_model.encoder.state_dict(),
        'checkpoints/ssl_encoder_final.pt'
    )
    logger.info("\n💾 Saved final encoder to checkpoints/ssl_encoder_final.pt")
    
    logger.info("\n" + "="*80)
    logger.info("✅ Self-supervised pre-training complete!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
