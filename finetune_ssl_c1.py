"""
Progressive Fine-tuning of SSL Pre-trained Encoder for Age Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_finetune_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


class CompactCNNEncoder(nn.Module):
    """Encoder architecture (same as SSL pre-training)"""
    
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


class FineTunedModel(nn.Module):
    """Full model with encoder + regression head"""
    
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        output = self.regressor(features)
        return output.squeeze()


class H5Dataset(Dataset):
    """Standard dataset for supervised training"""
    
    def __init__(self, h5_paths):
        self.data = []
        self.labels = []
        
        logger.info(f"Loading data from {len(h5_paths)} files...")
        for h5_path in h5_paths:
            with h5py.File(h5_path, 'r') as f:
                self.data.append(torch.FloatTensor(f['eeg'][:]))
                self.labels.append(torch.FloatTensor(f['labels'][:]))
        
        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)
        logger.info(f"Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for X, y in progress_bar:
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        pred = model(X)
        loss = criterion(pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            
            pred = model(X)
            loss = criterion(pred, y)
            
            total_loss += loss.item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mae = np.mean(np.abs(all_preds - all_targets))
    
    return avg_loss, rmse, mae


def phase1_train_head_only(model, train_loader, val_loader, epochs=5):
    """Phase 1: Freeze encoder, train only regression head"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: Training regression head only (encoder frozen)")
    logger.info("="*80)
    
    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # Only regressor parameters
    optimizer = torch.optim.AdamW(
        model.regressor.parameters(),
        lr=0.001,
        weight_decay=0.05
    )
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_rmse, val_mae = validate(model, val_loader, criterion)
        
        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"  Train Loss: {train_loss:.6f}")
        logger.info(f"  Val Loss: {val_loss:.6f}, RMSE: {val_rmse:.6f}, MAE: {val_mae:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/finetune_phase1_best.pt')
            logger.info(f"  💾 Saved best Phase 1 model")
    
    logger.info(f"\n✅ Phase 1 complete. Best Val Loss: {best_val_loss:.6f}")
    return model


def phase2_finetune_top_layers(model, train_loader, val_loader, epochs=10):
    """Phase 2: Unfreeze top conv block, fine-tune with lower LR"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: Fine-tuning top layers (Conv Block 3 + head)")
    logger.info("="*80)
    
    # Unfreeze only Conv Block 3
    for param in model.encoder.conv3.parameters():
        param.requires_grad = True
    for param in model.encoder.bn3.parameters():
        param.requires_grad = True
    
    # Lower learning rate
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001,  # 10x lower
        weight_decay=0.05
    )
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_rmse, val_mae = validate(model, val_loader, criterion)
        
        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"  Train Loss: {train_loss:.6f}")
        logger.info(f"  Val Loss: {val_loss:.6f}, RMSE: {val_rmse:.6f}, MAE: {val_mae:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'checkpoints/finetune_phase2_best.pt')
            logger.info(f"  💾 Saved best Phase 2 model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  ⏹️ Early stopping at epoch {epoch+1}")
                break
    
    logger.info(f"\n✅ Phase 2 complete. Best Val Loss: {best_val_loss:.6f}")
    return model


def phase3_full_finetune(model, train_loader, val_loader, epochs=15):
    """Phase 3: Unfreeze all layers, full fine-tuning with very low LR"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: Full fine-tuning (all layers)")
    logger.info("="*80)
    
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    # Very low learning rate
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.00005,  # 20x lower than initial
        weight_decay=0.05
    )
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3
    )
    
    best_val_loss = float('inf')
    patience = 8
    patience_counter = 0
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_rmse, val_mae = validate(model, val_loader, criterion)
        
        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"  Train Loss: {train_loss:.6f}")
        logger.info(f"  Val Loss: {val_loss:.6f}, RMSE: {val_rmse:.6f}, MAE: {val_mae:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'checkpoints/finetune_phase3_best.pt')
            logger.info(f"  💾 Saved best Phase 3 model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  ⏹️ Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step(val_loss)
    
    logger.info(f"\n✅ Phase 3 complete. Best Val Loss: {best_val_loss:.6f}")
    logger.info(f"   Best Val NRMSE: {np.sqrt(best_val_loss):.6f}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str, 
                       default='checkpoints/ssl_encoder_final.pt',
                       help='Path to pre-trained encoder')
    parser.add_argument('--phase', type=str, default='all',
                       choices=['all', 'phase1', 'phase2', 'phase3'],
                       help='Which phase to run')
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("Progressive Fine-tuning of SSL Pre-trained Encoder")
    logger.info("="*80)
    
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
    logger.info("\n📦 Loading datasets...")
    train_dataset = H5Dataset(train_files)
    val_dataset = H5Dataset(val_files)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Load pre-trained encoder
    logger.info(f"\n🏗️ Loading pre-trained encoder from {args.pretrained}...")
    encoder = CompactCNNEncoder(in_channels=129)
    
    try:
        encoder.load_state_dict(torch.load(args.pretrained))
        logger.info("✅ Loaded pre-trained weights")
    except:
        logger.warning("⚠️ Could not load pre-trained weights, using random init")
    
    # Create full model
    model = FineTunedModel(encoder).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Run progressive fine-tuning
    if args.phase == 'all' or args.phase == 'phase1':
        model = phase1_train_head_only(model, train_loader, val_loader, epochs=5)
    
    if args.phase == 'all' or args.phase == 'phase2':
        # Load best from phase 1 if we're continuing
        if args.phase == 'phase2':
            model.load_state_dict(torch.load('checkpoints/finetune_phase1_best.pt'))
        model = phase2_finetune_top_layers(model, train_loader, val_loader, epochs=10)
    
    if args.phase == 'all' or args.phase == 'phase3':
        # Load best from phase 2 if we're continuing
        if args.phase == 'phase3':
            model.load_state_dict(torch.load('checkpoints/finetune_phase2_best.pt'))
        model = phase3_full_finetune(model, train_loader, val_loader, epochs=15)
    
    # Final evaluation
    logger.info("\n" + "="*80)
    logger.info("FINAL EVALUATION")
    logger.info("="*80)
    
    # Load best model from phase 3
    model.load_state_dict(torch.load('checkpoints/finetune_phase3_best.pt'))
    criterion = nn.MSELoss()
    val_loss, val_rmse, val_mae = validate(model, val_loader, criterion)
    
    logger.info(f"Final Val Loss (MSE): {val_loss:.6f}")
    logger.info(f"Final Val NRMSE: {val_rmse:.6f}")
    logger.info(f"Final Val MAE: {val_mae:.6f}")
    
    # Save final model
    torch.save(model.state_dict(), 'checkpoints/ssl_finetuned_final.pt')
    logger.info("\n💾 Saved final fine-tuned model to checkpoints/ssl_finetuned_final.pt")
    
    logger.info("\n" + "="*80)
    logger.info("✅ Progressive fine-tuning complete!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
