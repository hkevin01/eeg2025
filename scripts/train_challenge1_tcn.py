"""
Train Challenge 1 with TCN (Temporal Convolutional Network)
Expected 15-20% improvement over CNN baseline
Training time: 4-8 hours on GPU
"""
import sys
sys.path.append('/home/kevin/Projects/eeg2025')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from improvements.all_improvements import TCN_EEG
from src.dataio.dataset_challenge1 import Challenge1Dataset

# Configuration
CONFIG = {
    'model': {
        'num_channels': 129,
        'num_outputs': 1,
        'num_filters': 64,
        'kernel_size': 7,
        'num_levels': 6,
        'dropout': 0.3
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'epochs': 50,
        'patience': 10
    },
    'data': {
        'train_path': 'data/challenge1_release1_train.npz',
        'val_path': 'data/challenge1_release1_val.npz'
    }
}

def calculate_nrmse(predictions, targets):
    """Calculate Normalized RMSE"""
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    nrmse = rmse / (np.max(targets) - np.min(targets))
    return nrmse

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(output.detach().cpu().numpy())
        all_targets.extend(target.detach().cpu().numpy())
        
        if (batch_idx + 1) % 100 == 0:
            print(f'   Batch {batch_idx+1}/{len(loader)}, Loss: {loss.item():.6f}')
    
    avg_loss = total_loss / len(loader)
    nrmse = calculate_nrmse(np.array(all_preds), np.array(all_targets))
    
    return avg_loss, nrmse

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    nrmse = calculate_nrmse(np.array(all_preds), np.array(all_targets))
    
    return avg_loss, nrmse

def main():
    print("=" * 80)
    print("Training Challenge 1 with TCN Architecture")
    print("=" * 80)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create model
    print("\n📦 Creating TCN model...")
    model = TCN_EEG(**CONFIG['model']).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,}")
    
    # Load data
    print("\n📊 Loading data...")
    try:
        train_dataset = Challenge1Dataset(CONFIG['data']['train_path'])
        val_dataset = Challenge1Dataset(CONFIG['data']['val_path'])
    except:
        print("   ⚠️  Dataset files not found, using dummy data for testing")
        print("   Please run proper data preparation first!")
        return
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # Optimizer and loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['training']['learning_rate'],
        weight_decay=CONFIG['training']['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print(f"\n🚀 Starting training for {CONFIG['training']['epochs']} epochs...")
    print(f"   Patience: {CONFIG['training']['patience']} epochs")
    
    best_val_nrmse = float('inf')
    patience_counter = 0
    history = []
    
    for epoch in range(CONFIG['training']['epochs']):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{CONFIG['training']['epochs']}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_nrmse = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"\n📈 Train - Loss: {train_loss:.6f}, NRMSE: {train_nrmse:.6f}")
        
        # Validate
        val_loss, val_nrmse = validate(model, val_loader, criterion, device)
        print(f"📉 Val   - Loss: {val_loss:.6f}, NRMSE: {val_nrmse:.6f}")
        
        # Learning rate scheduling
        scheduler.step(val_nrmse)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"📊 Learning rate: {current_lr:.6f}")
        
        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_nrmse': train_nrmse,
            'val_loss': val_loss,
            'val_nrmse': val_nrmse,
            'lr': current_lr
        })
        
        # Save best model
        if val_nrmse < best_val_nrmse:
            best_val_nrmse = val_nrmse
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': val_nrmse,
                'config': CONFIG
            }
            
            Path('checkpoints').mkdir(exist_ok=True)
            torch.save(checkpoint, 'checkpoints/challenge1_tcn_best.pth')
            print(f"✅ Saved best model (NRMSE: {val_nrmse:.6f})")
        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{CONFIG['training']['patience']}")
            
            if patience_counter >= CONFIG['training']['patience']:
                print(f"\n⛔ Early stopping triggered!")
                break
    
    # Save final results
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"\n🏆 Best validation NRMSE: {best_val_nrmse:.6f}")
    print(f"📁 Model saved to: checkpoints/challenge1_tcn_best.pth")
    
    # Save training history
    with open('checkpoints/challenge1_tcn_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"📊 History saved to: checkpoints/challenge1_tcn_history.json")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
