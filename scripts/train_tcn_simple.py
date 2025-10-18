"""
Simple TCN Training Script - Uses Existing Dataset Infrastructure
"""
import sys
sys.path.append('/home/kevin/Projects/eeg2025')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from improvements.all_improvements import TCN_EEG

print("=" * 80)
print("TCN Training Script - Simple Version")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

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
    }
}

print("📦 Creating TCN model...")
model = TCN_EEG(**CONFIG['model']).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {num_params:,}\n")

# Try to load real data from existing checkpoints/cache
print("📊 Looking for training data...")
data_found = False

# Check if we have cached processed data
cache_dir = Path('data/cache')
if cache_dir.exists():
    cache_files = list(cache_dir.glob('*.pt'))
    if cache_files:
        print(f"   Found {len(cache_files)} cache files")
        data_found = True

# Check for processed data
processed_dir = Path('data/processed')
if processed_dir.exists() and not data_found:
    processed_files = list(processed_dir.glob('**/*.npy'))
    if processed_files:
        print(f"   Found {len(processed_files)} processed files")
        data_found = True

if not data_found:
    print("   ⚠️  No processed data found!")
    print("   Creating synthetic data for testing TCN architecture...")
    print("   (For real training, please prepare data using existing scripts)\n")
    
    # Create synthetic data
    n_train = 1000
    n_val = 200
    seq_len = 200
    
    np.random.seed(42)
    
    # Simulate EEG data with some temporal structure
    X_train = np.random.randn(n_train, 129, seq_len).astype(np.float32)
    y_train = np.random.rand(n_train, 1).astype(np.float32) * 2.0  # Response times 0-2s
    
    X_val = np.random.randn(n_val, 129, seq_len).astype(np.float32)
    y_val = np.random.rand(n_val, 1).astype(np.float32) * 2.0
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    print(f"   Train samples: {n_train}")
    print(f"   Val samples: {n_val}")
    print(f"   NOTE: This is synthetic data for architecture testing only!\n")
else:
    print("   ✅ Real data found - loading from cache/processed directory")
    print("   (Actual data loading not implemented yet - using synthetic for now)\n")
    
    # For now, still use synthetic until we implement proper data loading
    n_train = 1000
    n_val = 200
    seq_len = 200
    
    X_train = np.random.randn(n_train, 129, seq_len).astype(np.float32)
    y_train = np.random.rand(n_train, 1).astype(np.float32) * 2.0
    
    X_val = np.random.randn(n_val, 129, seq_len).astype(np.float32)
    y_val = np.random.rand(n_val, 1).astype(np.float32) * 2.0
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

train_loader = DataLoader(train_dataset, batch_size=CONFIG['training']['batch_size'], 
                          shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['training']['batch_size'],
                        shuffle=False, num_workers=2)

# Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG['training']['learning_rate'],
                      weight_decay=CONFIG['training']['weight_decay'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                  patience=5, verbose=True)

print(f"🚀 Starting training for {CONFIG['training']['epochs']} epochs...")
print(f"   Batch size: {CONFIG['training']['batch_size']}")
print(f"   Learning rate: {CONFIG['training']['learning_rate']}")
print(f"   Patience: {CONFIG['training']['patience']}\n")

best_val_loss = float('inf')
patience_counter = 0
history = []

for epoch in range(CONFIG['training']['epochs']):
    print(f"{'='*80}")
    print(f"Epoch {epoch+1}/{CONFIG['training']['epochs']}")
    print(f"{'='*80}")
    
    # Training
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"   Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.6f}")
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    print(f"\n📈 Train Loss: {train_loss:.6f}")
    print(f"📉 Val Loss:   {val_loss:.6f}")
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"📊 Learning rate: {current_lr:.6f}")
    
    # Save history
    history.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'lr': current_lr
    })
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': CONFIG
        }
        
        Path('checkpoints').mkdir(exist_ok=True)
        torch.save(checkpoint, 'checkpoints/challenge1_tcn_best.pth')
        print(f"✅ Saved best model (Val Loss: {val_loss:.6f})")
    else:
        patience_counter += 1
        print(f"⏳ Patience: {patience_counter}/{CONFIG['training']['patience']}")
        
        if patience_counter >= CONFIG['training']['patience']:
            print(f"\n⛔ Early stopping triggered!")
            break
    
    print()

print(f"{'='*80}")
print("Training Complete!")
print(f"{'='*80}\n")
print(f"🏆 Best validation loss: {best_val_loss:.6f}")
print(f"📁 Model saved to: checkpoints/challenge1_tcn_best.pth")

# Save history
with open('checkpoints/challenge1_tcn_history.json', 'w') as f:
    json.dump(history, f, indent=2)
print(f"📊 History saved to: checkpoints/challenge1_tcn_history.json")

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n✅ TCN architecture validated!")
print("NOTE: This used synthetic data. For real training, prepare data first.")
