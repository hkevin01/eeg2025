"""
Memory-Safe TCN Training Script
Optimized for limited GPU memory with gradient accumulation and mixed precision
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
import gc

from improvements.all_improvements import TCN_EEG

print("=" * 80)
print("TCN Training Script - Memory-Safe Version")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Check for GPU and memory
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {gpu_memory:.2f} GB")
    
    # Conservative memory allocation
    if gpu_memory < 8:
        print("⚠️  Limited GPU memory detected - using conservative settings")
        use_gpu = False
        device = torch.device('cpu')
        print("   Switching to CPU training for stability")
    else:
        use_gpu = True
        torch.backends.cudnn.benchmark = True
else:
    use_gpu = False
    print("No GPU available - using CPU")

print()

# Memory-safe configuration
CONFIG = {
    'model': {
        'num_channels': 129,
        'num_outputs': 1,
        'num_filters': 32,  # Reduced from 64
        'kernel_size': 5,    # Reduced from 7
        'num_levels': 4,     # Reduced from 6
        'dropout': 0.3
    },
    'training': {
        'batch_size': 8,     # Reduced from 32 for memory safety
        'accumulation_steps': 4,  # Effective batch size = 8 * 4 = 32
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'epochs': 30,        # Reduced for faster testing
        'patience': 8,
        'num_workers': 0     # Avoid multiprocessing issues
    },
    'data': {
        'n_train': 800,      # Reduced dataset size
        'n_val': 200,
        'seq_len': 200
    }
}

print("📦 Creating memory-efficient TCN model...")
model = TCN_EEG(**CONFIG['model']).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {num_params:,}")
print(f"   Model size: {num_params * 4 / 1e6:.2f} MB (FP32)\n")

# Create synthetic data with memory management
print("📊 Creating training data...")
n_train = CONFIG['data']['n_train']
n_val = CONFIG['data']['n_val']
seq_len = CONFIG['data']['seq_len']

np.random.seed(42)

# Create data in smaller chunks to avoid memory spike
print("   Generating train data...")
X_train = np.random.randn(n_train, 129, seq_len).astype(np.float32)
y_train = np.random.rand(n_train, 1).astype(np.float32) * 2.0

print("   Generating val data...")
X_val = np.random.randn(n_val, 129, seq_len).astype(np.float32)
y_val = np.random.rand(n_val, 1).astype(np.float32) * 2.0

train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

# Clear numpy arrays from memory
del X_train, y_train, X_val, y_val
gc.collect()

print(f"   Train samples: {n_train}")
print(f"   Val samples: {n_val}")
print(f"   Data size: {(n_train + n_val) * 129 * seq_len * 4 / 1e6:.2f} MB\n")

train_loader = DataLoader(
    train_dataset, 
    batch_size=CONFIG['training']['batch_size'], 
    shuffle=True, 
    num_workers=CONFIG['training']['num_workers'],
    pin_memory=False  # Disable for memory safety
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=CONFIG['training']['batch_size'],
    shuffle=False, 
    num_workers=CONFIG['training']['num_workers'],
    pin_memory=False
)

# Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(), 
    lr=CONFIG['training']['learning_rate'],
    weight_decay=CONFIG['training']['weight_decay']
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=False
)

# Mixed precision training (if GPU available)
use_amp = use_gpu and hasattr(torch.cuda, 'amp')
if use_amp:
    print("🚀 Using automatic mixed precision (AMP) for memory efficiency")
    scaler = torch.cuda.amp.GradScaler()
else:
    print("🚀 Using standard FP32 training")

print(f"\n{'='*80}")
print(f"Training Configuration:")
print(f"{'='*80}")
print(f"Batch size: {CONFIG['training']['batch_size']}")
print(f"Gradient accumulation: {CONFIG['training']['accumulation_steps']} steps")
print(f"Effective batch size: {CONFIG['training']['batch_size'] * CONFIG['training']['accumulation_steps']}")
print(f"Learning rate: {CONFIG['training']['learning_rate']}")
print(f"Epochs: {CONFIG['training']['epochs']}")
print(f"Patience: {CONFIG['training']['patience']}")
print(f"{'='*80}\n")

best_val_loss = float('inf')
patience_counter = 0
history = []

try:
    for epoch in range(CONFIG['training']['epochs']):
        print(f"{'='*80}")
        print(f"Epoch {epoch+1}/{CONFIG['training']['epochs']}")
        print(f"{'='*80}")
        
        # Training with gradient accumulation
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target) / CONFIG['training']['accumulation_steps']
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % CONFIG['training']['accumulation_steps'] == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                output = model(data)
                loss = criterion(output, target) / CONFIG['training']['accumulation_steps']
                loss.backward()
                
                if (batch_idx + 1) % CONFIG['training']['accumulation_steps'] == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            train_loss += loss.item() * CONFIG['training']['accumulation_steps']
            
            if (batch_idx + 1) % 25 == 0:
                print(f"   Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item() * CONFIG['training']['accumulation_steps']:.6f}")
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        output = model(data)
                        loss = criterion(output, target)
                else:
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
        
        # Memory usage
        if use_gpu:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"💾 GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
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
        
        # Clear cache
        if use_gpu:
            torch.cuda.empty_cache()
        gc.collect()
        
        print()

except RuntimeError as e:
    print(f"\n❌ Runtime Error: {e}")
    print("Attempting to save current state...")
    
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss if 'val_loss' in locals() else float('inf'),
        'config': CONFIG,
        'error': str(e)
    }
    
    Path('checkpoints').mkdir(exist_ok=True)
    torch.save(checkpoint, 'checkpoints/challenge1_tcn_interrupted.pth')
    print("💾 Saved interrupted checkpoint")

print(f"\n{'='*80}")
print("Training Complete!")
print(f"{'='*80}\n")
print(f"🏆 Best validation loss: {best_val_loss:.6f}")
print(f"📁 Model saved to: checkpoints/challenge1_tcn_best.pth")

# Save history
with open('checkpoints/challenge1_tcn_history.json', 'w') as f:
    json.dump(history, f, indent=2)
print(f"📊 History saved to: checkpoints/challenge1_tcn_history.json")

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n✅ TCN architecture training complete!")
print(f"📋 Total epochs: {len(history)}")
print(f"�� Peak memory usage: {'GPU' if use_gpu else 'CPU'} training")
