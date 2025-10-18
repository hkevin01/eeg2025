"""
Enhanced TCN Training Script
- Improved architecture with residual connections
- Better data augmentation
- Longer training with proper scheduling
- Memory-efficient implementation
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
print("ENHANCED TCN TRAINING - Full Architecture")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Device selection
device = torch.device('cpu')  # Use CPU for stability with limited GPU
print(f"Device: {device} (CPU training for memory safety)")
print()

# Enhanced configuration
CONFIG = {
    'model': {
        'num_channels': 129,
        'num_outputs': 1,
        'num_filters': 48,      # Increased from 32
        'kernel_size': 7,       # Increased from 5
        'num_levels': 5,        # Increased from 4
        'dropout': 0.3
    },
    'training': {
        'batch_size': 16,       # Increased from 8
        'accumulation_steps': 2, # Effective batch = 32
        'learning_rate': 2e-3,  # Higher initial LR
        'weight_decay': 1e-4,
        'epochs': 100,          # Much longer training
        'patience': 15,         # More patience
        'num_workers': 0
    },
    'data': {
        'n_train': 2000,        # More training data
        'n_val': 400,           # More validation data
        'seq_len': 200
    },
    'augmentation': {
        'noise_std': 0.05,      # Gaussian noise
        'scale_range': (0.9, 1.1),  # Random scaling
        'shift_range': (-10, 10)     # Time shifting
    }
}

print("📦 Creating Enhanced TCN Model...")
model = TCN_EEG(**CONFIG['model']).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {num_params:,}")
print(f"   Model size: {num_params * 4 / 1e6:.2f} MB (FP32)")
print(f"   Architecture: {CONFIG['model']['num_levels']} levels, {CONFIG['model']['num_filters']} filters\n")

# Data augmentation function
def augment_eeg(x, noise_std=0.05, scale_range=(0.9, 1.1), shift_range=(-10, 10)):
    """Apply realistic EEG augmentations"""
    # Gaussian noise
    if np.random.rand() > 0.5:
        noise = np.random.randn(*x.shape) * noise_std
        x = x + noise
    
    # Random scaling
    if np.random.rand() > 0.5:
        scale = np.random.uniform(*scale_range)
        x = x * scale
    
    # Time shifting (circular shift)
    if np.random.rand() > 0.5:
        shift = np.random.randint(*shift_range)
        x = np.roll(x, shift, axis=-1)
    
    return x

# Create enhanced synthetic data with realistic patterns
print("📊 Creating Enhanced Training Data...")
n_train = CONFIG['data']['n_train']
n_val = CONFIG['data']['n_val']
seq_len = CONFIG['data']['seq_len']

np.random.seed(42)

print("   Generating realistic EEG patterns...")
# Create base patterns with temporal structure
def create_eeg_data(n_samples):
    """Create synthetic EEG with realistic temporal patterns"""
    X = np.zeros((n_samples, 129, seq_len), dtype=np.float32)
    y = np.zeros((n_samples, 1), dtype=np.float32)
    
    for i in range(n_samples):
        # Base signal with multiple frequency components
        t = np.linspace(0, 2, seq_len)
        
        for ch in range(129):
            # Mix of frequencies (alpha, beta, theta rhythms)
            alpha = np.sin(2 * np.pi * 10 * t) * np.random.uniform(0.5, 1.5)
            beta = np.sin(2 * np.pi * 20 * t) * np.random.uniform(0.3, 1.0)
            theta = np.sin(2 * np.pi * 5 * t) * np.random.uniform(0.4, 1.2)
            
            # Add noise
            noise = np.random.randn(seq_len) * 0.2
            
            # Combine
            X[i, ch, :] = alpha + beta + theta + noise
        
        # Response time correlates with signal amplitude and frequency
        amplitude = np.mean(np.abs(X[i]))
        freq_power = np.mean(np.abs(np.fft.rfft(X[i, 0, :])))
        
        y[i, 0] = 0.5 + amplitude * 0.3 + freq_power * 0.01 + np.random.randn() * 0.1
        y[i, 0] = np.clip(y[i, 0], 0.2, 2.0)  # Realistic response time range
    
    return X, y

print("   Train data generation...")
X_train, y_train = create_eeg_data(n_train)

print("   Val data generation...")
X_val, y_val = create_eeg_data(n_val)

# Create datasets
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

# Clear from memory
del X_train, y_train, X_val, y_val
gc.collect()

print(f"   ✅ Train samples: {n_train}")
print(f"   ✅ Val samples: {n_val}")
print(f"   Data size: {(n_train + n_val) * 129 * seq_len * 4 / 1e6:.2f} MB\n")

# Data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=CONFIG['training']['batch_size'], 
    shuffle=True, 
    num_workers=CONFIG['training']['num_workers'],
    pin_memory=False
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
optimizer = optim.AdamW(  # AdamW for better generalization
    model.parameters(), 
    lr=CONFIG['training']['learning_rate'],
    weight_decay=CONFIG['training']['weight_decay']
)

# Cosine annealing with warm restarts
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

print(f"{'='*80}")
print(f"Training Configuration:")
print(f"{'='*80}")
print(f"Optimizer: AdamW")
print(f"Scheduler: CosineAnnealingWarmRestarts")
print(f"Batch size: {CONFIG['training']['batch_size']}")
print(f"Gradient accumulation: {CONFIG['training']['accumulation_steps']} steps")
print(f"Effective batch size: {CONFIG['training']['batch_size'] * CONFIG['training']['accumulation_steps']}")
print(f"Initial learning rate: {CONFIG['training']['learning_rate']}")
print(f"Epochs: {CONFIG['training']['epochs']}")
print(f"Patience: {CONFIG['training']['patience']}")
print(f"Data augmentation: ON")
print(f"{'='*80}\n")

best_val_loss = float('inf')
patience_counter = 0
history = []

print("🚀 Starting Enhanced Training...\n")

try:
    for epoch in range(CONFIG['training']['epochs']):
        epoch_start = datetime.now()
        
        print(f"{'='*80}")
        print(f"Epoch {epoch+1}/{CONFIG['training']['epochs']}")
        print(f"{'='*80}")
        
        # Training with augmentation
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Apply augmentation
            if np.random.rand() > 0.3:  # 70% of batches get augmentation
                data_np = data.numpy()
                for i in range(data_np.shape[0]):
                    data_np[i] = augment_eeg(
                        data_np[i], 
                        **CONFIG['augmentation']
                    )
                data = torch.FloatTensor(data_np)
            
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target) / CONFIG['training']['accumulation_steps']
            loss.backward()
            
            if (batch_idx + 1) % CONFIG['training']['accumulation_steps'] == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * CONFIG['training']['accumulation_steps']
            
            if (batch_idx + 1) % 25 == 0:
                print(f"   Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item() * CONFIG['training']['accumulation_steps']:.6f}")
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                val_predictions.extend(output.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Calculate correlation
        val_predictions = np.array(val_predictions).flatten()
        val_targets = np.array(val_targets).flatten()
        correlation = np.corrcoef(val_predictions, val_targets)[0, 1]
        
        print(f"\n📈 Train Loss: {train_loss:.6f}")
        print(f"📉 Val Loss:   {val_loss:.6f}")
        print(f"📊 Correlation: {correlation:.4f}")
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"🎓 Learning rate: {current_lr:.6f}")
        
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        print(f"⏱️  Epoch time: {epoch_time:.1f}s")
        
        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'correlation': correlation,
            'lr': current_lr,
            'epoch_time': epoch_time
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'correlation': correlation,
                'config': CONFIG,
                'history': history
            }
            
            Path('checkpoints').mkdir(exist_ok=True)
            torch.save(checkpoint, 'checkpoints/challenge1_tcn_enhanced_best.pth')
            print(f"✅ Saved best model (Val Loss: {val_loss:.6f}, Corr: {correlation:.4f})")
        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{CONFIG['training']['patience']}")
            
            if patience_counter >= CONFIG['training']['patience']:
                print(f"\n⛔ Early stopping triggered!")
                break
        
        # Periodic checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save(checkpoint, f'checkpoints/challenge1_tcn_enhanced_epoch{epoch+1}.pth')
            print(f"💾 Saved checkpoint at epoch {epoch+1}")
        
        gc.collect()
        print()

except KeyboardInterrupt:
    print(f"\n⚠️  Training interrupted by user")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Save final checkpoint
    checkpoint = {
        'epoch': epoch + 1 if 'epoch' in locals() else 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss if 'val_loss' in locals() else float('inf'),
        'config': CONFIG,
        'history': history
    }
    
    Path('checkpoints').mkdir(exist_ok=True)
    torch.save(checkpoint, 'checkpoints/challenge1_tcn_enhanced_final.pth')
    print("💾 Saved final checkpoint")

print(f"\n{'='*80}")
print("Training Complete!")
print(f"{'='*80}\n")
print(f"🏆 Best validation loss: {best_val_loss:.6f}")
print(f"📁 Model saved to: checkpoints/challenge1_tcn_enhanced_best.pth")
print(f"📊 Total epochs trained: {len(history)}")

# Save history
with open('checkpoints/challenge1_tcn_enhanced_history.json', 'w') as f:
    json.dump(history, f, indent=2)
print(f"📈 History saved to: checkpoints/challenge1_tcn_enhanced_history.json")

# Plot summary
if history:
    train_losses = [h['train_loss'] for h in history]
    val_losses = [h['val_loss'] for h in history]
    
    print(f"\n📊 Training Summary:")
    print(f"   Initial train loss: {train_losses[0]:.6f}")
    print(f"   Final train loss:   {train_losses[-1]:.6f}")
    print(f"   Initial val loss:   {val_losses[0]:.6f}")
    print(f"   Best val loss:      {best_val_loss:.6f}")
    print(f"   Improvement:        {(val_losses[0] - best_val_loss) / val_losses[0] * 100:.1f}%")

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n✅ Enhanced TCN training complete!")
print(f"📋 Note: Trained on synthetic data - for production, use real EEG data")
