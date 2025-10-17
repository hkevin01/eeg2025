#!/usr/bin/env python3
"""
Foundation Model Training - Scaled Up
With GPU safeguards and auto-fallback to CPU
"""

import os
import sys
from pathlib import Path

# GPU Safety - set BEFORE importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import time
import json
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

print("🚀 Foundation Model Training")
print("=" * 70)

# Configuration - SCALED UP
CONFIG = {
    'data_dir': Path(__file__).parent.parent / "data" / "raw" / "hbn",
    'output_dir': Path(__file__).parent.parent / "outputs",
    'checkpoint_dir': Path(__file__).parent.parent / "checkpoints",
    'log_dir': Path(__file__).parent.parent / "logs",
    
    # Data
    'max_subjects': None,  # Use ALL subjects
    'train_split': 0.8,
    
    # Model - SCALED UP
    'hidden_dim': 128,  # 64 -> 128
    'n_heads': 8,       # 4 -> 8
    'n_layers': 4,      # 2 -> 4
    'dropout': 0.1,
    
    # Training
    'batch_size': 16,   # Larger batches
    'epochs': 20,       # More epochs
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    
    # Safety
    'gradient_clip': 1.0,
    'save_every': 2,
    'gpu_timeout': 10,  # 10 second timeout for GPU test
}

# Create directories
for dir_path in [CONFIG['output_dir'], CONFIG['checkpoint_dir'], CONFIG['log_dir']]:
    dir_path.mkdir(exist_ok=True)

print(f"📁 Directories ready")
print(f"=" * 70)


# GPU Safety Function with Timeout
def setup_device_safe():
    """Setup device with GPU timeout protection"""
    from multiprocessing import Process, Queue
    import queue
    
    def test_gpu(result_queue):
        """Test GPU in separate process"""
        try:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                # Test tensor operations
                test = torch.randn(100, 100).to(device)
                result = test @ test.T
                result_queue.put(('success', device))
            else:
                result_queue.put(('no_cuda', None))
        except Exception as e:
            result_queue.put(('error', str(e)))
    
    print("\n🔧 Device Setup (with GPU safeguards)")
    print("-" * 70)
    
    # Try GPU with timeout
    if torch.cuda.is_available():
        print("   GPU detected, testing with timeout...")
        result_queue = Queue()
        gpu_process = Process(target=test_gpu, args=(result_queue,))
        gpu_process.start()
        gpu_process.join(timeout=CONFIG['gpu_timeout'])
        
        if gpu_process.is_alive():
            print("   ⚠️  GPU test timed out - using CPU for safety")
            gpu_process.terminate()
            gpu_process.join()
            return torch.device('cpu')
        
        try:
            status, result = result_queue.get_nowait()
            if status == 'success':
                print("   ✅ GPU test passed!")
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
                return result
            else:
                print(f"   ⚠️  GPU test failed: {result}")
                print("   Using CPU for safety")
                return torch.device('cpu')
        except queue.Empty:
            print("   ⚠️  GPU test returned no result - using CPU")
            return torch.device('cpu')
    else:
        print("   No GPU available, using CPU")
        return torch.device('cpu')


# Model Definition - SCALED UP
class FoundationTransformer(nn.Module):
    """Scaled up transformer for foundation model"""
    def __init__(self, n_channels=129, seq_len=1000, 
                 hidden_dim=128, n_heads=8, n_layers=4, 
                 n_classes=2, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(n_channels, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,  # Standard 4x expansion
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x):
        # x: (batch, channels, time)
        x = x.transpose(1, 2)  # (batch, time, channels)
        x = self.input_proj(x)  # (batch, time, hidden)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.classifier(x)
        return x


# Training Functions
def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    print(f"\n📚 Epoch {epoch+1} - Training")
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip'])
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"   Batch {batch_idx+1}/{len(loader)}: "
                  f"loss={loss.item():.4f}, "
                  f"acc={100.*correct/total:.1f}%")
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device, epoch):
    """Validate"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    print(f"\n🔍 Epoch {epoch+1} - Validation")
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


# Main Training
def main():
    start_time = time.time()
    
    # Setup device
    device = setup_device_safe()
    print(f"\n📱 Using: {device}")
    print("=" * 70)
    
    # Load dataset
    print("\n📂 Loading Dataset")
    from scripts.models.eeg_dataset_simple import SimpleEEGDataset
    
    dataset = SimpleEEGDataset(
        data_dir=CONFIG['data_dir'],
        max_subjects=CONFIG['max_subjects']
    )
    
    print(f"   Total windows: {len(dataset)}")
    
    # Split
    train_size = int(CONFIG['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=4)
    
    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create model
    print("\n🧠 Creating Model")
    sample_data, _ = dataset[0]
    n_channels, seq_len = sample_data.shape
    
    model = FoundationTransformer(
        n_channels=n_channels,
        seq_len=seq_len,
        hidden_dim=CONFIG['hidden_dim'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,} (~{n_params*4/(1024**2):.1f} MB)")
    
    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=CONFIG['learning_rate'],
                                  weight_decay=CONFIG['weight_decay'])
    
    # Training loop
    print("\n" + "=" * 70)
    print("🏋️  Starting Training")
    print("=" * 70)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(CONFIG['epochs']):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        
        # Record
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"   Train: loss={train_loss:.4f}, acc={train_acc:.1f}%")
        print(f"   Val:   loss={val_loss:.4f}, acc={val_acc:.1f}%")
        
        # Save
        if (epoch + 1) % CONFIG['save_every'] == 0 or val_loss < best_val_loss:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': CONFIG
            }
            
            path = CONFIG['checkpoint_dir'] / f"foundation_epoch_{epoch+1}.pth"
            torch.save(checkpoint, path)
            print(f"   💾 Saved: {path.name}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = CONFIG['checkpoint_dir'] / "foundation_best.pth"
                torch.save(checkpoint, best_path)
                print(f"   ⭐ Best model!")
    
    # Finish
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("✅ Training Complete!")
    print(f"{'='*70}")
    print(f"   Time: {total_time/60:.1f} minutes")
    print(f"   Best val loss: {best_val_loss:.4f}")
    
    # Save history
    history_path = CONFIG['log_dir'] / f"foundation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"   History: {history_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training stopped by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
