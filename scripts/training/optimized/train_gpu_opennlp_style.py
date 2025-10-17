#!/usr/bin/env python3
"""
GPU Training with OpenNLP-GPU patterns
- Proper environment variable setup
- Graceful fallback to CPU
- Tested small operations before full training
"""

import os
import sys
from pathlib import Path

# Set ROCm environment variables FIRST (from opennlp-gpu)
os.environ['ROCM_PATH'] = '/opt/rocm'
os.environ['HIP_PATH'] = '/opt/rocm'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['HIP_PLATFORM'] = 'amd'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:128'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import time
import json
from datetime import datetime
import warnings

# Suppress hipBLASLt warning
warnings.filterwarnings('ignore', message='.*hipBLASLt.*')

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.models.eeg_dataset_simple import SimpleEEGDataset

print("🚀 GPU Training (OpenNLP-GPU Style)")
print("=" * 70)

# Configuration
CONFIG = {
    'data_dir': Path(__file__).parent.parent / "data" / "raw" / "hbn",
    'output_dir': Path(__file__).parent.parent / "outputs",
    'checkpoint_dir': Path(__file__).parent.parent / "checkpoints",
    'log_dir': Path(__file__).parent.parent / "logs",
    
    # Data
    'max_subjects': None,
    'train_split': 0.8,
    
    # Model - Start small for GPU testing
    'hidden_dim': 128,
    'n_heads': 8,
    'n_layers': 4,
    'dropout': 0.1,
    
    # Training
    'batch_size': 16,
    'epochs': 20,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'gradient_clip': 1.0,
    'save_every': 2,
}

# Create directories
for dir_path in [CONFIG['output_dir'], CONFIG['checkpoint_dir'], CONFIG['log_dir']]:
    dir_path.mkdir(exist_ok=True)

# Model
class FoundationTransformer(nn.Module):
    def __init__(self, n_channels=129, seq_len=1000, 
                 hidden_dim=128, n_heads=8, n_layers=4, 
                 n_classes=2, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(n_channels, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
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
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

def test_gpu_with_model(device):
    """Test GPU with actual model (OpenNLP-GPU pattern)"""
    print("\n🧪 Testing GPU with model...")
    
    try:
        # Create small test model
        test_model = FoundationTransformer(
            n_channels=129,
            seq_len=1000,
            hidden_dim=64,  # Smaller for test
            n_heads=4,
            n_layers=2,
            dropout=0.1
        ).to(device)
        
        # Test forward pass
        test_input = torch.randn(2, 129, 1000).to(device)
        test_output = test_model(test_input)
        
        print(f"✅ Model test passed: {test_output.shape}")
        
        # Cleanup
        del test_model, test_input, test_output
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def setup_device():
    """Setup device with OpenNLP-GPU patterns"""
    print("\n🔧 Device Setup")
    print("-" * 70)
    
    if torch.cuda.is_available():
        print("✅ GPU detected by PyTorch")
        print(f"   Name: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        
        device = torch.device('cuda:0')
        
        # Test with model (OpenNLP-GPU approach)
        if test_gpu_with_model(device):
            print("✅ GPU ready for training")
            return device
        else:
            print("⚠️  GPU test failed, falling back to CPU")
            return torch.device('cpu')
    else:
        print("⭕ No GPU detected, using CPU")
        return torch.device('cpu')

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    print(f"\n📚 Epoch {epoch+1} - Training")
    batch_count = 0
    for batch_idx, (data, target) in enumerate(loader):
        try:
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
                print(f"   [{batch_idx+1}/{len(loader)}] loss={loss.item():.4f}, acc={100.*correct/total:.1f}%")
            
            batch_count += 1
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n⚠️  OOM at batch {batch_idx}, skipping...")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                continue
            else:
                raise
    
    if batch_count == 0:
        return 0.0, 0.0
    
    return total_loss / batch_count, 100. * correct / total if total > 0 else 0

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

def main():
    start_time = time.time()
    
    # Setup device
    device = setup_device()
    print(f"\n📱 Training on: {device}")
    print("=" * 70)
    
    # Load dataset
    print("\n📂 Loading Dataset...")
    dataset = SimpleEEGDataset(data_dir=CONFIG['data_dir'], max_subjects=CONFIG['max_subjects'])
    print(f"   Total: {len(dataset)} windows")
    
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
    
    # Model
    print("\n🧠 Creating Model...")
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
    
    # Training
    print("\n" + "=" * 70)
    print("🏋️  Training Started")
    print("=" * 70)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(CONFIG['epochs']):
        epoch_start = time.time()
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"{'='*70}")
        
        # Train & validate
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        
        epoch_time = time.time() - epoch_start
        
        # Record
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train: loss={train_loss:.4f}, acc={train_acc:.1f}%")
        print(f"   Val:   loss={val_loss:.4f}, acc={val_acc:.1f}%")
        print(f"   Time:  {epoch_time:.1f}s")
        
        if device.type == 'cuda':
            mem_allocated = torch.cuda.memory_allocated() / (1024**3)
            mem_reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"   GPU Memory: {mem_allocated:.2f}GB alloc, {mem_reserved:.2f}GB reserved")
        
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
            
            path = CONFIG['checkpoint_dir'] / f"gpu_opennlp_epoch_{epoch+1}.pth"
            torch.save(checkpoint, path)
            print(f"   💾 Checkpoint: {path.name}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = CONFIG['checkpoint_dir'] / "gpu_opennlp_best.pth"
                torch.save(checkpoint, best_path)
                print(f"   ⭐ New best model!")
        
        # Clear cache after epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Finish
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("✅ Training Complete!")
    print(f"{'='*70}")
    print(f"   Device: {device}")
    print(f"   Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Checkpoints: {CONFIG['checkpoint_dir']}")
    
    # Save history
    history_path = CONFIG['log_dir'] / f"gpu_opennlp_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"   History: {history_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
