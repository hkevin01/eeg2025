#!/usr/bin/env python3
"""
Conservative GPU Training with ROCm
- Maximum stability settings
- Gradual GPU usage
- Extensive error handling
"""

import os
import sys
from pathlib import Path

# Set ROCm environment variables BEFORE importing torch
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Synchronous ops for stability

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from tqdm import tqdm
import time
import json
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.models.eeg_dataset_production import ProductionEEGDataset

print("🚀 Conservative GPU Training with ROCm")
print("=" * 70)

# Ultra-conservative configuration
CONFIG = {
    'data_dir': Path(__file__).parent.parent / "data" / "raw" / "hbn",
    'output_dir': Path(__file__).parent.parent / "outputs",
    'checkpoint_dir': Path(__file__).parent.parent / "checkpoints",
    'log_dir': Path(__file__).parent.parent / "logs",
    
    # Data - START VERY SMALL
    'max_subjects': 2,  # Only 2 subjects to start
    'max_windows_per_subject': 100,  # Limit windows
    'train_split': 0.8,
    
    # Model - VERY SMALL
    'hidden_dim': 64,  # Much smaller
    'n_heads': 4,
    'n_layers': 2,
    'dropout': 0.1,
    
    # Training - VERY CONSERVATIVE
    'batch_size': 2,  # Tiny batches
    'epochs': 3,  # Just 3 epochs for testing
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    
    # Safety
    'gradient_clip': 1.0,
    'save_every': 1,
    'clear_cache_every': 10,  # Clear GPU cache frequently
}

class TinyTransformerEEG(nn.Module):
    """Tiny transformer for testing"""
    def __init__(self, n_channels=129, seq_len=1000, hidden_dim=64, 
                 n_heads=4, n_layers=2, n_classes=2, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(n_channels, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,  # Smaller feedforward
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

def setup_gpu():
    """Setup GPU with maximum safety"""
    print(f"\n{'='*70}")
    print("🔧 GPU Setup")
    print(f"{'='*70}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA/ROCm not available!")
        return None, {}
    
    device = torch.device('cuda:0')
    
    try:
        # Test GPU with tiny tensor
        print("Testing GPU with small tensor...")
        test_tensor = torch.randn(10, 10).to(device)
        result = test_tensor @ test_tensor.T
        print(f"✅ GPU test passed: {result.shape}")
        
        # Get GPU info
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_mem = props.total_memory / (1024**3)
        
        print(f"\n📊 GPU Information:")
        print(f"   Name: {gpu_name}")
        print(f"   Total Memory: {total_mem:.2f} GB")
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   ROCm/HIP: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")
        
        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.5, device=device)
        print(f"   Memory limit: 50% ({total_mem * 0.5:.2f} GB)")
        
        config = {
            'device': device,
            'gpu_name': gpu_name,
            'total_memory_gb': total_mem,
        }
        
        return device, config
        
    except Exception as e:
        print(f"❌ GPU initialization failed: {e}")
        print("Falling back to CPU")
        return torch.device('cpu'), {'device': torch.device('cpu')}

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train with extensive error handling"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    
    try:
        for batch_idx, (data, target) in enumerate(pbar):
            try:
                # Move to device
                data = data.to(device, non_blocking=False)
                target = target.to(device, non_blocking=False)
                
                # Forward pass
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip'])
                
                optimizer.step()
                
                # Stats
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Update progress
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.1f}%'
                })
                
                # Clear cache periodically
                if batch_idx % CONFIG['clear_cache_every'] == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n❌ OOM at batch {batch_idx}! Clearing cache...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        accuracy = 100. * correct / total if total > 0 else 0
        return avg_loss, accuracy
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

def validate(model, dataloader, criterion, device, epoch):
    """Validate with error handling"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    try:
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]  ")
            for data, target in pbar:
                data = data.to(device, non_blocking=False)
                target = target.to(device, non_blocking=False)
                
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.1f}%'
                })
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        accuracy = 100. * correct / total if total > 0 else 0
        return avg_loss, accuracy
        
    except Exception as e:
        print(f"\n❌ Validation error: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

def main():
    start_time = time.time()
    
    # Create directories
    CONFIG['output_dir'].mkdir(exist_ok=True)
    CONFIG['checkpoint_dir'].mkdir(exist_ok=True)
    CONFIG['log_dir'].mkdir(exist_ok=True)
    
    # Setup GPU
    device, gpu_config = setup_gpu()
    print(f"\n📱 Using device: {device}")
    
    # Load dataset
    print(f"\n{'='*70}")
    print("📂 Loading Dataset (SMALL for testing)")
    print(f"{'='*70}")
    
    try:
        dataset = ProductionEEGDataset(
            data_dir=CONFIG['data_dir'],
            max_subjects=CONFIG['max_subjects'],
            verbose=True
        )
        
        if len(dataset) == 0:
            print("❌ No data loaded!")
            return
        
        # Limit to max windows
        if len(dataset) > CONFIG['max_windows_per_subject'] * CONFIG['max_subjects']:
            indices = list(range(CONFIG['max_windows_per_subject'] * CONFIG['max_subjects']))
            dataset = Subset(dataset, indices)
            print(f"   Limited to {len(dataset)} windows for testing")
        
        # Split
        train_size = int(CONFIG['train_split'] * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"\n📊 Splits: Train={len(train_dataset)}, Val={len(val_dataset)}")
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=True,
            num_workers=0,  # No multiprocessing for stability
            pin_memory=(device.type == 'cuda')
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == 'cuda')
        )
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return
    
    # Create model
    print(f"\n{'='*70}")
    print("🧠 Creating Model")
    print(f"{'='*70}")
    
    try:
        # Get sample
        if isinstance(dataset, Subset):
            sample_data, _ = dataset.dataset[0]
        else:
            sample_data, _ = dataset[0]
        n_channels, seq_len = sample_data.shape
        
        model = TinyTransformerEEG(
            n_channels=n_channels,
            seq_len=seq_len,
            hidden_dim=CONFIG['hidden_dim'],
            n_heads=CONFIG['n_heads'],
            n_layers=CONFIG['n_layers'],
            dropout=CONFIG['dropout']
        ).to(device)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {n_params:,}")
        print(f"   Memory: ~{n_params * 4 / (1024**2):.1f} MB")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return
    
    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Training
    print(f"\n{'='*70}")
    print("🏋️  Training")
    print(f"{'='*70}")
    print(f"   Epochs: {CONFIG['epochs']}")
    print(f"   Batch size: {CONFIG['batch_size']}")
    print(f"   Device: {device}")
    print(f"{'='*70}\n")
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    try:
        for epoch in range(CONFIG['epochs']):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            
            # Validate
            val_loss, val_acc = validate(
                model, val_loader, criterion, device, epoch
            )
            
            epoch_time = time.time() - epoch_start
            
            # Record
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print
            print(f"\n📈 Epoch {epoch+1}/{CONFIG['epochs']}:")
            print(f"   Train: loss={train_loss:.4f}, acc={train_acc:.1f}%")
            print(f"   Val:   loss={val_loss:.4f}, acc={val_acc:.1f}%")
            print(f"   Time: {epoch_time:.1f}s")
            
            if device.type == 'cuda':
                mem_allocated = torch.cuda.memory_allocated() / (1024**3)
                mem_reserved = torch.cuda.memory_reserved() / (1024**3)
                print(f"   GPU: {mem_allocated:.2f}GB alloc, {mem_reserved:.2f}GB reserved")
            
            # Save
            if (epoch + 1) % CONFIG['save_every'] == 0 or val_loss < best_val_loss:
                checkpoint_path = CONFIG['checkpoint_dir'] / f"gpu_checkpoint_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"   💾 Saved: {checkpoint_path.name}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = CONFIG['checkpoint_dir'] / "gpu_best_model.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, best_path)
                    print(f"   ⭐ Best model updated!")
            
            # Clear cache after epoch
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            print()
        
        # Complete
        total_time = time.time() - start_time
        print(f"{'='*70}")
        print("✅ Training Complete!")
        print(f"{'='*70}")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Best val loss: {best_val_loss:.4f}")
        print(f"   Checkpoints: {CONFIG['checkpoint_dir']}")
        
        # Save history
        history_path = CONFIG['log_dir'] / f"gpu_training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"   History: {history_path}")
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ Training failed: {e}")
        print(f"{'='*70}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print("\n🧹 Cleanup complete")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
