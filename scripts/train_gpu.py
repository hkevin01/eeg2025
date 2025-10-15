#!/usr/bin/env python3
"""
GPU-Enabled Training Script for EEG Foundation Model
Supports: NVIDIA CUDA, AMD ROCm, Apple MPS, CPU fallback
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.simple_eeg_model import SimpleEEGTransformer
from scripts.eeg_dataset import EEGDataset


def detect_device():
    """Detect best available compute device."""
    print("\n" + "="*60)
    print("🔍 Detecting compute devices...")
    print("="*60)
    
    # Try CUDA (NVIDIA or ROCm-translated)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ CUDA available")
        print(f"   Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"      Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"      Compute: {props.major}.{props.minor}")
        
        # Check if this is ROCm
        if hasattr(torch.version, 'hip') and torch.version.hip:
            print(f"   Backend: ROCm (HIP {torch.version.hip})")
        else:
            print(f"   Backend: CUDA {torch.version.cuda}")
        return device
    
    # Try MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Apple MPS available")
        return device
    
    # Fallback to CPU
    device = torch.device("cpu")
    print(f"⚠️  No GPU detected, using CPU")
    print(f"   CPU cores: {os.cpu_count()}")
    print(f"   Recommendation: Install PyTorch with GPU support for better performance")
    return device


def setup_gpu_optimizations(device):
    """Set up GPU-specific optimizations."""
    if device.type == "cuda":
        # Enable TF32 for Ampere+ GPUs (faster on NVIDIA)
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("   ✅ TF32 enabled (Ampere+ GPU)")
        
        # Enable cuDNN benchmark mode
        torch.backends.cudnn.benchmark = True
        print("   ✅ cuDNN benchmark enabled")
        
        # Enable memory efficient attention if available
        try:
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                print("   ✅ Flash Attention available")
        except:
            pass


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler=None):
    """Train for one epoch with GPU support."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    use_amp = device.type == "cuda" and scaler is not None
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%', end='\r')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate model with GPU support."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def benchmark_speed(model, device, batch_size=32, n_iters=50):
    """Benchmark inference speed on GPU."""
    print("\n" + "="*60)
    print("🔥 Benchmarking inference speed...")
    print("="*60)
    
    model.eval()
    # Create dummy data
    dummy_input = torch.randn(batch_size, 129, 1000).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            start = time.time()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - start)
    
    times = np.array(times)
    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    p95_time = np.percentile(times, 95) * 1000
    
    print(f"Batch size: {batch_size}")
    print(f"Average latency: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"P95 latency: {p95_time:.2f} ms")
    print(f"Per-sample latency: {avg_time/batch_size:.2f} ms")
    print(f"Throughput: {1000*batch_size/avg_time:.1f} samples/sec")
    
    if device.type == "cuda":
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    
    return avg_time / batch_size  # Per-sample latency


def main():
    parser = argparse.ArgumentParser(description='GPU-Enabled EEG Training')
    parser.add_argument('--data-dir', type=str, default='data/raw/hbn',
                        help='Path to data directory')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run speed benchmark only')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu', 'mps'],
                        help='Device to use for training')
    args = parser.parse_args()
    
    # Detect device
    if args.device == 'auto':
        device = detect_device()
    else:
        device = torch.device(args.device)
        print(f"Using specified device: {device}")
    
    # Setup optimizations
    setup_gpu_optimizations(device)
    
    print("\n" + "="*60)
    print("📊 Loading dataset...")
    print("="*60)
    
    # Load dataset
    dataset = EEGDataset(args.data_dir)
    print(f"Total samples: {len(dataset)}")
    print(f"Channels: {dataset[0][0].shape[0]}")
    print(f"Time points: {dataset[0][0].shape[1]}")
    print(f"Classes: {len(set([y for _, y in dataset]))}")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    num_workers = 4 if device.type == "cpu" else 2
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\n" + "="*60)
    print("🧠 Creating model...")
    print("="*60)
    
    num_classes = len(set([y for _, y in dataset]))
    model = SimpleEEGTransformer(
        n_channels=129,
        n_timepoints=1000,
        n_classes=num_classes,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Model size: {n_params * 4 / 1e6:.2f} MB (FP32)")
    
    # Benchmark if requested
    if args.benchmark:
        benchmark_speed(model, device, batch_size=args.batch_size)
        return
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and not args.no_amp) else None
    if scaler:
        print("✅ Mixed precision training enabled (FP16)")
    
    # Training loop
    print("\n" + "="*60)
    print("🚀 Starting training...")
    print("="*60)
    
    best_val_acc = 0
    os.makedirs('outputs/checkpoints', exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)
        
        # Train
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                           optimizer, device, epoch, scaler)
        train_time = time.time() - start_time
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        print(f'\n  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        print(f'  Time: {train_time:.1f}s, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if device.type == "cuda":
            print(f'  GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, 'outputs/checkpoints/best_model_gpu.pth')
            print(f'  ✅ Saved best model (val_acc: {val_acc:.2f}%)')
    
    print("\n" + "="*60)
    print("✅ Training completed!")
    print("="*60)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Final benchmark
    print("\n📈 Final performance benchmark:")
    per_sample_latency = benchmark_speed(model, device, batch_size=32)
    
    if per_sample_latency < 50:
        print(f"\n✅ SUCCESS! Latency {per_sample_latency:.2f}ms < 50ms target")
    else:
        print(f"\n⚠️ Latency {per_sample_latency:.2f}ms > 50ms target (needs optimization)")


if __name__ == '__main__':
    main()
