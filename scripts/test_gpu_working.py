#!/usr/bin/env python3
"""
Test GPU training with HSA_OVERRIDE_GFX_VERSION=10.3.0 workaround
Based on GitHub issue: https://github.com/RadeonOpenCompute/ROCm/issues/2527
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

# Set the working environment variable
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

print("🧪 Testing GPU Training with gfx1030 Workaround")
print("=" * 50)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'Not set')}")

if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    try:
        print("🔬 Test 1: Basic operations")
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = x @ y
        print(f"✅ Matrix multiplication: {x.shape} @ {y.shape} = {z.shape}")
        
        print("\n🔬 Test 2: Small EEG-like model")
        # Create a smaller model that should work
        model = nn.Sequential(
            nn.Conv1d(32, 16, 3, padding=1),  # Reduced from 129 channels
            nn.ReLU(),
            nn.Conv1d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(8, 1)
        ).cuda()
        
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        print("\n🔬 Test 3: Small batch training")
        batch_size = 2  # Very small batch
        channels = 32   # Reduced channels
        time_points = 100  # Reduced time points
        
        # Simulate training
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(3):
            data = torch.randn(batch_size, channels, time_points).cuda()
            targets = torch.randn(batch_size, 1).cuda()
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            memory_used = torch.cuda.memory_allocated() / 1024**2
            print(f"  Epoch {epoch+1}: loss = {loss.item():.4f}, memory = {memory_used:.1f} MB")
        
        print("\n🎉 SUCCESS: GPU training works with gfx1030!")
        print(f"✅ Working configuration:")
        print(f"   - HSA_OVERRIDE_GFX_VERSION=10.3.0")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Channels: {channels}")
        print(f"   - Time points: {time_points}")
        print(f"   - Peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        print("The workaround may need further tuning...")

else:
    print("❌ CUDA not available")
