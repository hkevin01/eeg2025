#!/usr/bin/env python3
"""
GPU Training with PyTorch 1.13.1+rocm5.2 - Working Configuration
Uses smaller models that fit within GPU memory/convolution limits
"""

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print("🚀 GPU Training with PyTorch 1.13.1+rocm5.2")
print("=" * 50)
print(f"PyTorch: {torch.__version__}")
print(f"GPU Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    
    # Create a smaller EEG model that works within constraints
    class SmallEEGModel(nn.Module):
        """Reduced-size EEG model that works with gfx1030 limitations"""
        def __init__(self, input_channels=32, sequence_length=100):
            super().__init__()
            # Much smaller than full EEG (129 channels, 200 timesteps)
            self.conv1 = nn.Conv1d(input_channels, 16, 3, padding=1)
            self.conv2 = nn.Conv1d(16, 8, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(8, 1)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = x.flatten(1)
            x = self.fc(x)
            return x
    
    try:
        print("🔧 Creating small EEG model...")
        model = SmallEEGModel().cuda()
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        print("\n🔬 Testing forward pass...")
        batch_size = 2  # Small batch
        input_channels = 32  # Reduced from 129
        sequence_length = 100  # Reduced from 200
        
        # Simulate EEG data
        data = torch.randn(batch_size, input_channels, sequence_length).cuda()
        targets = torch.randn(batch_size, 1).cuda()
        
        outputs = model(data)
        print(f"✅ Forward pass: {data.shape} -> {outputs.shape}")
        
        print("\n🏋️ Testing training loop...")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(5):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            memory_used = torch.cuda.memory_allocated() / 1024**2
            print(f"  Epoch {epoch+1}: loss = {loss.item():.4f}, memory = {memory_used:.1f} MB")
        
        print("\n🎉 SUCCESS! GPU training works with constraints:")
        print(f"   - Input channels: {input_channels} (vs 129 in full EEG)")
        print(f"   - Sequence length: {sequence_length} (vs 200 in full EEG)")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
        
        print("\n📊 Scaling Strategy:")
        print("   1. Use this setup for rapid prototyping and experimentation")
        print("   2. Train multiple smaller models and ensemble them")
        print("   3. Use techniques like:")
        print("      - Channel reduction (PCA/ICA preprocessing)")
        print("      - Temporal downsampling")  
        print("      - Model distillation from CPU-trained larger models")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        
else:
    print("❌ GPU not available")
