#!/usr/bin/env python3
"""
Test GPU training capabilities with proper EEG dimensions
"""
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import time

print("=" * 80)
print("GPU Training Test (EEG Model)")
print("=" * 80)

# Check device
print("\n1️⃣ Device Check:")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device Name: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    device = torch.device('cuda')
else:
    print("   No GPU - using CPU")
    device = torch.device('cpu')

# Create EEG model (1D convolutions)
print("\n2️⃣ Creating EEG Model:")
class EEGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: (batch, channels, time)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

model = EEGModel().to(device)
print(f"   Model device: {next(model.parameters()).device}")
print(f"   Model architecture: 128ch -> 64ch -> 32ch -> 1 output")

# Create dummy EEG data: (batch, channels, time_samples)
print("\n3️⃣ Creating Dummy EEG Data:")
batch_size = 32
channels = 128
time_samples = 200

data = torch.randn(batch_size, channels, time_samples, device=device)
target = torch.randn(batch_size, 1, device=device)
print(f"   Data shape: {data.shape} (batch, channels, time)")
print(f"   Data device: {data.device}")

# Test forward pass
print("\n4️⃣ Testing Forward Pass:")
with torch.no_grad():
    output = model(data)
    print(f"   ✅ Forward pass successful")
    print(f"   Output shape: {output.shape}")
    print(f"   Output device: {output.device}")

# Test training step with AMP
print("\n5️⃣ Testing Training Step (with AMP):")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
scaler = GradScaler(device.type)

model.train()
start = time.time()

for i in range(10):
    optimizer.zero_grad()
    
    with autocast(device.type):
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    if i == 0:
        print(f"   Iteration 0 - Loss: {loss.item():.4f}")
    elif i == 9:
        print(f"   Iteration 9 - Loss: {loss.item():.4f}")

elapsed = time.time() - start
print(f"   ✅ Training successful")
print(f"   Time for 10 iterations: {elapsed:.3f}s")
print(f"   Iterations/sec: {10/elapsed:.2f}")

# Test GPU memory
if torch.cuda.is_available():
    print("\n6️⃣ GPU Memory Check:")
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    print(f"   Allocated: {allocated:.3f} GB")
    print(f"   Reserved: {reserved:.3f} GB")

# Intensive test to trigger GPU activity
print("\n7️⃣ Intensive GPU Test (500 iterations):")
print("   Run 'watch -n 1 rocm-smi' in another terminal to see GPU activity")
print("   Starting in 3 seconds...")
time.sleep(3)

start = time.time()
for i in range(500):
    optimizer.zero_grad()
    with autocast(device.type):
        output = model(data)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    if (i+1) % 100 == 0:
        print(f"   Progress: {i+1}/500 iterations")
        
elapsed = time.time() - start
print(f"   ✅ Completed 500 iterations in {elapsed:.3f}s")
print(f"   Throughput: {500/elapsed:.2f} iter/s")

if torch.cuda.is_available():
    print("\n8️⃣ Final GPU Memory:")
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    print(f"   Allocated: {allocated:.3f} GB")
    print(f"   Reserved: {reserved:.3f} GB")

print("\n" + "=" * 80)
print("✅ GPU Training Test Complete!")
print("=" * 80)
print("\nResults:")
if device.type == 'cuda':
    print("  ✅ GPU is working and will be used for training")
    print("  ✅ Mixed precision (AMP) is working")
    print(f"  ✅ Training speed: {500/elapsed:.2f} iterations/second")
else:
    print("  ⚠️  Using CPU - GPU not available")
