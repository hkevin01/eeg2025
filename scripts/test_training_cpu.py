#!/usr/bin/env python3
"""Minimal CPU training test"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from train_gpu_safe
from train_gpu_safe import SimpleEEGDataset, SimpleTransformer

print("🧪 CPU Training Test")
print("="*60)

# Force CPU
device = torch.device("cpu")
print(f"Device: {device}")

# Load small dataset
data_dir = Path(__file__).parent.parent / "data" / "raw" / "hbn"
print(f"\n📂 Loading 1 subject...")
dataset = SimpleEEGDataset(data_dir, max_subjects=1)

if len(dataset) == 0:
    print("❌ No data!")
    exit(1)

print(f"✅ Loaded {len(dataset)} windows")

# Create dataloader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

# Create model
sample_data, _ = dataset[0]
n_channels, seq_len = sample_data.shape
print(f"\n🧠 Creating model for input shape: ({n_channels}, {seq_len})")

model = SimpleTransformer(
    n_channels=n_channels,
    seq_len=seq_len,
    hidden_dim=64,  # Very small
    n_heads=2,
    n_layers=1,
    n_classes=2
).to(device)

print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
print("\n🔬 Testing forward pass...")
data, target = next(iter(dataloader))
data = data.to(device)
print(f"   Input shape: {data.shape}")

output = model(data)
print(f"   Output shape: {output.shape}")
print(f"✅ Forward pass successful!")

# Test backward pass
print("\n🔬 Testing backward pass...")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss = criterion(output, target)
print(f"   Loss: {loss.item():.4f}")

loss.backward()
optimizer.step()
print(f"✅ Backward pass successful!")

# Mini training loop (3 batches)
print("\n🏋️  Mini training loop (3 batches)...")
model.train()
for i, (data, target) in enumerate(dataloader):
    if i >= 3:
        break
    
    optimizer.zero_grad()
    output = model(data.to(device))
    loss = criterion(output, target.to(device))
    loss.backward()
    optimizer.step()
    
    print(f"   Batch {i+1}: loss={loss.item():.4f}")

print("\n✅ All tests passed! Data pipeline works correctly.")
print("="*60)
