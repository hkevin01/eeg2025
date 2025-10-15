#!/usr/bin/env python3
"""Minimal CPU-only test - no GPU at all"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HIP_VISIBLE_DEVICES'] = ''

import torch
import torch.nn as nn

print("="*60)
print("🛡️  MINIMAL CPU TEST")
print("="*60)
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: CPU only")

# Simple model
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# Test forward pass
x = torch.randn(4, 10)
output = model(x)

print(f"✅ Input shape: {x.shape}")
print(f"✅ Output shape: {output.shape}")
print(f"✅ Output values: {output.squeeze().tolist()}")

# Test training step
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

target = torch.randn(4, 1)
loss = criterion(output, target)
loss.backward()
optimizer.step()

print(f"✅ Loss: {loss.item():.4f}")
print("="*60)
print("🎉 CPU-ONLY TEST PASSED - NO CRASHES!")
print("="*60)
