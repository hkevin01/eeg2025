#!/usr/bin/env python3
"""Test GPU-Compatible EEGNeX on AMD gfx1030"""

import sys
sys.path.insert(0, 'src')

import torch
from models.eegnex_gpu_fix import EEGNeXGPUFix

print("=" * 60)
print("Testing GPU-Compatible EEGNeX on AMD RX 5600 XT (gfx1030)")
print("=" * 60)

# Create model
model = EEGNeXGPUFix(
    n_chans=129,
    n_outputs=1,
    n_times=200,
    sfreq=100,
)

print(f"\n✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters\n")

# Test CPU
print("Testing CPU forward pass...")
x_cpu = torch.randn(2, 129, 200)
y_cpu = model(x_cpu)
print(f"✅ CPU works: {x_cpu.shape} -> {y_cpu.shape}\n")

# Test GPU if available
if not torch.cuda.is_available():
    print("⚠️  CUDA not available")
    sys.exit(0)

print(f"GPU Device: {torch.cuda.get_device_name(0)}")
print("Moving model to GPU...")
model_gpu = model.cuda()
print("✅ Model on GPU\n")

print("Testing GPU forward pass (batch=2)...")
x_gpu = torch.randn(2, 129, 200, device='cuda')
print("  Input tensor created on GPU")

try:
    print("  Running forward pass...")
    y_gpu = model_gpu(x_gpu)
    print(f"✅ GPU works: {x_gpu.shape} -> {y_gpu.shape}")
    print("\n🎉 SUCCESS! GPU-compatible EEGNeX works on AMD gfx1030!")
except Exception as e:
    print(f"❌ GPU forward pass failed: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)
