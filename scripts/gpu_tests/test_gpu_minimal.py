#!/usr/bin/env python3
"""Minimal GPU test with ROCm fixes"""

import os
import sys

# Critical ROCm environment variables for Navi 10 stability
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['HSA_ENABLE_SDMA'] = '0'
os.environ['GPU_MAX_HW_QUEUES'] = '4'

print("🧪 Minimal GPU Test with ROCm Fixes")
print("="*60)

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("❌ No GPU available")
    sys.exit(1)

print(f"GPU device: {torch.cuda.get_device_name(0)}")
print(f"Device count: {torch.cuda.device_count()}")

print("\n�� Creating small tensor on GPU...")
try:
    x = torch.randn(10, 10).cuda()
    print(f"✅ Tensor created: {x.shape} on {x.device}")
    print(f"   Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

print("\n📊 Matrix multiplication test...")
try:
    y = x @ x.T
    print(f"✅ Result: {y.shape}")
    print(f"   Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

print("\n✅ All tests passed!")
