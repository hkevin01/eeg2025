#!/usr/bin/env python3
"""
Test GPU convolutions with IMMEDIATE mode (no database lookups)
"""

import os
import torch
import time
import sys

# Use IMMEDIATE mode - no database, compile kernels on-the-fly
print("🔧 Setting MIOpen to IMMEDIATE mode (no database)...")
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1030'
os.environ['MIOPEN_FIND_MODE'] = '2'  # Immediate mode - no DB lookup
os.environ['MIOPEN_DEBUG_DISABLE_FIND_DB'] = '1'  # Disable find database
os.environ['MIOPEN_DISABLE_CACHE'] = '1'  # Disable kernel cache

print("=" * 70)
print("🧪 GPU CONVOLUTION TEST - IMMEDIATE MODE")
print("=" * 70)

print(f"\n📋 Configuration:")
print(f"   PyTorch: {torch.__version__}")
print(f"   ROCm: {torch.version.hip}")
print(f"   Device: {torch.cuda.get_device_name(0)}")

def test_conv():
    print(f"\n{'='*70}")
    print(f"🔬 Testing Conv1d with immediate kernel compilation...")
    print(f"{'='*70}")
    
    try:
        print("⏳ Creating tensors...")
        x = torch.randn(2, 4, 100).cuda()  # Smaller for faster test
        print(f"✅ Input: {x.shape}")
        
        conv = torch.nn.Conv1d(4, 8, kernel_size=3, padding=1).cuda()
        print(f"✅ Conv layer ready")
        
        print(f"⏳ Running convolution (may take time for first compile)...")
        start = time.time()
        with torch.no_grad():
            y = conv(x)
        elapsed = time.time() - start
        
        print(f"✅ SUCCESS! Output: {y.shape}, Time: {elapsed:.3f}s")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

success = test_conv()

if not success:
    print("\n⚠️  GPU convolutions not working - using CPU fallback")
    print("    This is expected for gfx1030 with current MIOpen limitations")

sys.exit(0 if success else 1)
