#!/usr/bin/env python3
"""
Test GPU convolutions with relaxed MIOpen settings for gfx1030
"""

import os
import torch
import time
import sys

# Relaxed MIOpen settings - allow all algorithms
print("🔧 Applying relaxed MIOpen settings for gfx1030...")
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1030'
os.environ['MIOPEN_FIND_MODE'] = '3'  # Fast mode
os.environ['MIOPEN_ENABLE_LOGGING'] = '1'
os.environ['MIOPEN_ENABLE_LOGGING_CMD'] = '1'
# Don't disable algorithms - let MIOpen try everything
os.environ.pop('MIOPEN_DEBUG_CONV_GEMM', None)
os.environ.pop('MIOPEN_DEBUG_CONV_WINOGRAD', None)
os.environ.pop('MIOPEN_DEBUG_CONV_IMPLICIT_GEMM', None)
os.environ.pop('MIOPEN_DEBUG_FIND_ONLY_SOLVER', None)

print("=" * 70)
print("🧪 GPU CONVOLUTION TEST - RELAXED MODE")
print("=" * 70)

print(f"\n📋 Configuration:")
print(f"   PyTorch: {torch.__version__}")
print(f"   ROCm: {torch.version.hip}")
print(f"   Device: {torch.cuda.get_device_name(0)}")
print(f"   Architecture: {torch.cuda.get_device_properties(0).gcnArchName}")

def test_conv_simple():
    """Test simple Conv1d"""
    print(f"\n{'='*70}")
    print(f"🔬 TEST: Conv1d (4→8 channels, kernel=3)")
    print(f"{'='*70}")
    
    try:
        x = torch.randn(4, 4, 200).cuda()
        print(f"✅ Input tensor: {x.shape}")
        
        conv = torch.nn.Conv1d(4, 8, kernel_size=3, padding=1).cuda()
        print(f"✅ Conv layer on GPU")
        
        print(f"⏳ Running convolution (MIOpen will search for algorithms)...")
        start = time.time()
        with torch.no_grad():
            y = conv(x)
        elapsed = time.time() - start
        
        print(f"✅ SUCCESS! Output: {y.shape}, Time: {elapsed:.4f}s")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

success = test_conv_simple()

print("\n" + "="*70)
if success:
    print("🎉 GPU CONVOLUTION WORKING!")
else:
    print("❌ GPU CONVOLUTION FAILED - Trying CPU fallback")
    
sys.exit(0 if success else 1)
