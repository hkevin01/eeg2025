#!/usr/bin/env python3
"""
Test GPU convolutions with MIOpen workarounds for gfx1030
Based on GitHub pytorch/pytorch#5195 community solutions
"""

import os
import torch
import time
import sys

# Apply MIOpen workarounds for gfx1030 (AMD RX 5600 XT)
print("🔧 Applying MIOpen workarounds for gfx1030...")
os.environ['MIOPEN_DEBUG_CONV_GEMM'] = '0'
os.environ['MIOPEN_DEBUG_CONV_WINOGRAD'] = '0'
os.environ['MIOPEN_DEBUG_CONV_IMPLICIT_GEMM'] = '0'
os.environ['MIOPEN_DEBUG_CONV_FFT'] = '0'
os.environ['MIOPEN_DEBUG_CONV_DIRECT'] = '0'
os.environ['MIOPEN_FIND_MODE'] = '1'
os.environ['MIOPEN_DEBUG_FIND_ONLY_SOLVER'] = 'ConvBiasActivAsm1x1U'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1030'

print("=" * 70)
print("🧪 GPU CONVOLUTION TEST WITH MIOPEN WORKAROUNDS")
print("=" * 70)

# Must import torch AFTER setting environment variables
print(f"\n📋 Configuration:")
print(f"   PyTorch: {torch.__version__}")
print(f"   ROCm: {torch.version.hip}")
print(f"   Device: {torch.cuda.get_device_name(0)}")
print(f"   Architecture: {torch.cuda.get_device_properties(0).gcnArchName}")

def test_conv_simple():
    """Test simple Conv1d - this was hanging"""
    print(f"\n{'='*70}")
    print(f"🔬 TEST 1: Simple Conv1d (4→8 channels, kernel=3)")
    print(f"{'='*70}")
    
    try:
        x = torch.randn(4, 4, 200).cuda()
        print(f"✅ Input tensor created: {x.shape}")
        
        conv = torch.nn.Conv1d(4, 8, kernel_size=3, padding=1).cuda()
        print(f"✅ Conv layer on GPU")
        
        print(f"⏳ Running convolution...")
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

# Run test
success = test_conv_simple()

print("\n" + "="*70)
if success:
    print("🎉 GPU CONVOLUTION TEST PASSED!")
    print("="*70)
else:
    print("❌ GPU CONVOLUTION TEST FAILED")
    print("="*70)
    sys.exit(1)
