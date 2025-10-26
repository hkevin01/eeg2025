"""
GPU Test 02: Convolution Operations
Tests Conv1d, Conv2d operations (CRITICAL for EEG models)

REQUIRED ENVIRONMENT VARIABLES FOR RDNA1 (gfx1030):
  export MIOPEN_DEBUG_CONV_GEMM=0
  export MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2=0
  export MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW53=0
  export HSA_OVERRIDE_GFX_VERSION=10.3.0
"""

import torch
import torch.nn as nn
import sys
import time
import os

def check_environment():
    """Check if ROCm convolution fixes are applied"""
    print("\n🔧 Checking ROCm Environment Variables:")
    required_vars = {
        "MIOPEN_DEBUG_CONV_GEMM": "0",
        "MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2": "0",
        "MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW53": "0",
        "HSA_OVERRIDE_GFX_VERSION": "10.3.0"
    }

    all_set = True
    for var, expected in required_vars.items():
        value = os.environ.get(var, "NOT SET")
        status = "✅" if value == expected else "⚠️"
        print(f"   {status} {var}={value}")
        if value != expected:
            all_set = False

    if not all_set:
        print("\n⚠️  WARNING: Not all ROCm fixes are set!")
        print("   Convolutions may crash on RDNA1 GPUs (gfx1030)")
        print("   Run: source ~/.bashrc  (if fixes are in bashrc)")
    else:
        print("\n✅ All ROCm convolution fixes are active!")

    print()
    return all_set

def test_convolutions():
    """Test convolution operations"""
    print("="*70)
    print("TEST 02: Convolution Operations (CRITICAL)")
    print("="*70)

    check_environment()

    if not torch.cuda.is_available():
        print("❌ GPU not available")
        return False

    device = torch.device("cuda")

    # Test 1: Conv1d (used in EEG models)
    print("\n1️⃣ Conv1d (EEG-style):")
    try:
        conv1d = nn.Conv1d(129, 64, kernel_size=7).to(device)
        x = torch.randn(4, 129, 200, device=device)  # Batch, Channels, Time
        print(f"   Input: {x.shape}")

        start = time.time()
        out = conv1d(x)
        elapsed = time.time() - start

        print(f"   Output: {out.shape}")
        print(f"   Time: {elapsed*1000:.2f} ms")
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

    # Test 2: Multi-layer Conv1d
    print("\n2️⃣ Multi-Layer Conv1d:")
    try:
        model = nn.Sequential(
            nn.Conv1d(129, 64, 7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, 5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        ).to(device)

        x = torch.randn(8, 129, 200, device=device)
        start = time.time()
        out = model(x)
        elapsed = time.time() - start

        print(f"   Output: {out.shape}")
        print(f"   Time: {elapsed*1000:.2f} ms")
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

    # Test 3: Conv2d
    print("\n3️⃣ Conv2d:")
    try:
        conv2d = nn.Conv2d(3, 64, kernel_size=3).to(device)
        x = torch.randn(4, 3, 32, 32, device=device)
        out = conv2d(x)
        print(f"   Input: {x.shape} → Output: {out.shape}")
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

    # Test 4: Large batch conv1d (stress test)
    print("\n4️⃣ Large Batch Conv1d (Stress Test - Batch 32):")
    try:
        conv = nn.Conv1d(129, 128, 7).to(device)
        x = torch.randn(32, 129, 200, device=device)

        start = time.time()
        out = conv(x)
        elapsed = time.time() - start

        print(f"   Batch: 32, Time: {elapsed*1000:.2f} ms")
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

    # Test 4b: VERY Large batch conv1d (extreme stress test)
    print("\n4️⃣b Very Large Batch Conv1d (EXTREME - Batch 64):")
    try:
        conv = nn.Conv1d(129, 128, 7).to(device)
        x = torch.randn(64, 129, 200, device=device)

        start = time.time()
        out = conv(x)
        elapsed = time.time() - start

        print(f"   Batch: 64, Time: {elapsed*1000:.2f} ms")
        print("   ✅ PASSED (LARGE BATCH TEST)")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

    # Test 4c: Conv2d Large Batch (5x5 kernel)
    print("\n4️⃣c Conv2d Large Batch (5x5 kernel - Batch 64):")
    try:
        conv = nn.Conv2d(256, 256, kernel_size=5, padding=2).to(device)
        x = torch.randn(64, 256, 32, 32, device=device)

        start = time.time()
        out = conv(x)
        elapsed = time.time() - start

        print(f"   Batch: 64, Shape: {out.shape}, Time: {elapsed*1000:.2f} ms")
        print("   ✅ PASSED (CONV2D LARGE BATCH)")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

    # Test 5: Depthwise separable convolution
    print("\n5️⃣ Depthwise Separable Conv:")
    try:
        dw_conv = nn.Conv1d(64, 64, 7, groups=64).to(device)
        pw_conv = nn.Conv1d(64, 128, 1).to(device)

        x = torch.randn(8, 64, 100, device=device)
        out = pw_conv(dw_conv(x))
        print(f"   Output: {out.shape}")
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

    print("\n" + "="*70)
    print("✅ ALL CONVOLUTION TESTS PASSED")
    print("="*70)
    return True

if __name__ == "__main__":
    success = test_convolutions()
    sys.exit(0 if success else 1)
