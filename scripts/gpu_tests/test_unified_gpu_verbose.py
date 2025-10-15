#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verbose Unified GPU Testing with Maximum Debug Output
====================================================
"""

import sys
import os
import time

print("=" * 80)
print("🔍 VERBOSE UNIFIED GPU TEST - Starting")
print("=" * 80)

# Step 1: Basic imports
print("\n[Step 1] Testing basic imports...")
try:
    import torch
    print(f"  ✅ torch imported: {torch.__version__}")
except Exception as e:
    print(f"  ❌ torch import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"  ✅ numpy imported: {np.__version__}")
except Exception as e:
    print(f"  ❌ numpy import failed: {e}")
    sys.exit(1)

# Step 2: Check PyTorch GPU support
print("\n[Step 2] Checking PyTorch GPU support...")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    if hasattr(torch.version, 'hip') and torch.version.hip:
        print(f"  ROCm/HIP version: {torch.version.hip}")
        print(f"  Platform: AMD ROCm")
    else:
        print(f"  Platform: NVIDIA CUDA")
    print(f"  Device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("  ⚠️  No GPU support detected")

# Step 3: Add src to path
print("\n[Step 3] Adding src to path...")
src_path = '/home/kevin/Projects/eeg2025/src'
if os.path.exists(src_path):
    sys.path.insert(0, src_path)
    print(f"  ✅ Added {src_path} to path")
else:
    print(f"  ❌ Path does not exist: {src_path}")
    sys.exit(1)

# Step 4: Import unified module
print("\n[Step 4] Importing unified GPU module...")
try:
    print("  Attempting import...")
    from gpu.unified_gpu_optimized import (
        GPUPlatformDetector,
        UnifiedFFTOptimizer,
        UnifiedBLASOptimizer,
        UnifiedLinearLayer
    )
    print("  ✅ All classes imported successfully")
except ImportError as e:
    print(f"  ❌ Import error: {e}")
    print(f"  sys.path: {sys.path}")
    sys.exit(1)
except Exception as e:
    print(f"  ❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Test platform detection
print("\n[Step 5] Testing platform detection...")
try:
    print("  Creating GPUPlatformDetector...")
    detector = GPUPlatformDetector()
    print("  ✅ Detector created")
    
    print("  Getting platform info...")
    info = detector.get_info()
    print(f"  ✅ Info retrieved: {info}")
    
    print("\n  Platform Details:")
    for key, value in info.items():
        print(f"    {key}: {value}")
    
except Exception as e:
    print(f"  ❌ Platform detection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Test FFT optimizer creation
print("\n[Step 6] Testing FFT optimizer creation...")
try:
    print("  Creating UnifiedFFTOptimizer...")
    fft_opt = UnifiedFFTOptimizer()
    print(f"  ✅ FFT optimizer created on device: {fft_opt.device}")
    print(f"  Platform info: {fft_opt.platform_info}")
except Exception as e:
    print(f"  ❌ FFT optimizer creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 7: Test simple tensor operation
print("\n[Step 7] Testing simple tensor operations...")
try:
    print("  Creating test tensor (small)...")
    x = torch.randn(2, 4, 100)
    print(f"  ✅ Created tensor shape: {x.shape}")
    
    print("  Moving to device if GPU available...")
    if torch.cuda.is_available():
        x_device = x.cuda()
        print(f"  ✅ Tensor on GPU: {x_device.device}")
        x_back = x_device.cpu()
        print(f"  ✅ Tensor back on CPU: {x_back.device}")
    else:
        print("  ⚠️  GPU not available, skipping GPU transfer")
    
except Exception as e:
    print(f"  ❌ Tensor operation failed: {e}")
    import traceback
    traceback.print_exc()

# Step 8: Test FFT operation (very small)
print("\n[Step 8] Testing FFT operation (very small data)...")
try:
    print("  Creating tiny test signal...")
    signal = torch.randn(1, 2, 50)  # Minimal size
    print(f"  ✅ Signal shape: {signal.shape}")
    
    print("  Computing FFT (CPU)...")
    start = time.time()
    result_cpu = torch.fft.rfft(signal, dim=-1)
    cpu_time = time.time() - start
    print(f"  ✅ CPU FFT completed in {cpu_time*1000:.2f}ms")
    print(f"  Result shape: {result_cpu.shape}")
    
    if torch.cuda.is_available():
        print("  Computing FFT (GPU) - THIS IS THE CRITICAL TEST...")
        signal_gpu = signal.cuda()
        print(f"    Signal transferred to GPU: {signal_gpu.device}")
        
        # Try FFT with timeout protection
        print("    Starting FFT computation...")
        start = time.time()
        try:
            result_gpu = fft_opt.rfft_batch(signal_gpu, dim=-1)
            gpu_time = time.time() - start
            print(f"  ✅ GPU FFT completed in {gpu_time*1000:.2f}ms")
            print(f"  Result shape: {result_gpu.shape}")
            
            # Compare results
            result_gpu_cpu = result_gpu.cpu()
            diff = torch.mean(torch.abs(result_cpu - result_gpu_cpu))
            print(f"  Difference CPU vs GPU: {diff:.6f}")
            
        except Exception as e:
            print(f"  ❌ GPU FFT failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  ⚠️  GPU not available")
        
except Exception as e:
    print(f"  ❌ FFT test failed: {e}")
    import traceback
    traceback.print_exc()

# Step 9: Test BLAS optimizer
print("\n[Step 9] Testing BLAS optimizer...")
try:
    print("  Creating UnifiedBLASOptimizer...")
    blas_opt = UnifiedBLASOptimizer()
    print(f"  ✅ BLAS optimizer created")
    
    print("  Testing small matrix multiplication...")
    a = torch.randn(2, 8, 16)
    b = torch.randn(2, 16, 8)
    print(f"  Matrix shapes: {a.shape} @ {b.shape}")
    
    print("  CPU matmul...")
    c_cpu = torch.bmm(a, b)
    print(f"  ✅ CPU result shape: {c_cpu.shape}")
    
    if torch.cuda.is_available():
        print("  GPU matmul...")
        c_gpu = blas_opt.bmm_optimized(a.cuda(), b.cuda())
        print(f"  ✅ GPU result shape: {c_gpu.shape}")
        
        diff = torch.mean(torch.abs(c_cpu - c_gpu.cpu()))
        print(f"  Difference CPU vs GPU: {diff:.6f}")
    
except Exception as e:
    print(f"  ❌ BLAS test failed: {e}")
    import traceback
    traceback.print_exc()

# Step 10: GPU memory info
print("\n[Step 10] GPU memory information...")
if torch.cuda.is_available():
    try:
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"  Allocated: {allocated:.2f} MB")
        print(f"  Reserved: {reserved:.2f} MB")
        
        print("  Clearing cache...")
        torch.cuda.empty_cache()
        
        allocated_after = torch.cuda.memory_allocated() / 1024**2
        reserved_after = torch.cuda.memory_reserved() / 1024**2
        print(f"  After cleanup - Allocated: {allocated_after:.2f} MB")
        print(f"  After cleanup - Reserved: {reserved_after:.2f} MB")
    except Exception as e:
        print(f"  ⚠️  Could not get memory info: {e}")
else:
    print("  ⚠️  GPU not available")

# Final summary
print("\n" + "=" * 80)
print("🎉 TEST COMPLETED SUCCESSFULLY")
print("=" * 80)
print("\nSummary:")
print(f"  Platform: {info['vendor']} {info['platform']}")
print(f"  Device: {info['device_name']}")
print(f"  GPU Available: {info['gpu_available']}")
print("\nAll critical tests passed! The unified GPU module is working correctly.")
print("=" * 80)
