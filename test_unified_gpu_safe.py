#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe Testing for Unified GPU Optimization Module
==============================================

Comprehensive testing with AMD GPU safety features.
"""

import sys
import os
import time
import signal
import torch

# Add src to path
sys.path.insert(0, '/home/kevin/Projects/eeg2025/src')

def timeout_handler(signum, frame):
    """Handle timeout for GPU operations"""
    print("\n⚠️  TIMEOUT: GPU operation taking too long - aborting for safety")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    sys.exit(1)

def safe_gpu_test():
    """Test unified GPU module with safety measures"""
    print("🔒 Unified GPU Safe Testing")
    print("=" * 60)
    
    try:
        # Set timeout for safety (30 seconds)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        # Import unified module
        from gpu.unified_gpu_optimized import (
            GPUPlatformDetector,
            UnifiedFFTOptimizer,
            UnifiedBLASOptimizer,
            UnifiedLinearLayer
        )
        
        print("✅ Module import successful")
        
        # Platform detection
        print("\n🔍 Testing platform detection...")
        detector = GPUPlatformDetector()
        detector.print_info()
        
        if not detector.gpu_available:
            print("⚠️  No GPU available - testing CPU fallback only")
            signal.alarm(0)  # Cancel timeout
            return
        
        # Test FFT optimizer
        print("\n🔧 Testing FFT optimizer...")
        fft_opt = UnifiedFFTOptimizer()
        
        # Small test data for safety
        x = torch.randn(2, 16, 1000)  # Small batch
        print(f"   Input shape: {x.shape}")
        
        # Test FFT
        X = fft_opt.rfft_batch(x, dim=-1)
        print(f"   FFT output shape: {X.shape}")
        
        # Test inverse FFT
        x_reconstructed = fft_opt.irfft_batch(X, dim=-1)
        print(f"   iFFT output shape: {x_reconstructed.shape}")
        
        # Verify reconstruction
        error = torch.mean(torch.abs(x - x_reconstructed[:, :, :x.shape[-1]]))
        print(f"   Reconstruction error: {error:.6f}")
        
        if error < 1e-5:
            print("   ✅ FFT test passed")
        else:
            print("   ❌ FFT test failed")
        
        # Test BLAS optimizer
        print("\n🔧 Testing BLAS optimizer...")
        blas_opt = UnifiedBLASOptimizer()
        
        # Small matrices for safety
        a = torch.randn(4, 32, 64)
        b = torch.randn(4, 64, 32)
        
        c = blas_opt.bmm_optimized(a, b)
        print(f"   Matrix multiply result shape: {c.shape}")
        
        # Verify with CPU
        a_cpu = a.cpu()
        b_cpu = b.cpu()
        c_cpu = torch.bmm(a_cpu, b_cpu)
        c_from_gpu = c.cpu() if c.is_cuda else c
        
        error = torch.mean(torch.abs(c_cpu - c_from_gpu))
        print(f"   GPU vs CPU error: {error:.6f}")
        
        if error < 1e-4:
            print("   ✅ BLAS test passed")
        else:
            print("   ❌ BLAS test failed")
        
        # Test unified linear layer
        print("\n🔧 Testing unified linear layer...")
        linear = UnifiedLinearLayer(64, 32)
        
        x_linear = torch.randn(8, 64)
        y = linear(x_linear)
        print(f"   Linear layer output shape: {y.shape}")
        
        if y.shape == (8, 32):
            print("   ✅ Linear layer test passed")
        else:
            print("   ❌ Linear layer test failed")
        
        # Cleanup
        print("\n🧹 Cleaning up GPU memory...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            print(f"   GPU memory - Allocated: {allocated/1024/1024:.1f}MB, Cached: {cached/1024/1024:.1f}MB")
        
        print("\n✅ All tests completed successfully!")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always cleanup
        signal.alarm(0)  # Cancel timeout
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n🔒 Test completed safely")

def minimal_functionality_test():
    """Minimal test to verify basic functionality"""
    print("\n" + "="*40)
    print("Minimal Functionality Test")
    print("="*40)
    
    try:
        # Just test imports and basic detection
        from gpu.unified_gpu_optimized import GPUPlatformDetector
        
        detector = GPUPlatformDetector()
        info = detector.get_info()
        
        print(f"✅ Import successful")
        print(f"✅ GPU Available: {info['gpu_available']}")
        
        if info['gpu_available']:
            print(f"✅ Vendor: {info['vendor']}")
            print(f"✅ Platform: {info['platform']}")
            print(f"✅ Device: {info['device_name']}")
            
            # Test very basic tensor operation
            x = torch.tensor([1.0, 2.0, 3.0])
            if torch.cuda.is_available():
                x_gpu = x.cuda()
                x_back = x_gpu.cpu()
                print(f"✅ Basic GPU tensor transfer successful")
        
        print("✅ Minimal test passed")
        
    except Exception as e:
        print(f"❌ Minimal test failed: {e}")

if __name__ == "__main__":
    print("Starting unified GPU testing...")
    
    # Run minimal test first
    minimal_functionality_test()
    
    # Ask user before full test
    response = input("\nRun full GPU test? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        safe_gpu_test()
    else:
        print("Skipping full GPU test for safety")
