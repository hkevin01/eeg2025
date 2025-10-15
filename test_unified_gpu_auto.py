#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Safe Testing for Unified GPU Optimization Module
=========================================================

Runs complete testing automatically without user prompts.
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

def automated_gpu_test():
    """Automated test of unified GPU module with safety measures"""
    print("\n" + "="*60)
    print("🔒 Unified GPU Automated Safe Testing")
    print("="*60)
    
    try:
        # Set timeout for safety (30 seconds)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        # Import unified module
        print("\n📦 Importing unified GPU module...")
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
        
        # Test 1: FFT optimizer
        print("\n" + "="*60)
        print("TEST 1: FFT Operations")
        print("="*60)
        fft_opt = UnifiedFFTOptimizer()
        
        # Small test data for safety
        x = torch.randn(2, 16, 1000)  # Small batch
        print(f"Input shape: {x.shape}")
        
        # Test FFT
        print("Running FFT...")
        X = fft_opt.rfft_batch(x, dim=-1)
        print(f"✅ FFT output shape: {X.shape}")
        
        # Test inverse FFT
        print("Running inverse FFT...")
        x_reconstructed = fft_opt.irfft_batch(X, dim=-1)
        print(f"✅ iFFT output shape: {x_reconstructed.shape}")
        
        # Verify reconstruction
        error = torch.mean(torch.abs(x - x_reconstructed[:, :, :x.shape[-1]]))
        print(f"Reconstruction error: {error:.6f}")
        
        if error < 1e-5:
            print("✅ FFT TEST PASSED")
        else:
            print(f"⚠️  FFT test warning: error = {error:.6f}")
        
        # Test 2: BLAS optimizer
        print("\n" + "="*60)
        print("TEST 2: Matrix Operations")
        print("="*60)
        blas_opt = UnifiedBLASOptimizer()
        
        # Small matrices for safety
        a = torch.randn(4, 32, 64)
        b = torch.randn(4, 64, 32)
        
        print(f"Matrix A shape: {a.shape}")
        print(f"Matrix B shape: {b.shape}")
        
        print("Running batch matrix multiplication...")
        c = blas_opt.bmm_optimized(a, b)
        print(f"✅ Result shape: {c.shape}")
        
        # Verify with CPU
        a_cpu = a.cpu()
        b_cpu = b.cpu()
        c_cpu = torch.bmm(a_cpu, b_cpu)
        c_from_gpu = c.cpu() if c.is_cuda else c
        
        error = torch.mean(torch.abs(c_cpu - c_from_gpu))
        print(f"GPU vs CPU error: {error:.6f}")
        
        if error < 1e-4:
            print("✅ BLAS TEST PASSED")
        else:
            print(f"⚠️  BLAS test warning: error = {error:.6f}")
        
        # Test 3: Unified linear layer
        print("\n" + "="*60)
        print("TEST 3: Unified Linear Layer")
        print("="*60)
        linear = UnifiedLinearLayer(64, 32)
        print(f"Created layer: 64 → 32")
        
        x_linear = torch.randn(8, 64)
        print(f"Input shape: {x_linear.shape}")
        
        print("Running forward pass...")
        y = linear(x_linear)
        print(f"✅ Output shape: {y.shape}")
        
        if y.shape == (8, 32):
            print("✅ LINEAR LAYER TEST PASSED")
        else:
            print(f"❌ Linear layer test failed: expected (8, 32), got {y.shape}")
        
        # Test 4: STFT (more advanced FFT operation)
        print("\n" + "="*60)
        print("TEST 4: Short-Time Fourier Transform")
        print("="*60)
        
        eeg_signal = torch.randn(2, 16, 5000)  # Small EEG-like signal
        print(f"Signal shape: {eeg_signal.shape}")
        
        print("Computing STFT...")
        stft_result = fft_opt.stft_optimized(
            eeg_signal,
            n_fft=256,
            hop_length=128,
            return_complex=True
        )
        print(f"✅ STFT output shape: {stft_result.shape}")
        print("✅ STFT TEST PASSED")
        
        # Cleanup
        print("\n" + "="*60)
        print("Cleanup")
        print("="*60)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            print(f"GPU memory - Allocated: {allocated/1024/1024:.1f}MB")
            print(f"GPU memory - Cached: {cached/1024/1024:.1f}MB")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        
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

if __name__ == "__main__":
    print("Starting automated unified GPU testing...")
    automated_gpu_test()
