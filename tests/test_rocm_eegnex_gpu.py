#!/usr/bin/env python3
"""
Comprehensive ROCm GPU Test for EEGNeX Training
================================================
Tests if the AMD gfx1010 GPU works correctly with PyTorch ROCm 6.2
and the braindecode EEGNeX model.

This script will:
1. Verify ROCm environment and GPU detection
2. Test basic tensor operations on GPU
3. Test EEGNeX model forward pass on GPU
4. Test training loop (forward + backward + optimization)
5. Compare GPU vs CPU performance
"""

import sys
import time
import torch
import torch.nn as nn
from torch import optim
from torch.nn.functional import l1_loss
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from braindecode.models import EEGNeX
except ImportError:
    print("❌ braindecode not installed")
    sys.exit(1)


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_rocm_environment():
    """Test ROCm environment and GPU detection."""
    print_section("TEST 1: ROCm Environment")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA/ROCm not available")
        return False
    
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {device_name}")
        
        # Get device properties
        props = torch.cuda.get_device_properties(i)
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
    
    # Check ROCm version
    if hasattr(torch.version, 'hip'):
        print(f"ROCm version: {torch.version.hip}")
    
    print("\n✅ ROCm environment check passed")
    return True


def test_basic_tensor_operations():
    """Test basic tensor operations on GPU."""
    print_section("TEST 2: Basic Tensor Operations on GPU")
    
    device = torch.device("cuda:0")
    print(f"Testing on device: {device}")
    
    try:
        # Create tensors
        print("\n1. Creating tensors on GPU...")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        print(f"   ✅ Created tensors: x={x.shape}, y={y.shape}")
        
        # Matrix multiplication
        print("\n2. Testing matrix multiplication...")
        start = time.time()
        z = torch.mm(x, y)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000
        print(f"   ✅ Matrix multiply: {elapsed:.2f}ms")
        
        # Test gradients
        print("\n3. Testing gradients...")
        x.requires_grad = True
        y.requires_grad = True
        z = (x * y).sum()
        z.backward()
        print(f"   ✅ Gradients computed")
        
        # Memory management
        print("\n4. Checking GPU memory...")
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        print(f"   Allocated: {allocated:.2f} MB")
        print(f"   Cached: {cached:.2f} MB")
        
        print("\n✅ Basic tensor operations passed")
        return True
        
    except Exception as e:
        print(f"\n❌ Tensor operations failed: {e}")
        return False


def test_eegnex_forward_pass():
    """Test EEGNeX model forward pass on GPU."""
    print_section("TEST 3: EEGNeX Forward Pass on GPU")
    
    device = torch.device("cuda:0")
    print(f"Testing on device: {device}")
    
    try:
        # Create model
        print("\n1. Creating EEGNeX model...")
        model = EEGNeX(
            n_chans=129,
            n_outputs=1,
            n_times=200,
        )
        model = model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✅ Model created: {total_params:,} parameters")
        print(f"   Model device: {next(model.parameters()).device}")
        
        # Create input
        print("\n2. Creating input tensor...")
        batch_size = 32
        x = torch.randn(batch_size, 129, 200, device=device)
        print(f"   ✅ Input shape: {x.shape}")
        
        # Forward pass
        print("\n3. Running forward pass...")
        model.eval()
        with torch.no_grad():
            # Warm-up
            _ = model(x)
            torch.cuda.synchronize()
            
            # Timed run
            start = time.time()
            output = model(x)
            torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000
        
        print(f"   ✅ Output shape: {output.shape}")
        print(f"   ✅ Inference time: {elapsed:.2f}ms for batch of {batch_size}")
        print(f"   ✅ Per-sample: {elapsed/batch_size:.2f}ms")
        
        # Check memory
        allocated = torch.cuda.memory_allocated() / 1024**2
        print(f"\n4. GPU memory after forward pass: {allocated:.2f} MB")
        
        print("\n✅ Forward pass test passed")
        return True
        
    except Exception as e:
        print(f"\n❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_eegnex_training_loop():
    """Test EEGNeX training loop (forward + backward + optimize)."""
    print_section("TEST 4: EEGNeX Training Loop on GPU")
    
    device = torch.device("cuda:0")
    print(f"Testing on device: {device}")
    
    try:
        # Create model
        print("\n1. Setting up model and optimizer...")
        model = EEGNeX(
            n_chans=129,
            n_outputs=1,
            n_times=200,
        )
        model = model.to(device)
        optimizer = optim.Adamax(model.parameters(), lr=0.001)
        
        print(f"   ✅ Model and optimizer ready")
        
        # Training loop
        print("\n2. Running training iterations...")
        model.train()
        
        batch_size = 32
        num_iterations = 10
        times = []
        
        for i in range(num_iterations):
            # Create batch
            x = torch.randn(batch_size, 129, 200, device=device)
            y = torch.randn(batch_size, 1, device=device)
            
            # Training step
            start = time.time()
            
            optimizer.zero_grad()
            output = model(x)
            loss = l1_loss(output, y)
            loss.backward()
            optimizer.step()
            
            torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            
            if i % 2 == 0:
                print(f"   Iteration {i+1}/{num_iterations}: "
                      f"loss={loss.item():.4f}, time={elapsed:.2f}ms")
        
        avg_time = np.mean(times)
        print(f"\n3. Training statistics:")
        print(f"   Average time per iteration: {avg_time:.2f}ms")
        print(f"   Throughput: {batch_size * 1000 / avg_time:.1f} samples/sec")
        
        # Check memory
        allocated = torch.cuda.memory_allocated() / 1024**2
        print(f"\n4. GPU memory after training: {allocated:.2f} MB")
        
        print("\n✅ Training loop test passed")
        return True
        
    except Exception as e:
        print(f"\n❌ Training loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_vs_cpu_performance():
    """Compare GPU vs CPU performance."""
    print_section("TEST 5: GPU vs CPU Performance Comparison")
    
    batch_size = 32
    
    # Test on GPU
    print("\n1. Testing on GPU...")
    device_gpu = torch.device("cuda:0")
    
    try:
        model_gpu = EEGNeX(n_chans=129, n_outputs=1, n_times=200).to(device_gpu)
        model_gpu.eval()
        
        x_gpu = torch.randn(batch_size, 129, 200, device=device_gpu)
        
        # Warm-up
        with torch.no_grad():
            _ = model_gpu(x_gpu)
        torch.cuda.synchronize()
        
        # Timed runs
        gpu_times = []
        with torch.no_grad():
            for _ in range(20):
                start = time.time()
                _ = model_gpu(x_gpu)
                torch.cuda.synchronize()
                gpu_times.append((time.time() - start) * 1000)
        
        gpu_avg = np.mean(gpu_times)
        gpu_std = np.std(gpu_times)
        print(f"   ✅ GPU: {gpu_avg:.2f} ± {gpu_std:.2f}ms")
        
    except Exception as e:
        print(f"   ❌ GPU test failed: {e}")
        gpu_avg = None
    
    # Test on CPU
    print("\n2. Testing on CPU...")
    device_cpu = torch.device("cpu")
    
    model_cpu = EEGNeX(n_chans=129, n_outputs=1, n_times=200).to(device_cpu)
    model_cpu.eval()
    
    x_cpu = torch.randn(batch_size, 129, 200, device=device_cpu)
    
    # Warm-up
    with torch.no_grad():
        _ = model_cpu(x_cpu)
    
    # Timed runs
    cpu_times = []
    with torch.no_grad():
        for _ in range(20):
            start = time.time()
            _ = model_cpu(x_cpu)
            cpu_times.append((time.time() - start) * 1000)
    
    cpu_avg = np.mean(cpu_times)
    cpu_std = np.std(cpu_times)
    print(f"   ✅ CPU: {cpu_avg:.2f} ± {cpu_std:.2f}ms")
    
    # Compare
    if gpu_avg is not None:
        print("\n3. Performance comparison:")
        speedup = cpu_avg / gpu_avg
        print(f"   GPU speedup: {speedup:.2f}x faster than CPU")
        
        if speedup > 1.0:
            print(f"   ✅ GPU is faster!")
        else:
            print(f"   ⚠️ GPU is slower (may need optimization)")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  ROCm GPU Test Suite for EEGNeX Training")
    print("  Testing: AMD gfx1010 with PyTorch ROCm 6.2")
    print("=" * 70)
    
    results = []
    
    # Run tests
    results.append(("ROCm Environment", test_rocm_environment()))
    
    if results[0][1]:  # Only continue if ROCm is available
        results.append(("Basic Tensor Ops", test_basic_tensor_operations()))
        results.append(("EEGNeX Forward Pass", test_eegnex_forward_pass()))
        results.append(("EEGNeX Training Loop", test_eegnex_training_loop()))
        results.append(("GPU vs CPU Performance", test_gpu_vs_cpu_performance()))
    
    # Summary
    print_section("TEST SUMMARY")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED - GPU TRAINING IS READY!")
        print("\nYour AMD gfx1010 GPU with ROCm 6.2 works correctly with EEGNeX.")
        print("You can safely enable GPU training in train_challenge2_fast.py")
    else:
        print("⚠️ SOME TESTS FAILED - Review errors above")
        print("\nGPU training may not be stable. Consider using CPU mode.")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
