#!/usr/bin/env python3
"""
Quick test of EEGNeX with ROCm GPU acceleration.
Tests both CPU and GPU modes with a small sample.
"""
import time
import torch
from braindecode.models import EEGNeX

print("="*80)
print("🧪 TESTING EEGNeX WITH ROCM GPU ACCELERATION")
print("="*80)
print()

# Test parameters
n_chans = 129
n_times = 200  # 2 seconds at 100 Hz
batch_size = 16
n_batches = 50

# Check CUDA availability
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)
print()

def test_device(device_name):
    """Test training on a specific device."""
    device = torch.device(device_name)
    print(f"Testing on {device}...")
    
    # Create model
    model = EEGNeX(
        n_chans=n_chans,
        n_outputs=1,
        n_times=n_times,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.L1Loss()
    
    # Warmup
    print(f"  Warmup (5 batches)...")
    for _ in range(5):
        x = torch.randn(batch_size, n_chans, n_times, device=device)
        y = torch.randn(batch_size, 1, device=device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    
    # Timed run
    print(f"  Timed run ({n_batches} batches)...")
    start_time = time.time()
    
    for i in range(n_batches):
        x = torch.randn(batch_size, n_chans, n_times, device=device)
        y = torch.randn(batch_size, 1, device=device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"    Batch {i}/{n_batches}, Loss: {loss.item():.6f}")
    
    elapsed = time.time() - start_time
    batches_per_sec = n_batches / elapsed
    
    print(f"✅ Completed {n_batches} batches in {elapsed:.2f}s")
    print(f"   Throughput: {batches_per_sec:.2f} batches/sec")
    print(f"   Time per batch: {elapsed/n_batches*1000:.2f}ms")
    
    if device.type == "cuda":
        print(f"   GPU Memory: {torch.cuda.memory_allocated()/1024**3:.3f} GB")
    
    return elapsed

# Test CPU
print("="*80)
print("TEST 1: CPU PERFORMANCE")
print("="*80)
cpu_time = test_device("cpu")
print()

# Test GPU if available
if torch.cuda.is_available():
    print("="*80)
    print("TEST 2: GPU PERFORMANCE (ROCM)")
    print("="*80)
    try:
        gpu_time = test_device("cuda:0")
        print()
        
        print("="*80)
        print("PERFORMANCE COMPARISON")
        print("="*80)
        speedup = cpu_time / gpu_time
        print(f"CPU Time:  {cpu_time:.2f}s")
        print(f"GPU Time:  {gpu_time:.2f}s")
        print(f"Speedup:   {speedup:.2f}x")
        
        if speedup > 1.5:
            print("✅ GPU acceleration working well!")
        elif speedup > 1.0:
            print("⚠️  GPU slightly faster, but could be better")
        else:
            print("❌ GPU slower than CPU - check ROCm configuration")
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        print("   Check ROCm environment variables:")
        print("   - HSA_OVERRIDE_GFX_VERSION=10.3.0")
        print("   - PYTORCH_ROCM_ARCH=gfx1030")
else:
    print("ℹ️  GPU not available, skipping GPU test")

print()
print("="*80)
print("🎉 TEST COMPLETE")
print("="*80)
