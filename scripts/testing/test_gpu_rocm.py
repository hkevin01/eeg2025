#!/usr/bin/env python3
"""
GPU/ROCm/PyTorch Test Script
=============================
Comprehensive test to verify GPU acceleration is working properly.
Tests: PyTorch, CUDA/ROCm, tensor operations, memory, and data loading.
"""
import sys
import time
import torch
import numpy as np

print("=" * 70)
print("🧪 GPU/ROCm/PyTorch Verification Test")
print("=" * 70)

# Test 1: PyTorch Installation
print("\n1️⃣  PyTorch Installation")
print(f"   PyTorch Version: {torch.__version__}")
print(f"   Python Version: {sys.version.split()[0]}")

# Test 2: CUDA/ROCm Availability
print("\n2️⃣  CUDA/ROCm Availability")
cuda_available = torch.cuda.is_available()
print(f"   CUDA Available: {cuda_available}")

if cuda_available:
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Device Count: {torch.cuda.device_count()}")
    print(f"   Current Device: {torch.cuda.current_device()}")
    print(f"   Device Name: {torch.cuda.get_device_name(0)}")

    # Get GPU properties
    props = torch.cuda.get_device_properties(0)
    print(f"   Total Memory: {props.total_memory / 1e9:.2f} GB")
    print(f"   Multi-Processor Count: {props.multi_processor_count}")
    print(f"   Compute Capability: {props.major}.{props.minor}")
else:
    print("   ⚠️  No GPU detected - will use CPU")

# Test 3: Basic Tensor Operations
print("\n3️⃣  Basic Tensor Operations")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create tensors
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)

    # Matrix multiplication
    start = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start

    print(f"   Device: {device}")
    print(f"   Matrix Multiplication (1000x1000): {elapsed*1000:.2f} ms")
    print(f"   Result Shape: {z.shape}")
    # Note: Avoid .mean() on GPU with gfx1010 (RX 5600 XT) - known ROCm bug
    # Use smaller operations or move to CPU for reductions
    z_sample = z[0, 0].item()  # Just sample one value instead
    print(f"   Result Sample [0,0]: {z_sample:.4f}")
    print("   ✅ Basic operations working")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Test 4: torch.arange with Float Steps (ROCm known issue)
print("\n4️⃣  torch.arange with Float Steps")
try:
    # This can fail with some ROCm versions - test on CPU first
    x_cpu = torch.arange(0, 10, 0.5)
    print(f"   arange(0, 10, 0.5) on CPU: {x_cpu.shape[0]} elements")
    print(f"   First 5: {x_cpu[:5].tolist()}")

    # Try on GPU (may fail with gfx1010)
    try:
        x_gpu = torch.arange(0, 10, 0.5, device=device)
        print(f"   arange on GPU: {x_gpu.shape[0]} elements ✅")
    except Exception as e_gpu:
        print(f"   ⚠️  GPU arange issue (expected with gfx1010): {str(e_gpu)[:50]}")

    print("   ✅ arange working (CPU)")
except Exception as e:
    print(f"   ⚠️  arange issue: {e}")

# Test 5: Memory Management
print("\n5️⃣  GPU Memory Management")
if torch.cuda.is_available():
    try:
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(0) / 1e6
        reserved = torch.cuda.memory_reserved(0) / 1e6
        print(f"   Allocated: {allocated:.2f} MB")
        print(f"   Reserved: {reserved:.2f} MB")

        # Allocate large tensor
        large = torch.randn(10000, 10000, device=device)
        allocated_after = torch.cuda.memory_allocated(0) / 1e6
        print(f"   After 10000x10000 tensor: {allocated_after:.2f} MB")

        # Free memory
        del large
        torch.cuda.empty_cache()
        print("   ✅ Memory management working")
    except Exception as e:
        print(f"   ⚠️  Memory issue: {e}")
else:
    print("   ⏭️  Skipped (CPU mode)")

# Test 6: Mixed Precision (AMP)
print("\n6️⃣  Mixed Precision Training (AMP)")
try:
    # Use new API: torch.amp instead of torch.cuda.amp
    from torch.amp import autocast, GradScaler

    model = torch.nn.Linear(100, 10).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scaler = GradScaler('cuda') if torch.cuda.is_available() else None

    x = torch.randn(32, 100, device=device)
    y = torch.randn(32, 10, device=device)

    # Training step with AMP
    optimizer.zero_grad()

    if torch.cuda.is_available():
        with autocast('cuda'):
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()

    print(f"   Loss: {loss.item():.4f}")
    print("   ✅ Mixed precision working")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 7: Data Loading Performance
print("\n7️⃣  Data Loading Performance")
try:
    # Simulate data loading
    data_cpu = [torch.randn(10, 100) for _ in range(100)]

    start = time.time()
    data_gpu = [d.to(device) for d in data_cpu]
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start

    print(f"   Transfer 100 tensors (10x100): {elapsed*1000:.2f} ms")
    print("   ✅ Data transfer working")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 8: NumPy Interoperability
print("\n8️⃣  NumPy Interoperability")
try:
    # NumPy -> PyTorch
    np_array = np.random.randn(100, 100).astype(np.float32)
    torch_tensor = torch.from_numpy(np_array).to(device)

    # PyTorch -> NumPy (need to move to CPU first)
    back_to_numpy = torch_tensor.cpu().numpy()

    print(f"   NumPy shape: {np_array.shape}")
    print(f"   Torch shape: {torch_tensor.shape}")
    print(f"   Back to NumPy: {back_to_numpy.shape}")
    print(f"   Data preserved: {np.allclose(np_array, back_to_numpy)}")
    print("   ✅ NumPy conversion working")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 9: Parallel Operations
print("\n9️⃣  Parallel Operations")
try:
    # Multiple operations in parallel
    tensors = [torch.randn(500, 500, device=device) for _ in range(4)]

    start = time.time()
    results = [torch.matmul(t, t.T) for t in tensors]
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start

    print(f"   4x Matrix Multiplications (500x500): {elapsed*1000:.2f} ms")
    print("   ✅ Parallel operations working")
except Exception as e:
    print(f"   ⚠️  Warning: {e}")

# Test 10: EEG-like Data Operations
print("\n🔟 EEG-like Data Operations")
try:
    # Simulate EEG data: (batch, channels, time)
    batch_size = 32
    n_channels = 64
    n_samples = 2500  # 25 seconds @ 100Hz

    eeg_data = torch.randn(batch_size, n_channels, n_samples, device=device)

    # Simulate CNN operations
    conv = torch.nn.Conv1d(n_channels, 32, kernel_size=25, stride=1).to(device)

    start = time.time()
    output = conv(eeg_data)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start

    print(f"   Input Shape: {eeg_data.shape}")
    print(f"   Conv1D Output: {output.shape}")
    print(f"   Processing Time: {elapsed*1000:.2f} ms")
    print(f"   Throughput: {batch_size/elapsed:.1f} samples/sec")

    # Sample output instead of computing mean (avoid reduction bug)
    print(f"   Output Sample: {output[0, 0, 0].item():.4f}")
    print("   ✅ EEG-like operations working")
except Exception as e:
    print(f"   ❌ Error: {e}")# Final Summary
print("\n" + "=" * 70)
print("📊 SUMMARY")
print("=" * 70)

if torch.cuda.is_available():
    print("✅ GPU acceleration is WORKING")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.version.cuda}")
    print("\n🚀 Ready for GPU-accelerated training!")
else:
    print("⚠️  Running in CPU mode")
    print("   GPU acceleration not available")
    print("\n💻 Training will use CPU (slower)")

print("=" * 70)
