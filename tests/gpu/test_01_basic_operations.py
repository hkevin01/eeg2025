"""
GPU Test 01: Basic Operations
Tests fundamental PyTorch operations on AMD GPU
"""

import torch
import sys
import time

def test_basic_operations():
    """Test basic tensor operations"""
    print("="*70)
    print("TEST 01: Basic GPU Operations")
    print("="*70)
    
    # Check GPU availability
    print("\n1️⃣ GPU Detection:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("   ❌ FAILED: GPU not detected")
        return False
    
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"   Device: {props.name}")
    print(f"   Arch: {props.gcnArchName}")
    print(f"   Memory: {props.total_memory / 1024**3:.2f} GB")
    print("   ✅ PASSED")
    
    # Test 2: Tensor creation
    print("\n2️⃣ Tensor Creation:")
    try:
        x = torch.randn(100, 100, device=device)
        print(f"   Created tensor: {x.shape} on {x.device}")
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 3: Basic arithmetic
    print("\n3️⃣ Basic Arithmetic:")
    try:
        y = torch.randn(100, 100, device=device)
        z = x + y
        w = x * y
        print(f"   Addition: {z.shape}")
        print(f"   Multiplication: {w.shape}")
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 4: Matrix multiplication
    print("\n4️⃣ Matrix Multiplication:")
    try:
        start = time.time()
        result = torch.matmul(x, y)
        elapsed = time.time() - start
        print(f"   Result: {result.shape}")
        print(f"   Time: {elapsed*1000:.2f} ms")
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 5: Memory transfer
    print("\n5️⃣ Memory Transfer (CPU ↔ GPU):")
    try:
        cpu_tensor = torch.randn(100, 100)
        gpu_tensor = cpu_tensor.to(device)
        back_to_cpu = gpu_tensor.cpu()
        print(f"   CPU → GPU: {gpu_tensor.device}")
        print(f"   GPU → CPU: {back_to_cpu.device}")
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 6: Large tensor (memory stress)
    print("\n6️⃣ Large Tensor (Memory Stress):")
    try:
        large = torch.randn(1000, 1000, device=device)
        result = torch.matmul(large, large)
        print(f"   Created: {large.shape} ({large.numel() * 4 / 1024**2:.2f} MB)")
        print(f"   MatMul result: {result.shape}")
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    print("\n" + "="*70)
    print("✅ ALL BASIC OPERATIONS PASSED")
    print("="*70)
    return True

if __name__ == "__main__":
    success = test_basic_operations()
    sys.exit(0 if success else 1)
