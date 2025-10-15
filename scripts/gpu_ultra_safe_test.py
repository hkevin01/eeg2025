#!/usr/bin/env python3
"""
Ultra-safe GPU test with multiple guardrails to prevent system crashes
- Timeout wrapper
- Progressive testing (stop at first sign of trouble)
- Minimal operations
- No actual training, just device checks
"""

import os
import sys
import signal
import time
from pathlib import Path

# Set environment variables FIRST
os.environ['ROCM_PATH'] = '/opt/rocm'
os.environ['HIP_PATH'] = '/opt/rocm'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['HIP_PLATFORM'] = 'amd'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:64'  # Very conservative

print("🛡️  Ultra-Safe GPU Test")
print("=" * 70)
print("⚠️  This test has multiple guardrails to prevent system crashes")
print()

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out!")

def test_with_timeout(func, timeout_seconds, description):
    """Run a function with timeout protection"""
    print(f"\n🧪 Test: {description}")
    print(f"   Timeout: {timeout_seconds}s")
    
    # Set alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        start = time.time()
        result = func()
        elapsed = time.time() - start
        signal.alarm(0)  # Cancel alarm
        print(f"   ✅ PASS ({elapsed:.2f}s)")
        return True, result
    except TimeoutException:
        signal.alarm(0)
        print(f"   ❌ TIMEOUT - Operation hung!")
        return False, None
    except Exception as e:
        signal.alarm(0)
        print(f"   ❌ ERROR: {e}")
        return False, None

def test_1_import_torch():
    """Test 1: Import PyTorch"""
    import torch
    return torch.__version__

def test_2_check_cuda():
    """Test 2: Check if CUDA is available"""
    import torch
    return torch.cuda.is_available()

def test_3_get_device_name():
    """Test 3: Get GPU device name"""
    import torch
    if not torch.cuda.is_available():
        return "No GPU"
    return torch.cuda.get_device_name(0)

def test_4_create_device():
    """Test 4: Create device object"""
    import torch
    return torch.device('cuda:0')

def test_5_small_cpu_tensor():
    """Test 5: Create small tensor on CPU"""
    import torch
    t = torch.randn(5, 5)
    return t.shape

def test_6_tiny_gpu_tensor():
    """Test 6: Create TINY tensor on GPU (5x5)"""
    import torch
    t = torch.randn(5, 5, device='cuda:0')
    return t.shape

def test_7_tiny_gpu_operation():
    """Test 7: Tiny GPU operation (5x5 matmul)"""
    import torch
    a = torch.randn(5, 5, device='cuda:0')
    b = torch.randn(5, 5, device='cuda:0')
    c = a @ b
    return c.shape

def test_8_move_to_cpu():
    """Test 8: Move tensor from GPU to CPU"""
    import torch
    a = torch.randn(5, 5, device='cuda:0')
    b = a.cpu()
    return b.shape

def main():
    print("\n" + "=" * 70)
    print("Starting Progressive GPU Tests")
    print("=" * 70)
    print("\n⚠️  Each test has a timeout. If any test hangs, we STOP.")
    print("⚠️  Press Ctrl+C at ANY time to abort safely.\n")
    
    tests = [
        (test_1_import_torch, 5, "Import PyTorch"),
        (test_2_check_cuda, 5, "Check CUDA availability"),
        (test_3_get_device_name, 10, "Get GPU device name"),
        (test_4_create_device, 5, "Create device object"),
        (test_5_small_cpu_tensor, 5, "Create CPU tensor"),
        (test_6_tiny_gpu_tensor, 15, "Create TINY GPU tensor (5x5) - CRITICAL"),
        (test_7_tiny_gpu_operation, 15, "Tiny GPU matmul (5x5) - CRITICAL"),
        (test_8_move_to_cpu, 10, "Move GPU->CPU"),
    ]
    
    results = []
    
    for i, (test_func, timeout, desc) in enumerate(tests, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}/{len(tests)}: {desc}")
        print(f"{'='*70}")
        
        success, result = test_with_timeout(test_func, timeout, desc)
        results.append((desc, success, result))
        
        if not success:
            print(f"\n❌ Test {i} FAILED or TIMED OUT")
            print("�� STOPPING HERE for safety")
            break
        
        print(f"   Result: {result}")
        
        # Small delay between tests
        time.sleep(1)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for desc, success, result in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {desc}")
    
    print(f"\n{'='*70}")
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        print("   GPU appears safe for basic operations")
        print("   ⚠️  However, full training may still cause issues")
    elif passed >= 5:
        print(f"\n⚠️  Partial success ({passed}/{total})")
        print("   GPU detected but operations may hang")
        print("   🛑 DO NOT attempt full training")
    else:
        print(f"\n❌ Most tests failed ({passed}/{total})")
        print("   GPU not safe to use")
        print("   ✅ Stick with CPU training")
    
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user - GOOD!")
        print("   Better to stop early than crash the system")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
