#!/usr/bin/env python3
"""
CUDA Test with Timeouts and Safety Checks
"""
import sys
import time
import signal
import threading
from contextlib import contextmanager

class TimeoutError(Exception):
    pass

@contextmanager
def timeout_context(seconds):
    """Context manager for operation timeout"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def safe_step(step_num, description, func, timeout_seconds=30):
    """Execute a step with timeout protection"""
    print(f"\n{'='*50}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*50}")
    print(f"Time: {time.strftime('%H:%M:%S')}")
    print(f"Timeout: {timeout_seconds} seconds")
    print(f"Starting...", flush=True)
    
    try:
        with timeout_context(timeout_seconds):
            result = func()
            print(f"✅ STEP {step_num} COMPLETED SUCCESSFULLY")
            return result
    except TimeoutError as e:
        print(f"⏰ STEP {step_num} TIMED OUT: {e}")
        return None
    except Exception as e:
        print(f"❌ STEP {step_num} FAILED: {e}")
        return None

def test_pytorch_import():
    """Test 1: Import PyTorch"""
    print("Importing PyTorch...", flush=True)
    import torch
    print(f"PyTorch version: {torch.__version__}", flush=True)
    
    # Check for ROCm (AMD) vs CUDA
    if hasattr(torch.version, 'hip') and torch.version.hip:
        print(f"ROCm/HIP version: {torch.version.hip}", flush=True)
        print("⚠️  AMD GPU detected - some operations may behave differently", flush=True)
    
    return torch

def test_cuda_availability(torch_module):
    """Test 2: Check CUDA"""
    print("Checking CUDA availability...", flush=True)
    cuda_available = torch_module.cuda.is_available()
    print(f"CUDA available: {cuda_available}", flush=True)
    
    if cuda_available:
        print(f"CUDA version: {torch_module.version.cuda}", flush=True)
        device_count = torch_module.cuda.device_count()
        print(f"GPU count: {device_count}", flush=True)
        
        for i in range(device_count):
            name = torch_module.cuda.get_device_name(i)
            print(f"GPU {i}: {name}", flush=True)
            
            # Check memory
            props = torch_module.cuda.get_device_properties(i)
            total_mem = props.total_memory / (1024**3)
            print(f"  Memory: {total_mem:.1f} GB", flush=True)
    
    return cuda_available

def test_basic_gpu_operation(torch_module, cuda_available):
    """Test 3: Very basic GPU operation"""
    if not cuda_available:
        print("Skipping - CUDA not available", flush=True)
        return True
    
    print("Testing basic GPU memory allocation...", flush=True)
    
    # Tiny tensor - just 1KB
    print("Creating tiny tensor (1KB)...", flush=True)
    x = torch_module.randn(16, 16, device='cuda')  # 16x16 float32 = 1KB
    print(f"✅ GPU tensor created: {x.shape}", flush=True)
    
    print("Testing basic operation...", flush=True)
    y = x + 1
    print(f"✅ Basic operation completed: {y.shape}", flush=True)
    
    # Immediate cleanup
    print("Cleaning up...", flush=True)
    del x, y
    torch_module.cuda.empty_cache()
    torch_module.cuda.synchronize()
    print("✅ Cleanup completed", flush=True)
    
    return True

def test_safe_fft(torch_module, cuda_available):
    """Test 4: Safe FFT with small sizes"""
    print("Testing CPU FFT first...", flush=True)
    
    # Very small signal - 128 samples
    signal = torch_module.randn(128)
    print(f"Signal shape: {signal.shape}", flush=True)
    
    # CPU FFT
    print("CPU FFT...", flush=True)
    start = time.time()
    fft_cpu = torch_module.fft.rfft(signal)
    cpu_time = time.time() - start
    print(f"✅ CPU FFT: {signal.shape} -> {fft_cpu.shape}, time: {cpu_time*1000:.2f} ms", flush=True)
    
    if not cuda_available:
        print("Skipping GPU FFT - CUDA not available", flush=True)
        return True
    
    print("Attempting GPU FFT with timeout protection...", flush=True)
    
    try:
        # Move to GPU
        print("Moving signal to GPU...", flush=True)
        signal_gpu = signal.cuda()
        print("✅ Signal moved to GPU", flush=True)
        
        # GPU FFT with explicit synchronization
        print("Starting GPU FFT...", flush=True)
        start = time.time()
        fft_gpu = torch_module.fft.rfft(signal_gpu)
        
        print("Synchronizing GPU...", flush=True)
        torch_module.cuda.synchronize()  # Force completion
        gpu_time = time.time() - start
        
        print(f"✅ GPU FFT: {signal_gpu.shape} -> {fft_gpu.shape}, time: {gpu_time*1000:.2f} ms", flush=True)
        
        if gpu_time > 0 and cpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"Speedup: {speedup:.2f}x", flush=True)
        
        # Cleanup
        print("Cleaning up GPU FFT...", flush=True)
        del signal_gpu, fft_gpu
        torch_module.cuda.empty_cache()
        torch_module.cuda.synchronize()
        print("✅ GPU FFT cleanup completed", flush=True)
        
    except Exception as e:
        print(f"⚠️  GPU FFT failed: {e}", flush=True)
        print("This is common on some GPU configurations", flush=True)
        
        # Emergency cleanup
        try:
            torch_module.cuda.empty_cache()
            torch_module.cuda.synchronize()
        except:
            pass
    
    return True

def test_matrix_operations(torch_module, cuda_available):
    """Test 5: Matrix operations"""
    print("Testing CPU matrix operations...", flush=True)
    
    # Small matrices
    a = torch_module.randn(64, 64)
    b = torch_module.randn(64, 64)
    print(f"Matrix shapes: {a.shape}, {b.shape}", flush=True)
    
    # CPU matmul
    start = time.time()
    c_cpu = torch_module.matmul(a, b)
    cpu_time = time.time() - start
    print(f"✅ CPU matmul: {cpu_time*1000:.2f} ms", flush=True)
    
    if not cuda_available:
        print("Skipping GPU matmul - CUDA not available", flush=True)
        return True
    
    try:
        print("Testing GPU matrix operations...", flush=True)
        a_gpu = a.cuda()
        b_gpu = b.cuda()
        print("✅ Matrices moved to GPU", flush=True)
        
        start = time.time()
        c_gpu = torch_module.matmul(a_gpu, b_gpu)
        torch_module.cuda.synchronize()
        gpu_time = time.time() - start
        
        print(f"✅ GPU matmul: {gpu_time*1000:.2f} ms", flush=True)
        
        if gpu_time > 0 and cpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"Speedup: {speedup:.2f}x", flush=True)
        
        # Cleanup
        del a_gpu, b_gpu, c_gpu
        torch_module.cuda.empty_cache()
        torch_module.cuda.synchronize()
        print("✅ GPU matmul cleanup completed", flush=True)
        
    except Exception as e:
        print(f"⚠️  GPU matmul failed: {e}", flush=True)
        try:
            torch_module.cuda.empty_cache()
        except:
            pass
    
    return True

def main():
    print("🚀 CUDA TEST WITH TIMEOUT PROTECTION")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Import PyTorch (30 second timeout)
    torch_module = safe_step(1, "Import PyTorch", test_pytorch_import, 30)
    if not torch_module:
        print("❌ Cannot continue without PyTorch")
        return
    
    time.sleep(1)
    
    # Step 2: Check CUDA (15 second timeout)
    cuda_available = safe_step(2, "Check CUDA", 
                              lambda: test_cuda_availability(torch_module), 15)
    
    time.sleep(1)
    
    # Step 3: Basic GPU operation (20 second timeout)
    safe_step(3, "Basic GPU Operation", 
              lambda: test_basic_gpu_operation(torch_module, cuda_available), 20)
    
    time.sleep(1)
    
    # Step 4: FFT operations (45 second timeout - FFT can be slow)
    safe_step(4, "FFT Operations", 
              lambda: test_safe_fft(torch_module, cuda_available), 45)
    
    time.sleep(1)
    
    # Step 5: Matrix operations (30 second timeout)
    safe_step(5, "Matrix Operations", 
              lambda: test_matrix_operations(torch_module, cuda_available), 30)
    
    print(f"\n{'='*60}")
    print("🎉 ALL TESTS COMPLETED!")
    print(f"End time: {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    # Final cleanup
    if cuda_available:
        try:
            torch_module.cuda.empty_cache()
            print("✅ Final GPU cleanup completed")
        except:
            pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n⚠️ Test interrupted at {time.strftime('%H:%M:%S')}")
        # Emergency cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
