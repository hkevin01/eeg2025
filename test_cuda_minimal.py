#!/usr/bin/env python3
"""
Minimal CUDA Test - Step by Step with Immediate Output
"""
import sys
import time

def step_with_output(step_num, description, func):
    """Execute a step with immediate output"""
    print(f"\n{'='*50}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*50}")
    print(f"Time: {time.strftime('%H:%M:%S')}")
    print(f"Starting...", flush=True)
    
    try:
        result = func()
        print(f"✅ STEP {step_num} COMPLETED SUCCESSFULLY")
        return result
    except Exception as e:
        print(f"❌ STEP {step_num} FAILED: {e}")
        return None

def test_pytorch_import():
    """Test 1: Import PyTorch"""
    print("Importing PyTorch...", flush=True)
    import torch
    print(f"PyTorch version: {torch.__version__}", flush=True)
    return torch

def test_cuda_availability(torch):
    """Test 2: Check CUDA"""
    print("Checking CUDA availability...", flush=True)
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}", flush=True)
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}", flush=True)
        device_count = torch.cuda.device_count()
        print(f"GPU count: {device_count}", flush=True)
        
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {name}", flush=True)
    
    return cuda_available

def test_cpu_tensor(torch):
    """Test 3: CPU tensor operations"""
    print("Creating CPU tensors...", flush=True)
    x = torch.randn(100, 100)
    y = torch.randn(100, 100)
    print(f"Tensor shapes: {x.shape}, {y.shape}", flush=True)
    
    print("CPU matrix multiplication...", flush=True)
    start = time.time()
    z = torch.matmul(x, y)
    elapsed = time.time() - start
    print(f"Result shape: {z.shape}", flush=True)
    print(f"Time: {elapsed*1000:.2f} ms", flush=True)
    
    return True

def test_gpu_tensor(torch, cuda_available):
    """Test 4: GPU tensor operations"""
    if not cuda_available:
        print("Skipping GPU test - CUDA not available", flush=True)
        return True
    
    print("Creating GPU tensors...", flush=True)
    x = torch.randn(100, 100, device='cuda')
    y = torch.randn(100, 100, device='cuda')
    print(f"GPU tensor shapes: {x.shape}, {y.shape}", flush=True)
    
    print("GPU matrix multiplication...", flush=True)
    start = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()  # Wait for completion
    elapsed = time.time() - start
    print(f"Result shape: {z.shape}", flush=True)
    print(f"Time: {elapsed*1000:.2f} ms", flush=True)
    
    # Cleanup
    print("Cleaning up GPU memory...", flush=True)
    del x, y, z
    torch.cuda.empty_cache()
    print("GPU memory cleared", flush=True)
    
    return True

def test_fft_operations(torch, cuda_available):
    """Test 5: FFT operations"""
    print("Creating test signal...", flush=True)
    signal = torch.randn(1000)
    print(f"Signal shape: {signal.shape}", flush=True)
    
    # CPU FFT
    print("CPU FFT...", flush=True)
    start = time.time()
    fft_cpu = torch.fft.rfft(signal)
    cpu_time = time.time() - start
    print(f"CPU FFT result: {fft_cpu.shape}, time: {cpu_time*1000:.2f} ms", flush=True)
    
    if cuda_available:
        # GPU FFT
        print("GPU FFT...", flush=True)
        signal_gpu = signal.cuda()
        start = time.time()
        fft_gpu = torch.fft.rfft(signal_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"GPU FFT result: {fft_gpu.shape}, time: {gpu_time*1000:.2f} ms", flush=True)
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"Speedup: {speedup:.2f}x", flush=True)
        
        # Cleanup
        del signal_gpu, fft_gpu
        torch.cuda.empty_cache()
    
    return True

def main():
    print("🚀 MINIMAL CUDA TEST STARTING")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Import PyTorch
    torch_module = step_with_output(1, "Import PyTorch", test_pytorch_import)
    if not torch_module:
        return
    
    time.sleep(2)  # Pause between steps
    
    # Step 2: Check CUDA
    cuda_available = step_with_output(2, "Check CUDA Availability", 
                                    lambda: test_cuda_availability(torch_module))
    
    time.sleep(2)
    
    # Step 3: CPU operations
    step_with_output(3, "CPU Tensor Operations", 
                    lambda: test_cpu_tensor(torch_module))
    
    time.sleep(2)
    
    # Step 4: GPU operations
    step_with_output(4, "GPU Tensor Operations", 
                    lambda: test_gpu_tensor(torch_module, cuda_available))
    
    time.sleep(2)
    
    # Step 5: FFT operations
    step_with_output(5, "FFT Operations", 
                    lambda: test_fft_operations(torch_module, cuda_available))
    
    print(f"\n{'='*60}")
    print("🎉 ALL TESTS COMPLETED!")
    print(f"End time: {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n⚠️ Test interrupted at {time.strftime('%H:%M:%S')}")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
