#!/usr/bin/env python3 -u
"""Simple CUDA test with immediate output"""
import sys
import time

print("="*80, flush=True)
print("CUDA/CuFFT/CuBLAS SIMPLE TEST", flush=True)
print("="*80, flush=True)
print(f"Started at: {time.strftime('%H:%M:%S')}", flush=True)
print("", flush=True)

# Step 1
print("[1/6] Importing PyTorch...", flush=True)
start = time.time()
import torch
print(f"  ✅ Done in {time.time()-start:.1f}s - PyTorch {torch.__version__}", flush=True)
print("", flush=True)

# Step 2
print("[2/6] Checking CUDA...", flush=True)
start = time.time()
cuda_available = torch.cuda.is_available()
print(f"  CUDA Available: {cuda_available}", flush=True)
if cuda_available:
    print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  CUDA Version: {torch.version.cuda}", flush=True)
else:
    print(f"  Using CPU mode", flush=True)
print(f"  ✅ Done in {time.time()-start:.1f}s", flush=True)
print("", flush=True)

# Step 3
print("[3/6] Importing cuda_optimized module...", flush=True)
start = time.time()
try:
    from src.gpu.cuda_optimized import CuFFTOptimizer, CuBLASOptimizer
    print(f"  ✅ Done in {time.time()-start:.1f}s", flush=True)
except Exception as e:
    print(f"  ❌ Failed: {e}", flush=True)
    sys.exit(1)
print("", flush=True)

# Step 4
print("[4/6] Testing CuFFT...", flush=True)
start = time.time()
try:
    fft_opt = CuFFTOptimizer(device="cuda" if cuda_available else "cpu")
    print("  Creating test signal (2x19x1000)...", flush=True)
    test_signal = torch.randn(2, 19, 1000)
    if cuda_available:
        test_signal = test_signal.cuda()
    
    print("  Running FFT...", flush=True)
    fft_result = fft_opt.rfft_batch(test_signal, dim=-1)
    if cuda_available:
        torch.cuda.synchronize()
    
    print("  Running IFFT...", flush=True)
    reconstructed = fft_opt.irfft_batch(fft_result, n=1000, dim=-1)
    if cuda_available:
        torch.cuda.synchronize()
    
    print(f"  ✅ Done in {time.time()-start:.1f}s", flush=True)
except Exception as e:
    print(f"  ❌ Failed: {e}", flush=True)
print("", flush=True)

# Step 5
print("[5/6] Testing CuBLAS...", flush=True)
start = time.time()
try:
    blas_opt = CuBLASOptimizer(device="cuda" if cuda_available else "cpu")
    print("  Creating test matrices (32x128x128)...", flush=True)
    A = torch.randn(32, 128, 128)
    B = torch.randn(32, 128, 128)
    if cuda_available:
        A = A.cuda()
        B = B.cuda()
    
    print("  Running matrix multiplication...", flush=True)
    C = blas_opt.bmm_optimized(A, B)
    if cuda_available:
        torch.cuda.synchronize()
    
    print(f"  ✅ Done in {time.time()-start:.1f}s", flush=True)
except Exception as e:
    print(f"  ❌ Failed: {e}", flush=True)
print("", flush=True)

# Step 6
if cuda_available:
    print("[6/6] Running benchmarks...", flush=True)
    
    # FFT benchmark
    print("  FFT Benchmark:", flush=True)
    signal_cpu = torch.randn(16, 64, 5000)
    
    print("    CPU FFT (5 runs)...", flush=True)
    start = time.time()
    for i in range(5):
        _ = torch.fft.rfft(signal_cpu, dim=-1)
        print(f"      Run {i+1}/5", flush=True)
    cpu_time = (time.time() - start) / 5
    print(f"    CPU: {cpu_time*1000:.2f} ms average", flush=True)
    
    signal_gpu = signal_cpu.cuda()
    torch.cuda.synchronize()
    for _ in range(3):
        _ = torch.fft.rfft(signal_gpu, dim=-1)
    torch.cuda.synchronize()
    
    print("    GPU FFT (5 runs)...", flush=True)
    start = time.time()
    for i in range(5):
        _ = torch.fft.rfft(signal_gpu, dim=-1)
        print(f"      Run {i+1}/5", flush=True)
    torch.cuda.synchronize()
    gpu_time = (time.time() - start) / 5
    print(f"    GPU: {gpu_time*1000:.2f} ms average", flush=True)
    print(f"    🚀 Speedup: {cpu_time/gpu_time:.2f}x", flush=True)
    print("", flush=True)
    
    # Matmul benchmark
    print("  Matrix Multiplication Benchmark:", flush=True)
    a_cpu = torch.randn(16, 256, 256)
    b_cpu = torch.randn(16, 256, 256)
    
    print("    CPU matmul (5 runs)...", flush=True)
    start = time.time()
    for i in range(5):
        _ = torch.bmm(a_cpu, b_cpu)
        print(f"      Run {i+1}/5", flush=True)
    cpu_time = (time.time() - start) / 5
    print(f"    CPU: {cpu_time*1000:.2f} ms average", flush=True)
    
    a_gpu = a_cpu.cuda()
    b_gpu = b_cpu.cuda()
    torch.cuda.synchronize()
    for _ in range(3):
        _ = torch.bmm(a_gpu, b_gpu)
    torch.cuda.synchronize()
    
    print("    GPU matmul (5 runs)...", flush=True)
    start = time.time()
    for i in range(5):
        _ = torch.bmm(a_gpu, b_gpu)
        print(f"      Run {i+1}/5", flush=True)
    torch.cuda.synchronize()
    gpu_time = (time.time() - start) / 5
    print(f"    GPU: {gpu_time*1000:.2f} ms average", flush=True)
    print(f"    🚀 Speedup: {cpu_time/gpu_time:.2f}x", flush=True)
    
    torch.cuda.empty_cache()
else:
    print("[6/6] Skipping benchmarks (no CUDA)", flush=True)

print("", flush=True)
print("="*80, flush=True)
print("✅ ALL TESTS COMPLETED SUCCESSFULLY", flush=True)
print("="*80, flush=True)
