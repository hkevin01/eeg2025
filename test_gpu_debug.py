#!/usr/bin/env python3
"""
GPU Debug Test - Verbose output for troubleshooting
"""
import sys
import os
import time

print("="*70)
print("GPU DEBUG TEST - Verbose Mode")
print("="*70)
print()

# Step 1: Environment Check
print("STEP 1: Checking environment variables...")
gpu_env_vars = [
    'HSA_OVERRIDE_GFX_VERSION',
    'HIP_VISIBLE_DEVICES',
    'ROCR_VISIBLE_DEVICES',
    'PYTORCH_HIP_ALLOC_CONF',
    'HSA_ENABLE_SDMA'
]

for var in gpu_env_vars:
    value = os.environ.get(var, 'NOT SET')
    print(f"   {var}: {value}")
print()

# Step 2: Python and System Info
print("STEP 2: Python and system info...")
print(f"   Python version: {sys.version.split()[0]}")
print(f"   Python executable: {sys.executable}")
print()

# Step 3: Import PyTorch
print("STEP 3: Importing PyTorch...")
try:
    import torch
    print(f"   ✅ PyTorch imported successfully")
    print(f"   Version: {torch.__version__}")
except Exception as e:
    print(f"   ❌ Failed to import PyTorch: {e}")
    sys.exit(1)
print()

# Step 4: Check CUDA/ROCm availability
print("STEP 4: Checking GPU availability...")
try:
    is_available = torch.cuda.is_available()
    print(f"   CUDA/ROCm available: {is_available}")
    
    if is_available:
        print(f"   Device count: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
        print(f"   Device name: {torch.cuda.get_device_name(0)}")
        
        # Check for HIP
        if hasattr(torch.version, 'hip') and torch.version.hip:
            print(f"   Platform: AMD ROCm/HIP")
            print(f"   HIP version: {torch.version.hip}")
        else:
            print(f"   Platform: NVIDIA CUDA")
            print(f"   CUDA version: {torch.version.cuda}")
    else:
        print("   ⚠️  No GPU available - CPU only mode")
except Exception as e:
    print(f"   ❌ Error checking GPU: {e}")
    import traceback
    traceback.print_exc()
print()

# Step 5: Import unified module
print("STEP 5: Importing unified GPU module...")
sys.path.insert(0, '/home/kevin/Projects/eeg2025/src')
try:
    from gpu.unified_gpu_optimized import GPUPlatformDetector
    print(f"   ✅ Module imported successfully")
except Exception as e:
    print(f"   ❌ Failed to import module: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Step 6: Platform detection
print("STEP 6: Running platform detection...")
try:
    detector = GPUPlatformDetector()
    print(f"   ✅ Platform detector created")
    
    info = detector.get_info()
    print(f"   GPU Available: {info['gpu_available']}")
    if info['gpu_available']:
        print(f"   Vendor: {info['vendor']}")
        print(f"   Platform: {info['platform']}")
        print(f"   Version: {info['version']}")
        print(f"   Device: {info['device_name']}")
except Exception as e:
    print(f"   ❌ Platform detection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

if not detector.gpu_available:
    print("No GPU available - stopping here")
    sys.exit(0)

# Step 7: Simple tensor operations
print("STEP 7: Testing basic tensor operations...")
try:
    print("   Creating CPU tensor...")
    x_cpu = torch.randn(10, 10)
    print(f"   ✅ CPU tensor created: shape {x_cpu.shape}")
    
    print("   Transferring to GPU...")
    x_gpu = x_cpu.cuda()
    print(f"   ✅ GPU tensor created: device {x_gpu.device}")
    
    print("   Simple operation (multiply by 2)...")
    y_gpu = x_gpu * 2
    print(f"   ✅ Operation completed")
    
    print("   Transferring back to CPU...")
    y_cpu = y_gpu.cpu()
    print(f"   ✅ Transfer successful")
    
    print("   Cleaning up GPU memory...")
    del x_gpu, y_gpu
    torch.cuda.empty_cache()
    print(f"   ✅ Cleanup done")
    
except Exception as e:
    print(f"   ❌ Tensor operations failed: {e}")
    import traceback
    traceback.print_exc()
    torch.cuda.empty_cache()
print()

# Step 8: Test unified FFT (CAREFUL - this can hang on AMD)
print("STEP 8: Testing unified FFT optimizer...")
print("   ⚠️  This step may cause issues on AMD - using SMALL data")
try:
    from gpu.unified_gpu_optimized import UnifiedFFTOptimizer
    
    print("   Creating FFT optimizer...")
    fft_opt = UnifiedFFTOptimizer()
    print(f"   ✅ FFT optimizer created")
    
    print("   Creating small test signal (2, 4, 100)...")
    signal = torch.randn(2, 4, 100)  # VERY small for safety
    print(f"   ✅ Signal created")
    
    print("   Computing FFT...")
    import signal as sig
    
    def timeout_handler(signum, frame):
        raise TimeoutError("FFT operation timed out")
    
    sig.signal(sig.SIGALRM, timeout_handler)
    sig.alarm(5)  # 5 second timeout
    
    try:
        freq = fft_opt.rfft_batch(signal, dim=-1)
        sig.alarm(0)  # Cancel timeout
        print(f"   ✅ FFT completed: shape {freq.shape}")
        
        print("   Computing inverse FFT...")
        sig.alarm(5)
        reconstructed = fft_opt.irfft_batch(freq, n=100, dim=-1)
        sig.alarm(0)
        print(f"   ✅ iFFT completed: shape {reconstructed.shape}")
        
        error = torch.mean(torch.abs(signal - reconstructed)).item()
        print(f"   Reconstruction error: {error:.6f}")
        
    except TimeoutError:
        sig.alarm(0)
        print(f"   ⚠️  FFT timed out after 5s - this is a known AMD issue")
    
    print("   Cleaning up...")
    torch.cuda.empty_cache()
    print(f"   ✅ Cleanup done")
    
except Exception as e:
    print(f"   ❌ FFT test failed: {e}")
    import traceback
    traceback.print_exc()
    torch.cuda.empty_cache()
print()

# Step 9: Test BLAS operations
print("STEP 9: Testing unified BLAS optimizer...")
try:
    from gpu.unified_gpu_optimized import UnifiedBLASOptimizer
    
    print("   Creating BLAS optimizer...")
    blas_opt = UnifiedBLASOptimizer()
    print(f"   ✅ BLAS optimizer created")
    
    print("   Creating small matrices (2, 8, 8)...")
    A = torch.randn(2, 8, 8)
    B = torch.randn(2, 8, 8)
    print(f"   ✅ Matrices created")
    
    print("   Computing batch matrix multiply...")
    C = blas_opt.bmm_optimized(A, B)
    print(f"   ✅ BMM completed: shape {C.shape}")
    
    print("   Cleaning up...")
    torch.cuda.empty_cache()
    print(f"   ✅ Cleanup done")
    
except Exception as e:
    print(f"   ❌ BLAS test failed: {e}")
    import traceback
    traceback.print_exc()
    torch.cuda.empty_cache()
print()

# Final cleanup
print("="*70)
print("FINAL CLEANUP")
print("="*70)
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**2
    cached = torch.cuda.memory_reserved() / 1024**2
    print(f"GPU Memory - Allocated: {allocated:.1f}MB, Cached: {cached:.1f}MB")
    torch.cuda.empty_cache()
    print("✅ GPU memory cleared")

print()
print("="*70)
print("✅ DEBUG TEST COMPLETED")
print("="*70)
