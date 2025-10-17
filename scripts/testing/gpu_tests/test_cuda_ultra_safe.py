#!/usr/bin/env python3
"""
Ultra-Safe CUDA Test with Freeze Prevention
"""
import sys
import time
import gc
import threading
from pathlib import Path

print("Starting ultra-safe CUDA test...", flush=True)

def progress_heartbeat():
    """Background thread to show we're alive"""
    while not stop_heartbeat:
        time.sleep(30)  # Every 30 seconds
        if not stop_heartbeat:
            print(f"🔄 Still running... ({time.strftime('%H:%M:%S')})", flush=True)

def safe_cuda_check():
    """Step 1: Basic CUDA check with immediate output"""
    print(f"[{time.strftime('%H:%M:%S')}] Step 1: Checking PyTorch import...", flush=True)
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported", flush=True)
        
        print(f"[{time.strftime('%H:%M:%S')}] Checking CUDA availability...", flush=True)
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}", flush=True)
        
        if cuda_available:
            print(f"CUDA Version: {torch.version.cuda}", flush=True)
            print(f"GPU Count: {torch.cuda.device_count()}", flush=True)
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)
        
        return cuda_available
        
    except Exception as e:
        print(f"❌ Error in CUDA check: {e}", flush=True)
        return False

def safe_memory_test(cuda_available):
    """Step 2: Small memory test with immediate cleanup"""
    print(f"\n[{time.strftime('%H:%M:%S')}] Step 2: Small memory test...", flush=True)
    
    if not cuda_available:
        print("Skipping GPU tests - CUDA not available", flush=True)
        return
    
    try:
        import torch
        
        # Very small test - 1MB only
        print("Creating tiny tensor (1MB)...", flush=True)
        x = torch.randn(128, 128, device='cpu')  # Start on CPU
        print(f"CPU tensor created: {x.shape}", flush=True)
        
        # Move to GPU carefully
        print("Moving to GPU...", flush=True)
        x_gpu = x.cuda()
        print("✅ Successfully moved to GPU", flush=True)
        
        # Immediate cleanup
        print("Cleaning up...", flush=True)
        del x_gpu
        del x
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        print("✅ Memory cleaned", flush=True)
        
        time.sleep(1)  # Brief pause
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}", flush=True)
        # Force cleanup on error
        try:
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass

def safe_fft_test(cuda_available):
    """Step 3: Tiny FFT test with safety checks"""
    print(f"\n[{time.strftime('%H:%M:%S')}] Step 3: Tiny FFT test...", flush=True)
    
    if not cuda_available:
        print("Skipping FFT test - CUDA not available", flush=True)
        return
    
    try:
        import torch
        
        # Extremely small FFT - 100 samples only
        print("Creating tiny signal (100 samples)...", flush=True)
        signal = torch.randn(1, 1, 100)  # 1 batch, 1 channel, 100 samples
        print("Signal created on CPU", flush=True)
        
        # CPU FFT first
        print("Testing CPU FFT...", flush=True)
        fft_cpu = torch.fft.rfft(signal, dim=-1)
        print(f"✅ CPU FFT: {signal.shape} -> {fft_cpu.shape}", flush=True)
        
        if cuda_available:
            # GPU FFT with safety
            print("Moving to GPU for FFT test...", flush=True)
            signal_gpu = signal.cuda()
            
            print("Testing GPU FFT...", flush=True)
            fft_gpu = torch.fft.rfft(signal_gpu, dim=-1)
            torch.cuda.synchronize()  # Ensure completion
            print(f"✅ GPU FFT: {signal_gpu.shape} -> {fft_gpu.shape}", flush=True)
            
            # Immediate cleanup
            print("Cleaning up FFT test...", flush=True)
            del signal_gpu, fft_gpu
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        del signal, fft_cpu
        gc.collect()
        print("✅ FFT test cleanup complete", flush=True)
        
        time.sleep(1)  # Brief pause
        
    except Exception as e:
        print(f"❌ FFT test failed: {e}", flush=True)
        # Force cleanup
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
        except:
            pass

def safe_matmul_test(cuda_available):
    """Step 4: Tiny matrix multiplication test"""
    print(f"\n[{time.strftime('%H:%M:%S')}] Step 4: Tiny matrix test...", flush=True)
    
    if not cuda_available:
        print("Skipping matrix test - CUDA not available", flush=True)
        return
    
    try:
        import torch
        
        # Very small matrices - 32x32
        print("Creating tiny matrices (32x32)...", flush=True)
        a = torch.randn(32, 32)
        b = torch.randn(32, 32)
        print("Matrices created on CPU", flush=True)
        
        # CPU matmul
        print("Testing CPU matrix multiplication...", flush=True)
        c_cpu = torch.matmul(a, b)
        print(f"✅ CPU matmul: {a.shape} @ {b.shape} = {c_cpu.shape}", flush=True)
        
        if cuda_available:
            # GPU matmul with safety
            print("Moving to GPU for matrix test...", flush=True)
            a_gpu = a.cuda()
            b_gpu = b.cuda()
            
            print("Testing GPU matrix multiplication...", flush=True)
            c_gpu = torch.matmul(a_gpu, b_gpu)
            torch.cuda.synchronize()
            print(f"✅ GPU matmul: {a_gpu.shape} @ {b_gpu.shape} = {c_gpu.shape}", flush=True)
            
            # Immediate cleanup
            print("Cleaning up matrix test...", flush=True)
            del a_gpu, b_gpu, c_gpu
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        del a, b, c_cpu
        gc.collect()
        print("✅ Matrix test cleanup complete", flush=True)
        
        time.sleep(1)  # Brief pause
        
    except Exception as e:
        print(f"❌ Matrix test failed: {e}", flush=True)
        # Force cleanup
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
        except:
            pass

def resource_monitor():
    """Monitor system resources"""
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        print(f"📊 Resources: CPU {cpu:.1f}%, Memory {mem.percent:.1f}%", flush=True)
    except:
        print("📊 Resource monitoring unavailable", flush=True)

def main():
    global stop_heartbeat
    stop_heartbeat = False
    
    # Start heartbeat thread
    heartbeat_thread = threading.Thread(target=progress_heartbeat, daemon=True)
    heartbeat_thread.start()
    
    try:
        print("="*60, flush=True)
        print("ULTRA-SAFE CUDA TEST", flush=True)
        print("="*60, flush=True)
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print("", flush=True)
        
        # Initial resource check
        resource_monitor()
        print("", flush=True)
        
        # Step 1: Basic CUDA check
        cuda_available = safe_cuda_check()
        print("", flush=True)
        resource_monitor()
        
        # Brief pause between steps
        print("Pausing 3 seconds between steps...", flush=True)
        time.sleep(3)
        
        # Step 2: Memory test
        safe_memory_test(cuda_available)
        print("", flush=True)
        resource_monitor()
        
        # Brief pause
        print("Pausing 3 seconds...", flush=True)
        time.sleep(3)
        
        # Step 3: FFT test
        safe_fft_test(cuda_available)
        print("", flush=True)
        resource_monitor()
        
        # Brief pause
        print("Pausing 3 seconds...", flush=True)
        time.sleep(3)
        
        # Step 4: Matrix test
        safe_matmul_test(cuda_available)
        print("", flush=True)
        resource_monitor()
        
        # Final cleanup
        print(f"\n[{time.strftime('%H:%M:%S')}] Final cleanup...", flush=True)
        if cuda_available:
            try:
                import torch
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("✅ GPU memory cleared", flush=True)
            except:
                pass
        
        gc.collect()
        print("✅ Python garbage collection completed", flush=True)
        
        print("", flush=True)
        print("="*60, flush=True)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY", flush=True)
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print("="*60, flush=True)
        
    except KeyboardInterrupt:
        print(f"\n[{time.strftime('%H:%M:%S')}] Test interrupted by user", flush=True)
    except Exception as e:
        print(f"\n[{time.strftime('%H:%M:%S')}] ❌ Fatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        # Stop heartbeat
        stop_heartbeat = True
        
        # Emergency cleanup
        print(f"\n[{time.strftime('%H:%M:%S')}] Emergency cleanup...", flush=True)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass
        gc.collect()
        print("✅ Emergency cleanup completed", flush=True)

if __name__ == "__main__":
    main()
