#!/usr/bin/env python3
"""
AMD GPU Test - Safe for RX 5700 XT with ROCm
"""
import sys
import time
import subprocess

def check_amd_gpu():
    """Check AMD GPU and ROCm setup"""
    print("🔍 AMD GPU Detection")
    print("=" * 50)
    
    # Check GPU hardware
    try:
        result = subprocess.run(['lspci', '|', 'grep', '-i', 'vga'], 
                              shell=True, capture_output=True, text=True)
        if 'AMD' in result.stdout or 'ATI' in result.stdout:
            print("✅ AMD GPU detected via lspci")
            print(f"   {result.stdout.strip()}")
        else:
            print("❌ No AMD GPU found in lspci")
    except:
        print("⚠️  Cannot run lspci")
    
    # Check ROCm
    try:
        result = subprocess.run(['rocm-smi', '-i'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ ROCm working - rocm-smi responds")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Device Name' in line:
                    print(f"   {line.strip()}")
        else:
            print("❌ ROCm not responding")
    except subprocess.TimeoutExpired:
        print("⏰ ROCm command timed out")
    except FileNotFoundError:
        print("❌ ROCm not installed (rocm-smi not found)")
    
    print()

def test_pytorch_amd():
    """Test PyTorch with AMD GPU"""
    print("🧪 PyTorch AMD GPU Test")
    print("=" * 50)
    
    try:
        print("Importing PyTorch...", flush=True)
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        # Check CUDA (which is ROCm on AMD)
        print("Checking 'CUDA' (actually ROCm) availability...", flush=True)
        cuda_available = torch.cuda.is_available()
        print(f"CUDA/ROCm available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"GPU devices: {device_count}")
            
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                print(f"Device {i}: {name}")
                
                # Check if it's actually ROCm
                if 'Radeon' in name or 'AMD' in name:
                    print("✅ Confirmed AMD GPU via PyTorch")
                    
                    # Check HIP version
                    if hasattr(torch.version, 'hip') and torch.version.hip:
                        print(f"HIP version: {torch.version.hip}")
                    else:
                        print("⚠️  HIP version not available")
        
        return torch, cuda_available
        
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return None, False

def test_cpu_operations(torch_module):
    """Test CPU operations first (safe)"""
    print("\n💻 CPU Operations Test")
    print("=" * 50)
    
    try:
        print("Creating CPU tensors...", flush=True)
        x = torch_module.randn(100, 100)
        y = torch_module.randn(100, 100)
        print(f"✅ CPU tensors: {x.shape}, {y.shape}")
        
        print("CPU matrix multiplication...", flush=True)
        start = time.time()
        z = torch_module.matmul(x, y)
        cpu_time = time.time() - start
        print(f"✅ CPU matmul: {z.shape}, time: {cpu_time*1000:.2f} ms")
        
        print("CPU FFT...", flush=True)
        signal = torch_module.randn(1000)
        start = time.time()
        fft_result = torch_module.fft.rfft(signal)
        cpu_fft_time = time.time() - start
        print(f"✅ CPU FFT: {signal.shape} -> {fft_result.shape}, time: {cpu_fft_time*1000:.2f} ms")
        
        return True
        
    except Exception as e:
        print(f"❌ CPU operations failed: {e}")
        return False

def test_amd_gpu_safe(torch_module, gpu_available):
    """Test AMD GPU operations safely"""
    print("\n🎮 AMD GPU Safe Test")
    print("=" * 50)
    
    if not gpu_available:
        print("⚠️  Skipping GPU test - not available")
        return
    
    try:
        # Test 1: Basic memory allocation
        print("Test 1: Basic GPU memory...", flush=True)
        tiny_tensor = torch_module.randn(10, 10, device='cuda')
        print(f"✅ GPU tensor created: {tiny_tensor.shape}")
        print(f"   Device: {tiny_tensor.device}")
        
        # Immediate cleanup
        del tiny_tensor
        torch_module.cuda.empty_cache()
        print("✅ Memory cleaned")
        
        # Test 2: Simple arithmetic (safe)
        print("\nTest 2: Simple GPU arithmetic...", flush=True)
        a = torch_module.randn(50, 50, device='cuda')
        b = torch_module.randn(50, 50, device='cuda')
        c = a + b  # Simple addition
        print(f"✅ GPU addition: {a.shape} + {b.shape} = {c.shape}")
        
        # Cleanup
        del a, b, c
        torch_module.cuda.empty_cache()
        print("✅ Memory cleaned")
        
        # Test 3: Very small matrix multiplication
        print("\nTest 3: Small GPU matrix multiplication...", flush=True)
        x = torch_module.randn(32, 32, device='cuda')
        y = torch_module.randn(32, 32, device='cuda')
        
        start = time.time()
        z = torch_module.matmul(x, y)
        torch_module.cuda.synchronize()  # Wait for completion
        gpu_time = time.time() - start
        
        print(f"✅ GPU matmul: {x.shape} @ {y.shape} = {z.shape}")
        print(f"   Time: {gpu_time*1000:.2f} ms")
        
        # Cleanup
        del x, y, z
        torch_module.cuda.empty_cache()
        print("✅ Memory cleaned")
        
        print("\n✅ All AMD GPU tests passed!")
        
    except Exception as e:
        print(f"❌ AMD GPU test failed: {e}")
        print("This is common - AMD GPU support varies")
        
        # Emergency cleanup
        try:
            torch_module.cuda.empty_cache()
        except:
            pass

def test_amd_fft_careful(torch_module, gpu_available):
    """Test FFT very carefully on AMD GPU"""
    print("\n🌊 AMD GPU FFT Test (Careful)")
    print("=" * 50)
    
    if not gpu_available:
        print("⚠️  Skipping FFT test - GPU not available")
        return
    
    try:
        # Start with CPU baseline
        print("CPU FFT baseline...", flush=True)
        signal_cpu = torch_module.randn(256)  # Very small signal
        fft_cpu = torch_module.fft.rfft(signal_cpu)
        print(f"✅ CPU FFT: {signal_cpu.shape} -> {fft_cpu.shape}")
        
        # Try GPU FFT with extreme caution
        print("Attempting GPU FFT (timeout in 10 seconds)...", flush=True)
        
        # Move to GPU
        signal_gpu = signal_cpu.cuda()
        print("✅ Signal moved to GPU")
        
        # Very careful FFT attempt
        import signal as sig
        
        def timeout_handler(signum, frame):
            raise TimeoutError("FFT timed out")
        
        # Set 10 second timeout
        old_handler = sig.signal(sig.SIGALRM, timeout_handler)
        sig.alarm(10)
        
        try:
            start = time.time()
            fft_gpu = torch_module.fft.rfft(signal_gpu)
            torch_module.cuda.synchronize()
            gpu_time = time.time() - start
            
            sig.alarm(0)  # Cancel timeout
            
            print(f"✅ GPU FFT successful: {signal_gpu.shape} -> {fft_gpu.shape}")
            print(f"   Time: {gpu_time*1000:.2f} ms")
            
            # Cleanup
            del signal_gpu, fft_gpu
            torch_module.cuda.empty_cache()
            
        except TimeoutError:
            print("⏰ GPU FFT timed out (common on some AMD setups)")
            try:
                torch_module.cuda.empty_cache()
            except:
                pass
        finally:
            sig.signal(sig.SIGALRM, old_handler)
        
    except Exception as e:
        print(f"⚠️  FFT test failed: {e}")
        print("This is expected on some AMD configurations")
        try:
            torch_module.cuda.empty_cache()
        except:
            pass

def main():
    print("🚀 AMD GPU TEST - RX 5700 XT + ROCm")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Step 1: Check hardware
    check_amd_gpu()
    time.sleep(1)
    
    # Step 2: Test PyTorch
    torch_module, gpu_available = test_pytorch_amd()
    if not torch_module:
        print("❌ Cannot continue without PyTorch")
        return
    time.sleep(1)
    
    # Step 3: CPU operations (safe baseline)
    cpu_ok = test_cpu_operations(torch_module)
    if not cpu_ok:
        print("❌ CPU operations failed - something is very wrong")
        return
    time.sleep(1)
    
    # Step 4: Safe GPU tests
    test_amd_gpu_safe(torch_module, gpu_available)
    time.sleep(1)
    
    # Step 5: Careful FFT test
    test_amd_fft_careful(torch_module, gpu_available)
    
    print("\n" + "=" * 60)
    print("🎉 AMD GPU TEST COMPLETED")
    print(f"End time: {time.strftime('%H:%M:%S')}")
    print("=" * 60)
    
    # Final cleanup
    if gpu_available:
        try:
            torch_module.cuda.empty_cache()
            print("✅ Final cleanup completed")
        except:
            pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n⚠️ Test interrupted at {time.strftime('%H:%M:%S')}")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
