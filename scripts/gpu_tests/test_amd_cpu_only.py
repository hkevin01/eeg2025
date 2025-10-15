#!/usr/bin/env python3
"""
AMD-Safe Test - CPU Only with GPU Detection
NO GPU OPERATIONS - Detection Only
"""
import sys
import time
import os
import subprocess

def print_safe(message):
    """Safe print with immediate flush"""
    print(message, flush=True)
    time.sleep(0.1)  # Brief pause to ensure output

def check_system_info():
    """Check system without any GPU operations"""
    print_safe("\n" + "="*60)
    print_safe("SYSTEM INFORMATION CHECK (NO GPU OPERATIONS)")
    print_safe("="*60)
    
    # Check GPU hardware
    print_safe("\n1. GPU Hardware Detection:")
    try:
        result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=10)
        gpu_lines = [line for line in result.stdout.split('\n') if 'VGA' in line or 'Display' in line]
        for line in gpu_lines:
            print_safe(f"   {line}")
    except Exception as e:
        print_safe(f"   Could not detect GPU hardware: {e}")
    
    # Check ROCm installation
    print_safe("\n2. ROCm Installation Check:")
    rocm_paths = [
        "/opt/rocm",
        "/usr/bin/rocm-smi",
        "/usr/bin/rocminfo"
    ]
    
    for path in rocm_paths:
        if os.path.exists(path):
            print_safe(f"   ✅ Found: {path}")
        else:
            print_safe(f"   ❌ Missing: {path}")
    
    # Check PyTorch installation
    print_safe("\n3. PyTorch Installation Check:")
    try:
        import torch
        print_safe(f"   ✅ PyTorch version: {torch.__version__}")
        
        # Check build info without GPU operations
        if hasattr(torch.version, 'cuda') and torch.version.cuda:
            print_safe(f"   CUDA version in build: {torch.version.cuda}")
        
        if hasattr(torch.version, 'hip') and torch.version.hip:
            print_safe(f"   ✅ HIP/ROCm version: {torch.version.hip}")
        else:
            print_safe("   ❌ No HIP/ROCm support detected in PyTorch")
            
    except ImportError as e:
        print_safe(f"   ❌ PyTorch not available: {e}")
        return None
        
    return torch

def test_cpu_only_operations(torch_module):
    """Test CPU operations only - NO GPU"""
    print_safe("\n4. CPU-Only PyTorch Operations:")
    
    if not torch_module:
        print_safe("   Skipping - PyTorch not available")
        return
    
    try:
        # Small CPU tensor operations
        print_safe("   Creating CPU tensors...")
        a = torch_module.randn(100, 100)
        b = torch_module.randn(100, 100)
        print_safe(f"   ✅ Tensors created: {a.shape}, {b.shape}")
        
        # Matrix multiplication
        print_safe("   Testing CPU matrix multiplication...")
        start = time.time()
        c = torch_module.matmul(a, b)
        elapsed = time.time() - start
        print_safe(f"   ✅ CPU matmul: {elapsed*1000:.2f} ms")
        
        # FFT operations
        print_safe("   Testing CPU FFT...")
        signal = torch_module.randn(1000)
        start = time.time()
        fft_result = torch_module.fft.rfft(signal)
        elapsed = time.time() - start
        print_safe(f"   ✅ CPU FFT: {signal.shape} -> {fft_result.shape}, {elapsed*1000:.2f} ms")
        
        print_safe("   ✅ All CPU operations successful")
        
    except Exception as e:
        print_safe(f"   ❌ CPU operations failed: {e}")

def check_rocm_status():
    """Check ROCm status without GPU operations"""
    print_safe("\n5. ROCm Status Check (No GPU Operations):")
    
    # Check rocm-smi
    try:
        print_safe("   Checking rocm-smi...")
        result = subprocess.run(['rocm-smi', '--showid'], 
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print_safe("   ✅ rocm-smi accessible")
            lines = result.stdout.strip().split('\n')
            for line in lines[:5]:  # First 5 lines only
                if line.strip():
                    print_safe(f"     {line}")
        else:
            print_safe("   ❌ rocm-smi not working")
    except Exception as e:
        print_safe(f"   ⚠️  rocm-smi check failed: {e}")
    
    # Check environment variables
    print_safe("\n   ROCm Environment Variables:")
    rocm_vars = ['HIP_VISIBLE_DEVICES', 'ROCM_PATH', 'HIP_PATH']
    for var in rocm_vars:
        value = os.environ.get(var, 'Not set')
        print_safe(f"     {var}: {value}")

def safe_gpu_detection_only():
    """ONLY detect GPU, no operations"""
    print_safe("\n6. Safe GPU Detection (NO OPERATIONS):")
    
    try:
        import torch
        
        # Just check availability - no operations
        print_safe("   Checking torch.cuda.is_available()...")
        cuda_available = torch.cuda.is_available()
        print_safe(f"   CUDA available: {cuda_available}")
        
        if cuda_available:
            print_safe("   ⚠️  CUDA detected but we're on AMD - this may be wrong")
            
            # Check device count - this is usually safe
            try:
                device_count = torch.cuda.device_count()
                print_safe(f"   Device count: {device_count}")
            except Exception as e:
                print_safe(f"   Device count check failed: {e}")
        
        print_safe("\n   ✅ GPU detection completed (no operations performed)")
        
    except Exception as e:
        print_safe(f"   ❌ GPU detection failed: {e}")

def main():
    print_safe("🛡️  AMD-SAFE SYSTEM CHECK")
    print_safe("NO GPU OPERATIONS WILL BE PERFORMED")
    print_safe(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # System info
    torch_module = check_system_info()
    
    # CPU operations only
    test_cpu_only_operations(torch_module)
    
    # ROCm status
    check_rocm_status()
    
    # Safe GPU detection
    safe_gpu_detection_only()
    
    print_safe("\n" + "="*60)
    print_safe("✅ SAFE CHECK COMPLETED")
    print_safe("No GPU operations were performed")
    print_safe(f"End time: {time.strftime('%H:%M:%S')}")
    print_safe("="*60)
    
    # Recommendations
    print_safe("\n📋 RECOMMENDATIONS:")
    print_safe("1. Your AMD RX 5700 XT requires PyTorch with ROCm support")
    print_safe("2. Install PyTorch ROCm version:")
    print_safe("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7")
    print_safe("3. Avoid GPU operations until proper ROCm PyTorch is installed")
    print_safe("4. Current PyTorch may have CUDA support but not ROCm")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_safe(f"\n⚠️ Interrupted at {time.strftime('%H:%M:%S')}")
    except Exception as e:
        print_safe(f"\n❌ Error: {e}")
        print_safe("This script performs NO GPU operations and should be safe")
