#!/usr/bin/env python3
"""
GPU Diagnostics and Detection Script
Based on patterns from opennlp-gpu project
"""

import os
import sys
import subprocess
import torch

def run_command(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

def check_rocm_installation():
    """Check ROCm installation following OpenNLP-GPU approach"""
    print("\n🔍 Checking ROCm Installation")
    print("=" * 60)
    
    # Check rocm-smi
    success, output = run_command("rocm-smi --showproductname")
    if success:
        print("✅ ROCm Driver: Installed")
        print(f"   Output: {output.strip()[:100]}")
    else:
        print("❌ ROCm Driver: Not found or not working")
    
    # Check ROCm path
    rocm_paths = ["/opt/rocm", "/usr/local/rocm"]
    rocm_found = False
    for path in rocm_paths:
        if os.path.exists(path):
            print(f"✅ ROCm Path: {path}")
            rocm_found = True
            break
    
    if not rocm_found:
        print("❌ ROCm Path: Not found")
    
    # Check HIP compiler
    success, output = run_command("hipcc --version")
    if success:
        print("✅ HIP Compiler: Available")
    else:
        print("⚠️  HIP Compiler: Not found")
    
    return rocm_found

def check_pytorch_rocm():
    """Check PyTorch ROCm integration"""
    print("\n🔍 Checking PyTorch ROCm Integration")
    print("=" * 60)
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        
        # Get device properties
        props = torch.cuda.get_device_properties(0)
        print(f"Total Memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"Compute Capability: {props.major}.{props.minor}")
    else:
        print("⚠️  No CUDA/ROCm devices detected by PyTorch")
    
    # Check for ROCm-specific version info
    if hasattr(torch.version, 'hip'):
        print(f"✅ HIP Version: {torch.version.hip}")
    else:
        print("⚠️  HIP version not found in PyTorch")

def test_gpu_tensor_operations():
    """Test basic GPU tensor operations with safety"""
    print("\n🧪 Testing GPU Tensor Operations")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠️  Skipping - No GPU available")
        return False
    
    try:
        print("Creating small test tensor...")
        device = torch.device('cuda:0')
        
        # Very small test
        test_tensor = torch.randn(10, 10, device=device)
        result = test_tensor @ test_tensor.T
        
        print(f"✅ Basic tensor operations: Working")
        print(f"   Test result shape: {result.shape}")
        
        # Slightly larger test
        test_tensor2 = torch.randn(100, 100, device=device)
        result2 = test_tensor2 @ test_tensor2.T
        
        print(f"✅ Medium tensor operations: Working")
        print(f"   Test result shape: {result2.shape}")
        
        # Cleanup
        del test_tensor, result, test_tensor2, result2
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ GPU tensor operations failed: {e}")
        return False

def check_environment_variables():
    """Check GPU-related environment variables"""
    print("\n🔍 Checking Environment Variables")
    print("=" * 60)
    
    env_vars = [
        'ROCM_PATH',
        'HIP_PATH',
        'HSA_OVERRIDE_GFX_VERSION',
        'HIP_PLATFORM',
        'CUDA_VISIBLE_DEVICES',
        'LD_LIBRARY_PATH'
    ]
    
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}: {value[:50]}...")
        else:
            print(f"⭕ {var}: Not set")

def get_recommendations():
    """Provide recommendations based on OpenNLP-GPU patterns"""
    print("\n💡 Recommendations (from opennlp-gpu patterns)")
    print("=" * 60)
    
    print("""
For AMD RX 5700 XT (gfx1010) with ROCm 6.2:

1. Set environment variables (from opennlp-gpu):
   export ROCM_PATH=/opt/rocm
   export HIP_PATH=/opt/rocm
   export HSA_OVERRIDE_GFX_VERSION=10.3.0
   export HIP_PLATFORM=amd
   export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

2. For Python/PyTorch:
   - Use CPU training for stability (proven in opennlp-gpu)
   - Implement graceful GPU fallback
   - Test GPU operations before full training

3. Consider downgrading to ROCm 5.x if needed
   (RX 5700 XT has better support in ROCm 5.x)

4. Alternative: Use CPU with multiprocessing
   (opennlp-gpu shows this is viable for production)
""")

def main():
    print("🚀 EEG2025 GPU Diagnostics")
    print("=" * 60)
    print("Based on opennlp-gpu diagnostic patterns")
    print()
    
    # System info
    print("🖥️  System Information")
    print("=" * 60)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    
    # Run checks
    rocm_installed = check_rocm_installation()
    check_pytorch_rocm()
    check_environment_variables()
    gpu_works = test_gpu_tensor_operations()
    
    # Summary
    print("\n📊 Summary")
    print("=" * 60)
    
    if rocm_installed and torch.cuda.is_available() and gpu_works:
        print("✅ Status: GPU READY")
        print("   Your GPU appears to be working with PyTorch")
    elif rocm_installed and torch.cuda.is_available():
        print("⚠️  Status: GPU DETECTED BUT UNSTABLE")
        print("   GPU is detected but tensor operations may be unreliable")
        print("   Recommendation: Use CPU training with multiprocessing")
    else:
        print("❌ Status: CPU ONLY")
        print("   GPU not available or not working")
        print("   Recommendation: Use CPU training (proven stable)")
    
    get_recommendations()

if __name__ == "__main__":
    main()
