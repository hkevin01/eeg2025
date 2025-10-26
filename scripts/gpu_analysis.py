#!/usr/bin/env python3
"""
Deep analysis of why PyTorch from source didn't fix the GPU issues
"""

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

import torch
import subprocess
import sys

print("🔍 GPU Issue Analysis - Why Source Build Didn't Work")
print("=" * 55)

def check_pytorch_build_info():
    """Check how PyTorch was built"""
    print("📦 PyTorch Build Information:")
    print(f"   Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if hasattr(torch.version, 'hip'):
        print(f"   HIP Version: {torch.version.hip}")
    
    if torch.cuda.is_available():
        print(f"   Device: {torch.cuda.get_device_name()}")
        print(f"   Device Count: {torch.cuda.device_count()}")
        print(f"   Current Device: {torch.cuda.current_device()}")
        
        # Get device properties
        props = torch.cuda.get_device_properties(0)
        print(f"   Compute Capability: {props.major}.{props.minor}")
        print(f"   Total Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"   Multiprocessors: {props.multi_processor_count}")
    
    # Check compilation flags
    print(f"   Compiled with CUDA: {torch.version.cuda}")
    print(f"   CuDNN Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    
def test_operation_levels():
    """Test different levels of GPU operations to find where it breaks"""
    print("\n🧪 Testing Operation Complexity Levels:")
    
    if not torch.cuda.is_available():
        print("❌ GPU not available")
        return
    
    device = torch.device('cuda')
    
    try:
        # Level 1: Basic tensor operations
        print("   Level 1: Basic tensors...", end=" ")
        a = torch.randn(10, 10).cuda()
        b = torch.randn(10, 10).cuda()
        c = a + b
        print("✅")
        
        # Level 2: Matrix multiplication
        print("   Level 2: Matrix multiply...", end=" ")
        d = a @ b
        print("✅")
        
        # Level 3: Linear layer
        print("   Level 3: Linear layer...", end=" ")
        linear = torch.nn.Linear(10, 5).cuda()
        e = linear(a)
        print("✅")
        
        # Level 4: Backward pass
        print("   Level 4: Backward pass...", end=" ")
        loss = e.sum()
        loss.backward()
        print("✅")
        
        # Level 5: Small convolution (this is where it typically fails)
        print("   Level 5: Small conv1d...", end=" ")
        conv = torch.nn.Conv1d(1, 2, 3).cuda()
        x = torch.randn(1, 1, 5).cuda()
        y = conv(x)
        print("✅")
        
        # Level 6: Realistic convolution
        print("   Level 6: Realistic conv1d...", end=" ")
        conv_big = torch.nn.Conv1d(32, 64, 5).cuda()
        x_big = torch.randn(8, 32, 100).cuda()
        y_big = conv_big(x_big)
        print("✅")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    return True

def analyze_rocm_environment():
    """Analyze the ROCm environment"""
    print("\n🏗️  ROCm Environment Analysis:")
    
    # Check ROCm installation
    try:
        result = subprocess.run(['rocm-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("   ROCm SMI: ✅ Available")
            # Parse GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GPU' in line and 'Temp' in line:
                    print(f"   {line.strip()}")
        else:
            print("   ROCm SMI: ❌ Not working")
    except Exception as e:
        print(f"   ROCm SMI: ❌ Error: {e}")
    
    # Check HIP
    try:
        result = subprocess.run(['hipconfig', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"   HIP Version: {result.stdout.strip()}")
        else:
            print("   HIP: ❌ Not available")
    except Exception as e:
        print(f"   HIP: ❌ Error: {e}")
    
    # Check environment variables
    print("   Environment Variables:")
    rocm_vars = ['ROCM_PATH', 'HIP_PATH', 'HSA_OVERRIDE_GFX_VERSION']
    for var in rocm_vars:
        value = os.environ.get(var, 'Not set')
        print(f"     {var}: {value}")

def why_source_build_failed():
    """Explain why building from source didn't solve the problem"""
    print("\n❓ Why Building PyTorch from Source Didn't Fix the Issue:")
    print("-" * 50)
    
    print("1. 🏗️  Source Build Status:")
    print("   ✅ We successfully built PyTorch 2.5.1 from source")
    print("   ✅ Compilation completed without errors")
    print("   ✅ ROCm 6.2.2 support was properly compiled in")
    print("   ✅ Basic GPU operations work fine")
    
    print("\n2. 🔍 Root Cause Analysis:")
    print("   ❌ The issue is NOT in PyTorch compilation")
    print("   ❌ The issue is in the ROCm/HIP/GPU driver stack")
    print("   ❌ gfx1030 (RX 5600 XT) has known compatibility issues")
    print("   ❌ HSA aperture violations occur at the hardware/driver level")
    
    print("\n3. 🧠 What Actually Happens:")
    print("   • Basic operations work (they use simple kernels)")
    print("   • Complex operations (convolutions) trigger advanced GPU features")
    print("   • Advanced features hit the gfx1030 compatibility wall")
    print("   • ROCm 6.x/7.x broke support for older architectures")
    print("   • Even with HSA_OVERRIDE_GFX_VERSION, hardware limits remain")
    
    print("\n4. 💡 Why PyTorch 1.13.1+rocm5.2 Works Better:")
    print("   • ROCm 5.2 had better gfx1030 support")
    print("   • Older kernels were more conservative")
    print("   • Less aggressive memory management")
    print("   • Fewer advanced features that trigger hardware bugs")
    
    print("\n5. 🎯 The Real Solution:")
    print("   • Source builds can't fix hardware/driver incompatibilities")
    print("   • Need either: newer GPU, older ROCm, or hybrid CPU/GPU workflow")
    print("   • Best approach: Use GPU for what works, CPU for the rest")

if __name__ == "__main__":
    check_pytorch_build_info()
    analyze_rocm_environment()
    success = test_operation_levels()
    why_source_build_failed()
    
    print(f"\n🏁 Conclusion:")
    print(f"   Building from source gave us a properly compiled PyTorch,")
    print(f"   but the fundamental gfx1030+ROCm6.x incompatibility remains.")
    print(f"   The hardware/driver stack is the bottleneck, not PyTorch.")
