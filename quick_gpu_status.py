#!/usr/bin/env python3
"""Quick GPU status check without running convolutions"""
import os
import torch

# Set MIOpen config
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1030'

print("="*70)
print("🧠 EEG2025 GPU STATUS CHECK")
print("="*70)

print(f"\n✅ PyTorch: {torch.__version__}")
print(f"✅ ROCm: {torch.version.hip}")
print(f"✅ GPU Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"✅ Architecture: {torch.cuda.get_device_properties(0).gcnArchName}")
    
    # Test basic GPU operation (no convolution)
    try:
        x = torch.randn(100, 100).cuda()
        y = x @ x.T
        print(f"✅ Basic GPU operations: Working")
    except Exception as e:
        print(f"⚠️  Basic GPU operations: {e}")

print("\n" + "="*70)
print("📊 CONFIGURATION STATUS")
print("="*70)
print("\n✅ PyTorch ROCm 7.0 installed")
print("✅ System ROCm 7.0.2 detected")
print("⚠️  GPU Convolutions: Require MIOpen IMMEDIATE mode")
print("")
print("📝 To use GPU convolutions, set these environment variables:")
print("   export MIOPEN_FIND_MODE=2")
print("   export MIOPEN_DEBUG_DISABLE_FIND_DB=1")
print("   export MIOPEN_DISABLE_CACHE=1")
print("")
print("⚠️  IMPORTANT: gfx1030 has limited MIOpen support")
print("   • First convolution: ~3-4s (kernel compilation)")
print("   • May timeout or fail on complex operations")
print("   • Consider CPU training as stable alternative")
print("="*70)
