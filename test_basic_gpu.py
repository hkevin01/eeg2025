"""Test very basic GPU operations on gfx1030"""
import torch
import os

# Try with minimal ROCm environment  
os.environ.pop('HSA_OVERRIDE_GFX_VERSION', None)
os.environ.pop('PYTORCH_ROCM_ARCH', None)

print("Testing basic GPU operations")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # Test 1: Simple tensor operations
    print("\n1. Testing tensor creation...")
    x = torch.randn(10, 10, device='cuda')
    print(f"  ✅ Created tensor on GPU: {x.shape}")
    
    # Test 2: Matrix multiplication
    print("\n2. Testing matrix multiplication...")
    a = torch.randn(100, 100, device='cuda')
    b = torch.randn(100, 100, device='cuda')
    c = torch.matmul(a, b)
    print(f"  ✅ Matrix mul works: {c.shape}")
    
    # Test 3: Simple 1D convolution
    print("\n3. Testing 1D convolution...")
    import torch.nn as nn
    conv1d = nn.Conv1d(16, 32, kernel_size=3).cuda()
    x = torch.randn(2, 16, 100, device='cuda')
    y = conv1d(x)
    print(f"  ✅ Conv1D works: {x.shape} -> {y.shape}")
    
    # Test 4: 2D convolution (small kernel)
    print("\n4. Testing 2D convolution (small kernel)...")
    conv2d_small = nn.Conv2d(16, 32, kernel_size=3).cuda()
    x = torch.randn(2, 16, 32, 32, device='cuda')
    y = conv2d_small(x)
    print(f"  ✅ Conv2D small works: {x.shape} -> {y.shape}")
    
    # Test 5: 2D convolution (medium kernel)
    print("\n5. Testing 2D convolution (medium kernel 7x7)...")
    conv2d_med = nn.Conv2d(16, 32, kernel_size=7).cuda()
    x = torch.randn(2, 16, 64, 64, device='cuda')
    y = conv2d_med(x)
    print(f"  ✅ Conv2D medium works: {x.shape} -> {y.shape}")
    
    print("\n✅ All basic tests passed!")
else:
    print("CUDA not available")
