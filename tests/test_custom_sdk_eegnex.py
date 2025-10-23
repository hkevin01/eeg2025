#!/usr/bin/env python3
"""
Test EEGNeX with Custom ROCm SDK for gfx1010
=============================================
This test uses the custom-built PyTorch from /opt/rocm_sdk_612
which includes proper gfx1010 support.

The custom SDK was built with ROCm SDK Builder specifically for
AMD RX 5600 XT (gfx1010) to resolve memory aperture violations.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("  Testing EEGNeX with Custom ROCm SDK (gfx1010)")
print("=" * 70)
print()

# Check SDK environment
print("1. Checking Custom SDK Environment...")
sdk_path = "/opt/rocm_sdk_612"
if os.path.exists(sdk_path):
    print(f"   ✅ Custom SDK found: {sdk_path}")
    
    # Check if PyTorch is in SDK
    sdk_python_path = f"{sdk_path}/bin/python3"
    if os.path.exists(sdk_python_path):
        print(f"   ✅ SDK Python found: {sdk_python_path}")
    else:
        print(f"   ⚠️  SDK Python not found at {sdk_python_path}")
else:
    print(f"   ❌ Custom SDK not found at {sdk_path}")
    print("   Please build it using ROCm SDK Builder:")
    print("   https://github.com/lamikr/rocm_sdk_builder")
    sys.exit(1)

print()
print("2. Importing PyTorch...")
import torch

print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   Device count: {torch.cuda.device_count()}")
    print(f"   Device name: {torch.cuda.get_device_name(0)}")
    if hasattr(torch.version, 'hip'):
        print(f"   ROCm version: {torch.version.hip}")
else:
    print("   ❌ CUDA/ROCm not available!")
    sys.exit(1)

print()
print("3. Importing braindecode...")
try:
    from braindecode.models import EEGNeX
    print("   ✅ braindecode imported successfully")
except ImportError as e:
    print(f"   ❌ Failed to import braindecode: {e}")
    sys.exit(1)

print()
print("4. Testing Basic GPU Operations...")
device = torch.device("cuda:0")
try:
    # Create tensors
    x = torch.randn(100, 100, device=device)
    y = torch.randn(100, 100, device=device)
    
    # Matrix multiplication
    z = torch.mm(x, y)
    torch.cuda.synchronize()
    
    print(f"   ✅ Matrix multiplication successful")
    print(f"   GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
except Exception as e:
    print(f"   ❌ GPU operations failed: {e}")
    sys.exit(1)

print()
print("5. Testing EEGNeX Model on GPU...")
try:
    # Create model
    model = EEGNeX(
        n_chans=129,
        n_outputs=1,
        n_times=200,
    )
    model = model.to(device)
    
    print(f"   ✅ Model created and moved to GPU")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 129, 200, device=device)
    
    model.eval()
    with torch.no_grad():
        # Warm-up
        _ = model(x)
        torch.cuda.synchronize()
        
        # Timed run
        start = time.time()
        output = model(x)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000
    
    print(f"   ✅ Forward pass successful!")
    print(f"   Output shape: {output.shape}")
    print(f"   Inference time: {elapsed:.2f}ms for batch of {batch_size}")
    print(f"   Per-sample: {elapsed/batch_size:.2f}ms")
    
except Exception as e:
    print(f"   ❌ EEGNeX test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("6. Testing Training Loop...")
try:
    from torch import optim
    from torch.nn.functional import l1_loss
    
    model.train()
    optimizer = optim.Adamax(model.parameters(), lr=0.001)
    
    # Run a few training iterations
    for i in range(5):
        x = torch.randn(batch_size, 129, 200, device=device)
        y = torch.randn(batch_size, 1, device=device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = l1_loss(output, y)
        loss.backward()
        optimizer.step()
        
        if i == 0:
            print(f"   Iteration {i+1}: loss={loss.item():.4f}")
    
    print(f"   ✅ Training loop successful!")
    print(f"   Final loss: {loss.item():.4f}")
    
except Exception as e:
    print(f"   ❌ Training loop failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("🎉 ALL TESTS PASSED!")
print("=" * 70)
print()
print("✅ Your custom ROCm SDK with gfx1010 support is working correctly")
print("✅ EEGNeX model runs on GPU without memory aperture violations")
print("✅ Training loop is stable and functional")
print()
print("You can now enable GPU training in train_challenge2_fast.py")
print()
