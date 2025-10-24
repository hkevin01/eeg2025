#!/usr/bin/env python3
"""Test EEGNeX with different ROCm configurations"""

import torch
import os
import sys

# Try different ROCm environment configurations
configs = [
    {
        "name": "Default",
        "env": {}
    },
    {
        "name": "Conservative Memory",
        "env": {
            "HSA_OVERRIDE_GFX_VERSION": "10.3.0",
            "PYTORCH_HIP_ALLOC_CONF": "max_split_size_mb:64",
            "HSA_ENABLE_SDMA": "0",
        }
    },
    {
        "name": "Force CPU Fallback for Conv",
        "env": {
            "HSA_OVERRIDE_GFX_VERSION": "10.3.0",
            "PYTORCH_HIP_ALLOC_CONF": "max_split_size_mb:128",
            "HSA_ENABLE_SDMA": "0",
            "MIOPEN_DISABLE_CACHE": "1",
            "MIOPEN_DEBUG_DISABLE_FIND_DB": "1",
        }
    },
    {
        "name": "ROCm 5.x Emulation",
        "env": {
            "HSA_OVERRIDE_GFX_VERSION": "9.0.0",  # Pretend to be older arch
            "PYTORCH_HIP_ALLOC_CONF": "max_split_size_mb:256",
        }
    }
]

sys.path.insert(0, '/home/kevin/Projects/eeg2025/src')

for config in configs:
    print(f"\n{'='*60}")
    print(f"Testing: {config['name']}")
    print(f"{'='*60}")
    
    # Set environment
    for key, value in config['env'].items():
        os.environ[key] = value
        print(f"  {key}={value}")
    
    try:
        # Clear any cached modules
        if 'braindecode.models' in sys.modules:
            del sys.modules['braindecode.models']
        
        from braindecode.models import EEGNeX
        
        print("\n  Creating model...")
        model = EEGNeX(
            n_outputs=1,
            n_chans=129,
            n_times=200,
            sfreq=100,
            drop_prob=0.5
        ).cuda()
        
        print(f"  ✅ Model created on GPU")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass with small batch
        print("  Testing forward pass (batch=2)...")
        x = torch.randn(2, 129, 200, device='cuda')
        with torch.no_grad():
            y = model(x)
        print(f"  ✅ Forward pass works! Output: {y.shape}")
        
        # Test with larger batch
        print("  Testing forward pass (batch=16)...")
        x = torch.randn(16, 129, 200, device='cuda')
        with torch.no_grad():
            y = model(x)
        print(f"  ✅ Batch 16 works! Output: {y.shape}")
        
        # Test backward pass
        print("  Testing backward pass...")
        x = torch.randn(4, 129, 200, device='cuda', requires_grad=True)
        y = model(x)
        loss = y.mean()
        loss.backward()
        print(f"  ✅ Backward pass works!")
        
        print(f"\n  🎉 SUCCESS with {config['name']}!")
        print(f"  This configuration WORKS for training!")
        break
        
    except Exception as e:
        print(f"  ❌ Failed: {type(e).__name__}: {str(e)[:100]}")
        # Clean up
        torch.cuda.empty_cache()
        continue

print(f"\n{'='*60}")
print("Testing complete")
print(f"{'='*60}")
