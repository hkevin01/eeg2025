"""Test different convolution backends on AMD gfx1030"""
import os
import torch
import torch.nn as nn

print("=" * 60)
print("Testing Convolution Backends on gfx1030")
print("=" * 60)

# Test configurations
configs = [
    {
        "name": "Default MIOpen",
        "env": {}
    },
    {
        "name": "MIOpen FallBack Algorithm",
        "env": {
            "MIOPEN_FIND_MODE": "1",  # Use fallback instead of find mode
            "MIOPEN_DEBUG_DISABLE_FIND_DB": "1",
        }
    },
    {
        "name": "MIOpen Direct Algorithm",
        "env": {
            "MIOPEN_FIND_MODE": "2",  # Direct algorithm
            "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM": "0",
        }
    },
    {
        "name": "Force GEMM Convolution",
        "env": {
            "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM": "1",
            "MIOPEN_DEBUG_CONV_DIRECT": "0",
            "MIOPEN_DEBUG_CONV_WINOGRAD": "0",
        }
    },
    {
        "name": "Disable MIOpen Caching",
        "env": {
            "MIOPEN_DISABLE_CACHE": "1",
            "MIOPEN_CUSTOM_CACHE_DIR": "/tmp/miopen_cache",
        }
    },
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

def test_conv(name, kernel_size, groups=1):
    """Test a specific convolution configuration"""
    try:
        print(f"  Testing {name}: kernel={kernel_size}, groups={groups}...")
        
        # Create simple conv layer
        if groups == 1:
            conv = nn.Conv2d(32, 64, kernel_size=kernel_size, padding='same').to(device)
        else:
            conv = nn.Conv2d(32, 64, kernel_size=kernel_size, groups=groups, padding='same').to(device)
        
        # Test input
        x = torch.randn(2, 32, 128, 200, device=device)
        
        # Forward pass
        y = conv(x)
        
        print(f"    ✅ SUCCESS: {x.shape} -> {y.shape}")
        return True
    except Exception as e:
        print(f"    ❌ FAILED: {str(e)[:80]}")
        return False

# Test each configuration
for config in configs:
    print(f"\n{'='*60}")
    print(f"Config: {config['name']}")
    print(f"{'='*60}")
    
    # Set environment
    for key, value in config['env'].items():
        os.environ[key] = value
        print(f"  {key}={value}")
    
    if config['env']:
        print()
    
    # Clear any cached modules
    torch.cuda.empty_cache()
    
    # Test different convolutions
    results = {
        "Small regular": test_conv("Small regular", (3, 3), groups=1),
        "Large regular": test_conv("Large regular", (129, 1), groups=1),
        "Small depthwise": test_conv("Small depthwise", (3, 3), groups=32),
        "Large depthwise": test_conv("Large depthwise", (129, 1), groups=32),
    }
    
    success_count = sum(results.values())
    print(f"\n  Summary: {success_count}/4 tests passed")
    
    if results["Large depthwise"]:
        print(f"\n  🎉 SOLUTION FOUND: {config['name']}")
        print(f"  Environment variables:")
        for key, value in config['env'].items():
            print(f"    export {key}={value}")
        break

print("\n" + "=" * 60)
print("Test complete")
print("=" * 60)
