#!/bin/bash
# Test script that uses SDK PyTorch with system Python

echo "Setting up environment for custom ROCm SDK..."

# Set Python path to find SDK PyTorch
export PYTHONPATH="/opt/rocm_sdk_612/lib/python3.11/site-packages:$PYTHONPATH"

# Set library path for ROCm libraries
export LD_LIBRARY_PATH="/opt/rocm_sdk_612/lib:$LD_LIBRARY_PATH"

# Set HSA settings for gfx1010
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HIP_VISIBLE_DEVICES=0

echo "Environment configured:"
echo "  PYTHONPATH includes: /opt/rocm_sdk_612/lib/python3.11/site-packages"
echo "  LD_LIBRARY_PATH includes: /opt/rocm_sdk_612/lib"
echo "  HSA_OVERRIDE_GFX_VERSION: $HSA_OVERRIDE_GFX_VERSION"
echo ""

# Run the test with Python 3.11 (SDK was built with 3.11)
python3.11 tests/test_custom_sdk_eegnex.py
