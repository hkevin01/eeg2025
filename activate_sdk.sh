#!/bin/bash
# Activate custom ROCm SDK with gfx1010 PyTorch support

export ROCM_SDK_PATH="/opt/rocm_sdk_612"
export PYTHONPATH="${ROCM_SDK_PATH}/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="${ROCM_SDK_PATH}/lib:${ROCM_SDK_PATH}/lib64:${LD_LIBRARY_PATH}"
export PATH="${ROCM_SDK_PATH}/bin:${PATH}"

# IMPORTANT: Unset HSA override - not needed with proper gfx1010 build
unset HSA_OVERRIDE_GFX_VERSION

# Use SDK Python
alias sdk_python="${ROCM_SDK_PATH}/bin/python3"
alias sdk_pip="${ROCM_SDK_PATH}/bin/pip3"

echo "✅ ROCm SDK 6.1.2 with PyTorch 2.4.1 (gfx1010 support) activated"
echo "📍 SDK Path: ${ROCM_SDK_PATH}"
echo "🐍 Python: ${ROCM_SDK_PATH}/bin/python3"
echo ""
echo "Quick test:"
sdk_python -c "import torch; print(f'PyTorch {torch.__version__} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "⚠️  Import failed - check dependencies"
echo ""
echo "Usage:"
echo "  sdk_python your_script.py"
echo "  sdk_pip install package_name"
