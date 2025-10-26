#!/bin/bash
# Setup script for EEG training with GPU acceleration on gfx1030

echo "🔧 Configuring GPU environment for EEG training..."

# Export MIOpen settings for gfx1030 (AMD RX 5600 XT)
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH=gfx1030
export MIOPEN_FIND_MODE=2  # Immediate mode - compile kernels on-demand
export MIOPEN_DEBUG_DISABLE_FIND_DB=1  # Bypass incomplete database
export MIOPEN_DISABLE_CACHE=1  # Fresh kernel compilation

# Activate virtual environment
source venv_pytorch28_rocm70/bin/activate

echo "✅ GPU environment configured"
echo "   PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "   ROCm: $(python3 -c 'import torch; print(torch.version.hip)')"
echo "   GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not available")')"
echo ""
echo "🎯 Ready for EEG model training!"
echo "   Note: First convolution will take ~3-4s for kernel compilation"
echo "   Subsequent operations will be faster"
echo ""
echo "Usage: source setup_gpu_training.sh"
