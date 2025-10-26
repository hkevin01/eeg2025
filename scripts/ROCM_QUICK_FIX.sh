#!/bin/bash
# Quick ROCm RDNA1 Convolution Fix Installer
# For AMD RX 5600 XT, RX 5700, RX 6800M (gfx1030/gfx1031)

set -e

echo "🔧 Installing ROCm RDNA1 Convolution Fixes..."
echo ""

# Backup existing bashrc
if [ -f ~/.bashrc ]; then
    cp ~/.bashrc ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)
    echo "✅ Backed up ~/.bashrc"
fi

# Check if fixes already applied
if grep -q "MIOPEN_DEBUG_CONV_GEMM" ~/.bashrc 2>/dev/null; then
    echo "⚠️  ROCm convolution fixes already present in ~/.bashrc"
    echo "   Skipping to avoid duplicates."
    exit 0
fi

# Add fixes to bashrc
cat >> ~/.bashrc << 'ROCM_FIXES'

# === ROCm RDNA1 GPU Convolution Fixes ===
# Source: https://github.com/ROCm/MIOpen/issues/3540
# Fixes memory access faults in convolution operations on RDNA1 GPUs

# GPU Detection
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH=gfx1030
export GPU_DEVICE_ORDINAL=0
export HIP_VISIBLE_DEVICES=0

# MIOpen Convolution Algorithm Fixes (CRITICAL!)
export MIOPEN_DEBUG_CONV_GEMM=0                # Disable GEMM convolutions
export MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2=0     # Disable Direct OCL WrW2
export MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW53=0    # Disable Direct OCL WrW53

# Optional: Performance tuning
export MIOPEN_FIND_MODE=normal
export MIOPEN_LOG_LEVEL=4  # Warnings and errors only

# === End ROCm RDNA1 Fixes ===
ROCM_FIXES

echo ""
echo "✅ ROCm convolution fixes added to ~/.bashrc"
echo ""
echo "📋 Next steps:"
echo "   1. Reload your shell: source ~/.bashrc"
echo "   2. Test GPU detection: python -c 'import torch; print(torch.cuda.is_available())'"
echo "   3. Test convolution: python -c 'import torch; c=torch.nn.Conv2d(3,16,3).cuda(); print(c(torch.randn(1,3,64,64).cuda()).shape)'"
echo ""
echo "📖 Full documentation: ROCM_CONVOLUTION_FIX.md"
echo ""
