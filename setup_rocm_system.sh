#!/bin/bash
# Make ROCm SDK the System Default Python Environment
# This creates symlinks and sets up environment variables globally

echo "════════════════════════════════════════════════════════════════════"
echo "  🚀 Setting up ROCm SDK as System Default Python"
echo "════════════════════════════════════════════════════════════════════"

# Create ~/.bashrc additions for ROCm SDK
BASHRC_FILE="$HOME/.bashrc"
ROCM_SECTION="# ROCm SDK Environment (Auto-added by setup_rocm_system.sh)"

# Remove existing ROCm section if present
sed -i "/$ROCM_SECTION/,/# End ROCm SDK Environment/d" "$BASHRC_FILE"

# Add ROCm SDK environment to bashrc
cat >> "$BASHRC_FILE" << 'BASHRC_EOF'

# ROCm SDK Environment (Auto-added by setup_rocm_system.sh)
export ROCM_PATH="/opt/rocm_sdk_612"
export LD_LIBRARY_PATH="/opt/rocm_sdk_612/lib:/opt/rocm_sdk_612/lib64:/opt/rocm_sdk_612/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/opt/rocm_sdk_612/lib/python3.11/site-packages:$PYTHONPATH"
export PATH="/opt/rocm_sdk_612/bin:$PATH"

# ROCm GPU configuration
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH="gfx1010"

# Memory optimization for ROCm
export HSA_XNACK=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROCM_MALLOC_PREFILL=1

# MNE Configuration (prevent GUI issues)
export MNE_USE_CUDA=false
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg

# Make python3 use ROCm SDK by default
alias python3='/opt/rocm_sdk_612/bin/python3'
alias python='/opt/rocm_sdk_612/bin/python3'
alias pip3='/opt/rocm_sdk_612/bin/pip3'
alias pip='/opt/rocm_sdk_612/bin/pip3'
# End ROCm SDK Environment

BASHRC_EOF

echo "✅ Added ROCm SDK environment to ~/.bashrc"

# Export for current session
export ROCM_PATH="/opt/rocm_sdk_612"
export LD_LIBRARY_PATH="/opt/rocm_sdk_612/lib:/opt/rocm_sdk_612/lib64:/opt/rocm_sdk_612/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/opt/rocm_sdk_612/lib/python3.11/site-packages:$PYTHONPATH"
export PATH="/opt/rocm_sdk_612/bin:$PATH"
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH="gfx1010"
export HSA_XNACK=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROCM_MALLOC_PREFILL=1
export MNE_USE_CUDA=false
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg

echo "✅ Exported ROCm SDK environment variables for current session"

# Verify setup
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  🔍 Verifying ROCm SDK Setup"
echo "════════════════════════════════════════════════════════════════════"

/opt/rocm_sdk_612/bin/python3 << 'PYTEST'
import torch
print(f"✅ Python: 3.11 (ROCm SDK)")
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ ROCm/HIP: {torch.version.hip}")
print(f"✅ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

import mne
print(f"✅ MNE: {mne.__version__}")
import braindecode
print(f"✅ braindecode: {braindecode.__version__}")
import pandas
print(f"✅ pandas: {pandas.__version__}")
import numpy
print(f"✅ numpy: {numpy.__version__}")
PYTEST

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  ✅ ROCm SDK is now the system default!"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "To use in current terminal, run:"
echo "  source ~/.bashrc"
echo ""
echo "Or restart your terminal for changes to take effect automatically."
echo ""
