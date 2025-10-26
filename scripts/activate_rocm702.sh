#!/bin/bash
# ROCm 7.0.2 Environment Setup

# Clear SDK pollution
unset ROCM_PATH
unset LD_LIBRARY_PATH
unset PYTHONPATH

# Set ROCm 7.0.2 paths
export ROCM_PATH=/opt/rocm-7.0.2
export LD_LIBRARY_PATH=/opt/rocm-7.0.2/lib:$LD_LIBRARY_PATH
export PATH=/opt/rocm-7.0.2/bin:$PATH
export PYTORCH_ROCM_ARCH=gfx1030
export HIP_VISIBLE_DEVICES=0

# Activate venv
source /home/kevin/Projects/eeg2025/venv_rocm702/bin/activate

echo "✅ ROCm 7.0.2 environment activated"
echo "   ROCM_PATH: $ROCM_PATH"
echo "   Python: $(which python)"
echo "   PyTorch location: $(python -c 'import torch; print(torch.__file__)' 2>/dev/null || echo 'Not installed yet')"
