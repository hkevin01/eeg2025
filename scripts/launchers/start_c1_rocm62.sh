#!/bin/bash

# Start Enhanced C1 Training with ROCm 6.2 System Installation
# This script uses the new venv_rocm622 with PyTorch 2.5.1+rocm6.2

echo "================================================================================================"
echo "🚀 Starting Enhanced Challenge 1 Training (ROCm 6.2)"
echo "================================================================================================"
echo "Features: Temporal Attention + Mixup + Multi-Scale + SAM"
echo "Device: AMD Radeon RX 5600 XT via ROCm 6.2"
echo "================================================================================================"

# Clear old ROCm SDK environment variables
unset ROCM_PATH
unset PYTORCH_ROCM_ARCH

# Setup ROCm 6.2 system environment (CLEAN)
export ROCM_PATH="/opt/rocm"
export PATH="/opt/rocm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH"

# Clear SDK Python path pollution
export PYTHONPATH=""

# ROCm GPU configuration
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Memory optimization for ROCm
export HSA_XNACK=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROCM_MALLOC_PREFILL=1
export AMD_SERIALIZE_KERNEL=3
export TORCH_USE_HIP_DSA=1

# MNE Configuration
export MNE_USE_CUDA=false
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg

# Activate clean venv
cd /home/kevin/Projects/eeg2025
source venv_rocm622/bin/activate

# Verify environment
echo "🔍 Verifying ROCm 6.2 Setup:"
echo "   Python: $(python --version)"
python -c "import torch; print(f'   PyTorch: {torch.__version__}'); print(f'   ROCm/HIP: {torch.version.hip}'); print(f'   CUDA Available: {torch.cuda.is_available()}'); print(f'   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo "================================================================================================"

# Optional quick dry-run flag (usage: ./start_c1_rocm62.sh --quick)
EXTRA_ARGS=()
if [[ "$1" == "--quick" ]]; then
    EXTRA_ARGS+=("--quick-dry-run")
    shift
fi

# Allow callers to forward additional CLI arguments
EXTRA_ARGS+=("$@")

# Run training with logging
python train_c1_enhanced.py \
    --data_dirs data/ds005506-bdf data/ds005507-bdf \
    --max_subjects 20 \
    --epochs 30 \
    --batch_size 4 \
    --lr 0.001 \
    --rho 0.05 \
    --mixup_alpha 0.0 \
    --exp_name enhanced_v5_rocm62_test \
    --early_stopping 15 \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee training_c1_rocm62.log

echo ""
echo "================================================================================================"
echo "✅ Training Complete! Check training_c1_rocm62.log for details"
echo "================================================================================================"
