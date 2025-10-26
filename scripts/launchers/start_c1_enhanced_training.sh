#!/bin/bash

# Start Enhanced C1 Training with ROCm SDK on GPU
# RULE: ALWAYS USE ROCM SDK PYTHON AND PYTORCH

echo "================================================================================================"
echo "🚀 Starting Enhanced Challenge 1 Training"
echo "================================================================================================"
echo "Features: Temporal Attention + Mixup + Multi-Scale + SAM"
echo "Device: AMD Radeon RX 5600 XT via ROCm SDK"
echo "================================================================================================"

# CRITICAL: Setup COMPLETE ROCm SDK environment
export ROCM_PATH="/opt/rocm_sdk_612"
export LD_LIBRARY_PATH="/opt/rocm_sdk_612/lib:/opt/rocm_sdk_612/lib64:/opt/rocm_sdk_612/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/opt/rocm_sdk_612/lib/python3.11/site-packages:$PYTHONPATH"
export PATH="/opt/rocm_sdk_612/bin:$PATH"

# ROCm GPU configuration
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH="gfx1030"

# Memory optimization for ROCm
export HSA_XNACK=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROCM_MALLOC_PREFILL=1
export AMD_SERIALIZE_KERNEL=3
export TORCH_USE_HIP_DSA=1

# Verify we're using ROCm SDK Python
echo "🔍 Verifying ROCm SDK Setup:"
echo "   Python: $(/opt/rocm_sdk_612/bin/python3 --version)"
/opt/rocm_sdk_612/bin/python3 -c "import torch; print(f'   PyTorch: {torch.__version__}'); print(f'   ROCm/HIP: {torch.version.hip}'); print(f'   CUDA Available: {torch.cuda.is_available()}')"
echo "================================================================================================"

# Optional quick dry-run flag (usage: ./start_c1_enhanced_training.sh --quick)
EXTRA_ARGS=()
if [[ "$1" == "--quick" ]]; then
    EXTRA_ARGS+=("--quick-dry-run")
    shift
fi

# Allow callers to forward additional CLI arguments
EXTRA_ARGS+=("$@")

# Run training with logging using ROCm SDK Python
cd /home/kevin/Projects/eeg2025

/opt/rocm_sdk_612/bin/python3 train_c1_enhanced.py \
    --data_dirs data/ds005506-bdf data/ds005507-bdf \
    --max_subjects 20 \
    --epochs 30 \
    --batch_size 4 \
    --lr 0.001 \
    --rho 0.05 \
    --mixup_alpha 0.0 \
    --exp_name enhanced_v4_rocm_nomixup_small \
    --early_stopping 15 \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee training_c1_enhanced.log

echo ""
echo "================================================================================================"
echo "✅ Training Complete! Check training_c1_enhanced.log for details"
echo "================================================================================================"
