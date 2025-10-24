#!/bin/bash
# GPU-Accelerated Challenge 2 Training (Skip Health Check)
# For AMD RX 5600 XT with ROCm workarounds

set -e

cd /home/kevin/Projects/eeg2025

# ROCm environment variables + skip health check
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128
export HSA_ENABLE_SDMA=0
export HSA_SKIP_GPU_CHECK=1  # Skip the problematic health check

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_c2_gpu_skip_${TIMESTAMP}.log"

echo "🚀 GPU Training - RX 5600 XT (Health Check Bypassed)"
echo "===================================================="
echo "Timestamp: $TIMESTAMP"
echo "Log file: $LOG_FILE"
echo ""
echo "ROCm Workarounds:"
echo "  - HSA_OVERRIDE_GFX_VERSION=10.3.0"
echo "  - PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128"
echo "  - HSA_ENABLE_SDMA=0"
echo "  - HSA_SKIP_GPU_CHECK=1"
echo ""
echo "Training Settings:"
echo "  - Batch size: 64 (conservative for RX 5600 XT)"
echo "  - Max epochs: 5"
echo "  - Device: CUDA (ROCm)"
echo "  - Workers: 2"
echo ""
echo "Expected: Much faster than 30 hours/epoch!"
echo "===================================================="
echo ""

# Run training
python -u scripts/training/train_challenge2_r1r2.py \
    --batch-size 64 \
    --num-workers 2 \
    --max-epochs 5 \
    --device cuda \
    --no-pin-memory \
    --note "GPU RX5600XT - Skip health check" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "✅ Training complete!"
