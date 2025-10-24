#!/bin/bash
# GPU-Accelerated Challenge 2 Training
# Optimized settings for fast convergence

set -e

cd /home/kevin/Projects/eeg2025

# Setup ROCm environment
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_c2_gpu_${TIMESTAMP}.log"

echo "🚀 Starting GPU-Accelerated Challenge 2 Training"
echo "================================================"
echo "Timestamp: $TIMESTAMP"
echo "Log file: $LOG_FILE"
echo "Device: AMD Radeon RX 5600 XT (ROCm)"
echo ""
echo "Settings:"
echo "  - Batch size: 128 (8x larger for GPU)"
echo "  - Max epochs: 5 (faster convergence)"
echo "  - Device: CUDA (ROCm)"
echo "  - Workers: 4 (parallel data loading)"
echo ""
echo "Expected time: ~30 minutes per epoch"
echo "================================================"
echo ""

# Run training with GPU-optimized settings
python -u scripts/training/train_challenge2_r1r2.py \
    --batch-size 128 \
    --num-workers 4 \
    --max-epochs 5 \
    --device cuda \
    --no-pin-memory \
    --note "GPU training - RX 5600 XT - Fast convergence" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "✅ Training complete! Check results in $LOG_FILE"
