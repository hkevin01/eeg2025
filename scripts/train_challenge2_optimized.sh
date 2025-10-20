#!/bin/bash
# Optimized Challenge 2 Training Launcher
# Configures ROCm environment and launches training with optimal settings

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     🧠 EEG Challenge 2 - Optimized Training Launcher         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Source ROCm environment
echo "📦 Loading ROCm environment..."
source scripts/setup_rocm_env.sh
echo ""

# Verify GPU status (optional)
echo "🔍 Verifying GPU status..."
python3 - <<'PY'
import torch

if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'✅ GPU detected: {name}')
    print(f'   Memory: {total_gb:.2f} GB')
else:
    print('⚠️  No CUDA/ROCm device detected; training will run on CPU.')
PY
echo ""

# Set optimal training parameters for RX 5600 XT
BATCH_SIZE=16
NUM_WORKERS=2
PREFETCH_FACTOR=2
EPOCHS=20

DEVICE_MODE=${DEVICE_MODE:-auto}

echo "🚀 Starting Challenge 2 training with optimized settings:"
echo "   Batch size: $BATCH_SIZE"
echo "   Workers: $NUM_WORKERS"
echo "   Epochs: $EPOCHS"
echo "   Device preference: $DEVICE_MODE (auto triggers GPU health-check fallback)"
echo ""

# Create log directory
mkdir -p logs

# Generate timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/training_c2_optimized_${TIMESTAMP}.log"

echo "📝 Logging to: $LOGFILE"
echo ""

# Launch training
python scripts/training/train_challenge2_r1r2.py \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --prefetch-factor $PREFETCH_FACTOR \
    --max-epochs $EPOCHS \
    --no-pin-memory \
    --no-persistent-workers \
    --device "$DEVICE_MODE" \
    --note "Challenge2 ROCm optimized training - gfx1030 override" \
    2>&1 | tee "$LOGFILE"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              Training Complete!                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Check results:"
echo "   Log file: $LOGFILE"
echo "   Checkpoints: checkpoints/challenge2_r1r2/"
echo ""
