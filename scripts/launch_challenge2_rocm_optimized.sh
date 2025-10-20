#!/bin/bash
#
# Optimized Challenge 2 Training Launcher for ROCm
# Based on research findings and health check diagnostics
#

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   🚀 Challenge 2 Training - ROCm Optimized Configuration   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Export recommended environment variables for RX 5600 XT
export PYTORCH_ROCM_ARCH=gfx1010
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # Match device capability from health check
export HIP_VISIBLE_DEVICES=0

# Disable hipBLAS workspaces to avoid memory allocation issues
export HIPBLAS_WORKSPACE_CONFIG=:0:0
export CUBLAS_WORKSPACE_CONFIG=:0:0  # Compatibility

# Conservative memory management
export PYTORCH_NO_HIP_MEMORY_CACHING=0  # Keep caching ON for performance, but monitor

# Debug level (set to 7 for maximum verbosity if troubleshooting)
export HSAKMT_DEBUG_LEVEL=4  # Moderate: errors + warnings

echo "🔧 ROCm Environment Configuration:"
echo "   PYTORCH_ROCM_ARCH: $PYTORCH_ROCM_ARCH"
echo "   HSA_OVERRIDE_GFX_VERSION: $HSA_OVERRIDE_GFX_VERSION"
echo "   HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
echo "   HIPBLAS_WORKSPACE_CONFIG: $HIPBLAS_WORKSPACE_CONFIG"
echo "   HSAKMT_DEBUG_LEVEL: $HSAKMT_DEBUG_LEVEL"
echo ""

# Training parameters (conservative defaults)
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-1}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
MAX_EPOCHS=${MAX_EPOCHS:-20}
LR=${LR:-0.001}

echo "📊 Training Configuration:"
echo "   Batch Size: $BATCH_SIZE"
echo "   Num Workers: $NUM_WORKERS"
echo "   Prefetch Factor: $PREFETCH_FACTOR"
echo "   Max Epochs: $MAX_EPOCHS"
echo "   Learning Rate: $LR"
echo ""

# Create log directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_rocm_optimized_${TIMESTAMP}.log"

echo "📝 Logging to: $LOG_FILE"
echo ""

# Confirmation prompt
read -p "🚀 Ready to launch training? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Training cancelled"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Starting training..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Launch training with optimized parameters
python3 scripts/training/train_challenge2_r1r2.py \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --prefetch-factor $PREFETCH_FACTOR \
    --max-epochs $MAX_EPOCHS \
    --lr $LR \
    --no-pin-memory \
    --no-persistent-workers \
    --note "ROCm optimized: conservative memory settings, CPU fallback enabled" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully"
else
    echo "❌ Training failed with exit code $EXIT_CODE"
    echo "   Check log file: $LOG_FILE"
    echo "   See docs/rocm_troubleshooting.md for debugging tips"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Show CPU fallback detection
if grep -q "CPU fallback ready" "$LOG_FILE"; then
    echo "⚠️  WARNING: GPU execution failed, training fell back to CPU"
    echo "   This will be MUCH slower but should complete successfully"
    echo "   GPU failure details:"
    grep -A 3 "ROCm GPU error detected" "$LOG_FILE" | sed 's/^/   /'
    echo ""
fi

exit $EXIT_CODE
