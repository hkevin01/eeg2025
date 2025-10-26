#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║       🧠 EEG Challenge 2025 - SAM Training Session                  ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Create training logs directory
mkdir -p logs/training_$(date +%Y%m%d)

# Training configuration
DEVICE="cpu"  # Using CPU due to gfx1030 GPU limitations
EPOCHS_C1=50
EPOCHS_C2=20
BATCH_SIZE_C1=32
BATCH_SIZE_C2=8

echo "📋 Training Configuration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Device: $DEVICE (CPU - stable for gfx1030)"
echo "  Challenge 1: $EPOCHS_C1 epochs, batch size $BATCH_SIZE_C1"
echo "  Challenge 2: $EPOCHS_C2 epochs, batch size $BATCH_SIZE_C2"
echo "  Optimizer: SAM (Sharpness-Aware Minimization)"
echo "  CPU Threads: $(nproc)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Set CPU threading for optimal performance
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)

echo "🎯 Challenge 1: Response Time Prediction (CCD Task)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Target: NRMSE < 0.35 (Current best: 0.3008)"
echo "Starting training..."
echo ""

python3 training/train_c1_sam_simple.py \
    --device $DEVICE \
    --epochs $EPOCHS_C1 \
    --batch_size $BATCH_SIZE_C1 \
    --learning_rate 0.001 \
    --sam_rho 0.05 \
    --weight_decay 0.0001 \
    --output_dir checkpoints/sam_training_$(date +%Y%m%d) \
    | tee logs/training_$(date +%Y%m%d)/c1_sam_training.log

echo ""
echo "✅ Challenge 1 training complete!"
echo ""

echo "🎯 Challenge 2: Externalizing Factor Prediction"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Target: NRMSE < 0.25 (Current best: 0.2042)"
echo "Starting training..."
echo ""

python3 training/train_c2_sam_real_data.py \
    --device $DEVICE \
    --epochs $EPOCHS_C2 \
    --batch_size $BATCH_SIZE_C2 \
    --learning_rate 0.001 \
    --sam_rho 0.05 \
    --weight_decay 0.0001 \
    --output_dir checkpoints/sam_training_$(date +%Y%m%d) \
    | tee logs/training_$(date +%Y%m%d)/c2_sam_training.log

echo ""
echo "✅ Challenge 2 training complete!"
echo ""

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║               🎉 All Training Complete!                             ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "�� Check Results:"
echo "  - Logs: logs/training_$(date +%Y%m%d)/"
echo "  - Checkpoints: checkpoints/sam_training_$(date +%Y%m%d)/"
echo ""
echo "📦 Next Steps:"
echo "  1. Copy best weights to submission package"
echo "  2. Test with: python test_submission_verbose.py"
echo "  3. Upload to competition platform"
echo ""
