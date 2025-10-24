#!/bin/bash
# Challenge 2 Training - Fast CPU Mode
# Optimized for quick convergence

cd /home/kevin/Projects/eeg2025

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_c2_fast_${TIMESTAMP}.log"

echo "🚀 Challenge 2 Fast Training (CPU)"
echo "=================================="
echo "Log: $LOG_FILE"
echo ""
echo "Settings:"
echo "  - Batch size: 128 (4x faster)"
echo "  - Max epochs: 3 (quick convergence)"
echo "  - Device: CPU"
echo "  - Workers: 0 (stable)"
echo ""
echo "Expected: ~7 hours/epoch = ~21 hours total"
echo "=================================="
echo ""

nohup python -u scripts/training/train_challenge2_r1r2.py \
    --batch-size 128 \
    --num-workers 0 \
    --max-epochs 3 \
    --device cpu \
    --no-pin-memory \
    --note "Fast CPU training - batch128 - 3 epochs" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "✅ Training started in background"
echo "   PID: $PID"
echo "   Log: $LOG_FILE"
echo ""
echo "Monitor with: tail -f $LOG_FILE"
