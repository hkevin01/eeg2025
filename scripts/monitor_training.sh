#!/bin/bash
# Monitor training progress

LOG_DIR="/home/kevin/Projects/eeg2025/logs"
CHECKPOINT_DIR="/home/kevin/Projects/eeg2025/checkpoints"

echo "🔍 Training Monitor"
echo "=" 
echo ""

# Find latest log
LATEST_LOG=$(ls -t $LOG_DIR/training_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ No training log found!"
    exit 1
fi

echo "📋 Latest log: $(basename $LATEST_LOG)"
echo ""

# Check if training is running
if ps aux | grep -v grep | grep "train_foundation_cpu.py" > /dev/null; then
    echo "✅ Training is RUNNING"
else
    echo "⚠️  Training process not found"
fi

echo ""
echo "📊 Recent progress:"
echo "─────────────────────────────────────────────"
tail -30 "$LATEST_LOG" | grep -E "(Epoch|Loss|Acc|saved|Complete)" || tail -15 "$LATEST_LOG"
echo "─────────────────────────────────────────────"

echo ""
echo "💾 Checkpoints:"
if [ -d "$CHECKPOINT_DIR" ]; then
    ls -lh "$CHECKPOINT_DIR"/*.pth 2>/dev/null || echo "   No checkpoints yet"
else
    echo "   Checkpoint directory not created yet"
fi

echo ""
echo "🔄 To view live updates: tail -f $LATEST_LOG"
