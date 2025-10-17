#!/bin/bash
# Monitor GPU training without blocking

LOG_DIR="/home/kevin/Projects/eeg2025/logs"
LATEST_LOG=$(ls -t $LOG_DIR/gpu_quick_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "No training log found"
    exit 1
fi

echo "📊 Monitoring: $LATEST_LOG"
echo "=" echo "================================"
echo ""

# Check if process is running
PID=$(ps aux | grep "train_gpu_quick.py" | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
    echo "✅ Training process running (PID: $PID)"
else
    echo "⚠️  No training process found"
fi

echo ""
echo "Latest output:"
echo "---"
tail -30 "$LATEST_LOG"

echo ""
echo "---"
echo "To watch live: watch -n 2 tail -20 $LATEST_LOG"
echo "To stop: pkill -f train_gpu_quick.py"
