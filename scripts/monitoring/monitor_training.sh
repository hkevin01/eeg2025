#!/bin/bash
echo "🔍 Challenge 1 Training Monitor"
echo "================================"
echo ""

# Check if training is running
if pgrep -f "train_c1_tonight.py" > /dev/null; then
    echo "✅ Training is RUNNING"
    PID=$(pgrep -f "train_c1_tonight.py")
    echo "   PID: $PID"
    echo ""
else
    echo "❌ Training is NOT running"
    echo ""
fi

# Find log file
if [ -f nohup.out ]; then
    LOG_FILE="nohup.out"
elif [ -f training_c1_tonight.log ]; then
    LOG_FILE="training_c1_tonight.log"
elif [ -f training_start.log ]; then
    LOG_FILE="training_start.log"
else
    echo "No log file found"
    exit 1
fi

echo "📊 Latest training output ($LOG_FILE):"
echo "================================"
tail -30 "$LOG_FILE"

echo ""
echo "================================"
echo "Commands:"
echo "  tail -f $LOG_FILE          # Follow log"
echo "  ps aux | grep train        # Check process"
echo "  kill <PID>                 # Stop training"
