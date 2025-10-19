#!/bin/bash

# Quick training status check (no auto-refresh)
LOG_FILE="logs/challenge2_correct_training.log"

echo "🧠 Challenge 2 Training - Quick Status"
echo "======================================="
echo ""

# Process check
if pgrep -f "train_challenge2_correct.py" > /dev/null; then
    PID=$(pgrep -f "train_challenge2_correct.py")
    RUNTIME=$(ps -p $PID -o etime --no-headers 2>/dev/null)
    echo "✅ Training RUNNING (PID: $PID, Runtime: $RUNTIME)"
else
    echo "❌ Training NOT running"
fi

echo ""
echo "📊 Current Progress:"
echo "-------------------"

# Get latest epoch info
LATEST_EPOCH=$(grep -oP "Epoch \K[0-9]+(?=/)" "$LOG_FILE" | tail -1)
LATEST_BATCH=$(grep -oP "Batch \K[0-9]+(?=/)" "$LOG_FILE" | tail -1)
TOTAL_BATCHES=$(grep -oP "Batch [0-9]+/\K[0-9]+" "$LOG_FILE" | tail -1)

if [ ! -z "$LATEST_EPOCH" ]; then
    PROGRESS=$((LATEST_BATCH * 100 / TOTAL_BATCHES))
    echo "Epoch: $LATEST_EPOCH/20"
    echo "Batch: $LATEST_BATCH/$TOTAL_BATCHES ($PROGRESS%)"
fi

# Get recent loss values
echo ""
echo "Recent Loss Values:"
grep -oP "Batch [0-9]+/[0-9]+ - Loss: \K[0-9.]+" "$LOG_FILE" | tail -10 | awk '{
    sum += $1
    count++
}
END {
    if (count > 0) {
        avg = sum / count
        printf "  Last 10 batches avg: %.4f\n", avg
    }
}'

# Show last 3 lines
echo ""
echo "Last 3 log lines:"
tail -3 "$LOG_FILE" | sed 's/^/  /'

echo ""
echo "Commands:"
echo "  Full monitor: ./monitor_challenge2.sh"
echo "  Live tail: tail -f $LOG_FILE"
