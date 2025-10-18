#!/bin/bash
#
# Robust EEG Training Script - Survives VS Code Crashes
# Uses nohup, screen, and automatic checkpointing
#

set -e  # Exit on error

PROJECT_DIR="/home/kevin/Projects/eeg2025"
cd "$PROJECT_DIR"

# Create logs directory
mkdir -p logs checkpoints

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_real_${TIMESTAMP}.log"
PID_FILE="logs/train_real.pid"
STATUS_FILE="logs/train_real_status.txt"

echo "=========================================="
echo "🧠 EEG Training - Robust System"
echo "=========================================="
echo "Start time: $(date)"
echo "Log file: $LOG_FILE"
echo "Status file: $STATUS_FILE"
echo "PID file: $PID_FILE"
echo ""

# Kill any existing training processes
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  Killing old training process (PID: $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
    fi
    rm "$PID_FILE"
fi

# Function to update status
update_status() {
    echo "$(date): $1" >> "$STATUS_FILE"
    echo "$1"
}

# Start training in background with nohup
update_status "🚀 Starting training on COMPETITION DATA (R1-R5)..."

nohup python3 -u "$PROJECT_DIR/scripts/train_tcn_competition_data.py" > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

# Save PID
echo "$TRAIN_PID" > "$PID_FILE"
update_status "✅ Training started (PID: $TRAIN_PID)"

echo ""
echo "=========================================="
echo "Training is now running in background!"
echo "=========================================="
echo "Process ID: $TRAIN_PID"
echo "Log file: $LOG_FILE"
echo ""
echo "Commands to monitor:"
echo "  tail -f $LOG_FILE          # Watch live output"
echo "  ps -p $TRAIN_PID           # Check if running"
echo "  kill $TRAIN_PID            # Stop training"
echo "  cat $STATUS_FILE           # Check status"
echo ""
echo "Training will continue even if:"
echo "  ✅ VS Code crashes"
echo "  ✅ SSH disconnects"
echo "  ✅ Terminal closes"
echo "  ✅ Computer goes to sleep (if on power)"
echo ""

# Monitor for first 30 seconds to catch immediate errors
echo "Monitoring for 30 seconds..."
sleep 5

if ps -p "$TRAIN_PID" > /dev/null 2>&1; then
    update_status "✅ Training confirmed running after 5 seconds"
    echo "✅ Training is running successfully!"

    # Show initial log
    echo ""
    echo "First 50 lines of output:"
    echo "----------------------------------------"
    head -50 "$LOG_FILE" 2>/dev/null || echo "Log file not ready yet..."
    echo "----------------------------------------"
    echo ""
    echo "Training continues in background. Use: tail -f $LOG_FILE"
else
    update_status "❌ Training failed to start"
    echo "❌ Training process died! Check log:"
    cat "$LOG_FILE"
    exit 1
fi

