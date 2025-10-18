#!/bin/bash
#
# Monitor Training Progress
#

PROJECT_DIR="/home/kevin/Projects/eeg2025"
cd "$PROJECT_DIR"

PID_FILE="logs/train_real.pid"
STATUS_FILE="logs/train_real_status.txt"

echo "=========================================="
echo "🔍 Training Status Monitor"
echo "=========================================="
echo ""

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "❌ No training process found (no PID file)"
    echo ""
    echo "Recent logs:"
    ls -lht logs/train_real_*.log 2>/dev/null | head -3 || echo "No log files found"
    exit 1
fi

TRAIN_PID=$(cat "$PID_FILE")

# Check if process is running
if ps -p "$TRAIN_PID" > /dev/null 2>&1; then
    echo "✅ Training is RUNNING (PID: $TRAIN_PID)"
    
    # Get process info
    echo ""
    echo "Process Info:"
    ps -p "$TRAIN_PID" -o pid,ppid,%cpu,%mem,etime,cmd
    
    # Find log file
    LOG_FILE=$(ls -t logs/train_real_*.log 2>/dev/null | head -1)
    if [ -n "$LOG_FILE" ]; then
        echo ""
        echo "Log file: $LOG_FILE"
        echo "Log size: $(ls -lh "$LOG_FILE" | awk '{print $5}')"
        echo ""
        
        # Show last 30 lines
        echo "Last 30 lines of output:"
        echo "----------------------------------------"
        tail -30 "$LOG_FILE"
        echo "----------------------------------------"
        
        # Check for checkpoints
        echo ""
        echo "Recent checkpoints:"
        ls -lht checkpoints/challenge1_tcn_real_*.pth 2>/dev/null | head -5 || echo "No checkpoints yet"
    fi
    
    # Show status
    if [ -f "$STATUS_FILE" ]; then
        echo ""
        echo "Status updates:"
        tail -10 "$STATUS_FILE"
    fi
    
else
    echo "❌ Training process NOT running (PID: $TRAIN_PID was running)"
    echo ""
    echo "Last status:"
    tail -5 "$STATUS_FILE" 2>/dev/null || echo "No status file"
    echo ""
    echo "Check the log file for errors:"
    LOG_FILE=$(ls -t logs/train_real_*.log 2>/dev/null | head -1)
    if [ -n "$LOG_FILE" ]; then
        echo "Log: $LOG_FILE"
        echo ""
        tail -50 "$LOG_FILE"
    fi
fi

echo ""
echo "=========================================="
echo "Commands:"
echo "  ./scripts/monitor_training.sh       # Run this script again"
echo "  tail -f logs/train_real_*.log       # Watch live output"
echo "  kill $TRAIN_PID                     # Stop training"
echo "=========================================="

