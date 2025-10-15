#!/bin/bash
# Monitor foundation training progress

echo "🔍 Foundation Training Monitor"
echo "========================================================================"

# Find the Python process
PID=$(ps aux | grep "python3 scripts/train_simple" | grep -v grep | awk '{print $2}')

if [ -z "$PID" ]; then
    echo "❌ No training process found"
    echo ""
    echo "Recent logs:"
    ls -lth logs/foundation_full_*.log 2>/dev/null | head -3
    exit 1
fi

# Show process info
echo "✅ Training process found: PID $PID"
ps -p $PID -o pid,etime,%cpu,%mem,cmd | tail -1
echo ""

# Show log file
LOG_FILE=$(ls -t logs/foundation_full_*.log 2>/dev/null | head -1)
if [ -n "$LOG_FILE" ]; then
    echo "📄 Log file: $LOG_FILE"
    SIZE=$(du -h "$LOG_FILE" | cut -f1)
    echo "   Size: $SIZE"
    echo ""
    echo "📊 Latest output:"
    echo "------------------------------------------------------------------------"
    tail -50 "$LOG_FILE" 2>/dev/null || echo "   (Log file is still being written...)"
else
    echo "⚠️  No log file found yet"
fi

echo "========================================================================"
echo ""
echo "Commands:"
echo "  Watch live: tail -f logs/foundation_full_*.log"
echo "  Kill training: kill $PID"
echo "  This script: ./monitor_foundation.sh"
