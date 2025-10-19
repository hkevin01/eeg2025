#!/bin/bash

echo "🧠 Hybrid Neuroscience + CNN Model Training Monitor"
echo "=" | awk '{s=sprintf("%80s",""); gsub(/ /,"=",$0); print}'
echo

# Check if training is running
PID=91691
if ps -p $PID > /dev/null 2>&1; then
    echo "✅ Training RUNNING (PID: $PID)"
    
    # Get CPU and memory usage
    ps -p $PID -o %cpu,%mem,etime,rss | tail -1 | awk '{
        printf "   CPU: %.1f%%\n", $1
        printf "   Memory: %.1fGB (%.1f%%)\n", $4/1048576, $2
        printf "   Runtime: %s\n", $3
    }'
    echo
else
    echo "❌ Training NOT RUNNING (PID: $PID)"
    echo "   Training may have completed or failed."
    echo
fi

# Show latest log file
LOG_FILE=$(ls -t logs/hybrid_training_*.log 2>/dev/null | head -1)

if [ -n "$LOG_FILE" ]; then
    echo "📊 Latest Progress (from $LOG_FILE):"
    echo "---"
    
    # Show last 20 lines
    tail -20 "$LOG_FILE" | grep -E "(Epoch|Train|Val|Best|Stopping|COMPARISON|✅|❌|⚠️)" || tail -20 "$LOG_FILE"
    
    echo
    echo "---"
    echo "📁 Full log: $LOG_FILE"
    echo "💡 Watch live: tail -f $LOG_FILE"
else
    echo "⚠️  No log file found in logs/ directory"
fi

echo
echo "🎯 Target: Beat baseline 0.26 NRMSE"
echo "⏱️  Expected completion: ~1-2 hours from start"

