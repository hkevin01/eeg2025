#!/bin/bash
echo "🔍 Challenge 1 Training Status"
echo "=" | awk '{for(i=1;i<=60;i++) printf "="; printf "\n"}'
echo

# Check if process is running
if ps aux | grep "train_challenge1_enhanced.py" | grep -v grep > /dev/null; then
    echo "✅ Training process IS RUNNING"
    ps aux | grep "train_challenge1_enhanced.py" | grep -v grep | awk '{printf "   CPU: %.1f%% | RAM: %.1f GB\n", $3, $6/1024/1024}'
    echo
else
    echo "❌ Training process NOT running"
    echo
fi

# Show last 30 lines of log
echo "📝 Last 30 lines of log:"
echo "---"
tail -30 training_challenge1.log 2>/dev/null || echo "No log file found yet"
echo
echo "---"
echo "💡 To watch live: tail -f training_challenge1.log"
