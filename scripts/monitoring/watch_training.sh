#!/bin/bash
# Quick monitor script for training

echo "🔍 Training Status Check"
echo "========================"
echo ""

# Check if running
if ps aux | grep -q "[p]ython train_c1_tonight.py"; then
    echo "✅ Training is RUNNING"
    echo ""
    
    # Show resource usage
    echo "📊 Resource Usage:"
    ps aux | grep "[p]ython train_c1_tonight.py" | awk '{printf "   CPU: %s%%  RAM: %s%%  Time: %s\n", $3, $4, $10}'
    echo ""
    
    # Show latest output
    echo "📝 Latest Output (last 25 lines):"
    echo "---"
    tmux capture-pane -t eeg_training -p | tail -25
else
    echo "❌ Training is NOT running"
    echo ""
    echo "Last log output:"
    tail -30 training_c1_tonight.log 2>/dev/null || echo "No log file found"
fi

echo ""
echo "💡 Commands:"
echo "   Watch live:  tmux attach -t eeg_training"
echo "   Detach:      Ctrl+B, then D"
echo "   Kill:        tmux kill-session -t eeg_training"
