#!/bin/bash

echo "🔍 Quick Training Status Check"
echo "=============================="
echo ""

# Check running processes
PROCS=$(ps aux | grep -E "train_challenge" | grep -v grep)
if [ -z "$PROCS" ]; then
    echo "❌ No training processes running"
    echo ""
    echo "To restart Challenge 1:"
    echo "  cd /home/kevin/Projects/eeg2025"
    echo "  nohup python3 scripts/training/challenge1/train_challenge1_multi_release.py > logs/training_comparison/challenge1_improved_\$(date +%Y%m%d_%H%M%S).log 2>&1 &"
else
    echo "✅ Training processes running:"
    echo "$PROCS" | awk '{print "   PID:", $2, "| CPU:", $3"%", "| Mem:", $4"%", "| Command:", $11, $12, $13}'
    echo ""
    
    # Find latest log
    LATEST_LOG=$(ls -t logs/training_comparison/challenge1_improved_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "📊 Latest Challenge 1 log: $LATEST_LOG"
        echo ""
        
        # Check for epoch progress
        EPOCHS=$(grep -E "Epoch [0-9]+/[0-9]+" "$LATEST_LOG" | tail -5)
        if [ -n "$EPOCHS" ]; then
            echo "📈 Recent training epochs:"
            echo "$EPOCHS" | sed 's/^/   /'
        else
            echo "⏳ Still in data loading/preprocessing phase"
            echo "   (Check log with: tail -f $LATEST_LOG)"
        fi
        
        echo ""
        
        # Check for best validation score
        BEST_VAL=$(grep "Best Val NRMSE" "$LATEST_LOG" | tail -1)
        if [ -n "$BEST_VAL" ]; then
            echo "🏆 Best validation score so far:"
            echo "   $BEST_VAL"
        fi
    fi
fi

echo ""
echo "📂 Model weights:"
ls -lh weights_challenge_*_multi_release.pt 2>/dev/null | awk '{print "   ", $9, "-", $5, "-", $6, $7, $8}' || echo "   No weights found yet"

echo ""
echo "💡 Useful commands:"
echo "   Watch live:  tail -f logs/training_comparison/challenge1_improved_*.log"
echo "   Check epochs: grep 'Epoch' logs/training_comparison/challenge1_improved_*.log | tail -10"
echo "   Monitor this: watch -n 30 './check_training_simple.sh'"
