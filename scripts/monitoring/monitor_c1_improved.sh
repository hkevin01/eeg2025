#!/bin/bash
# Monitor Challenge 1 Improved Training

LOG_FILE=$(ls -t logs/challenge1/training_improved_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ No training log found!"
    exit 1
fi

echo "📊 Challenge 1 Improved Training Monitor"
echo "=========================================="
echo "Log file: $LOG_FILE"
echo ""

# Check if training is still running
if ps aux | grep -E "[p]ython.*train_challenge1_improved" > /dev/null; then
    echo "✅ Training is RUNNING"
else
    echo "⏹️  Training has stopped"
fi

echo ""
echo "📈 Latest metrics:"
echo "===================="
tail -30 "$LOG_FILE" | grep -E "Epoch|Train Loss|Val Loss|NRMSE|Best|Early stopping"

echo ""
echo "📊 Progress summary:"
echo "===================="
grep -E "Epoch [0-9]+/100" "$LOG_FILE" | tail -5

echo ""
echo "🔍 Full log command:"
echo "   tail -f $LOG_FILE"
