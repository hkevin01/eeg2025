#!/bin/bash

echo "🎯 Challenge 1 Improved Training Monitor"
echo "========================================"
echo ""

# Check if process is running
if pgrep -f "train_c1_improved_final.py" > /dev/null; then
    echo "✅ Training is RUNNING"
    echo ""
    echo "📊 Latest log output:"
    echo "----------------------------------------"
    tail -30 logs/c1_improved_training.log
    echo "----------------------------------------"
    echo ""
    echo "📈 Progress Summary:"
    grep -E "Epoch.*Train.*Val" logs/c1_improved_training.log | tail -5 || echo "   No epochs completed yet"
    echo ""
    echo "🎯 Best Results So Far:"
    grep "BEST r!" logs/c1_improved_training.log | tail -1 || echo "   No best Pearson r yet"
    grep "BEST NRMSE!" logs/c1_improved_training.log | tail -1 || echo "   No best NRMSE yet"
else
    echo "❌ Training is NOT running"
    echo ""
    echo "📁 Last log output:"
    tail -20 logs/c1_improved_training.log
fi

echo ""
echo "⏱️  To watch in real-time: tail -f logs/c1_improved_training.log"
echo "🔄 To refresh this: ./monitor_improved_training.sh"

