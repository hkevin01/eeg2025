#!/bin/bash
# Monitor training progress in tmux session

echo "🔍 EEG Training Monitor"
echo "======================"
echo ""
echo "📊 Latest log output:"
echo "---"
tail -30 logs/train_all_rsets_20251028_145812.log
echo ""
echo "---"
echo ""
echo "⏱️  Training time: $(ps -o etime= -p $(pgrep -f train_c1_all_rsets.py | head -1) 2>/dev/null || echo 'Not running')"
echo "💾 Memory: $(ps -o rss= -p $(pgrep -f train_c1_all_rsets.py | head -1) 2>/dev/null | awk '{print $1/1024 " MB"}' || echo 'N/A')"
echo ""
echo "Commands:"
echo "  Watch live:    tmux attach -t eeg_training"
echo "  Detach:        Ctrl+B then D"
echo "  Kill training: tmux kill-session -t eeg_training"
echo "  This monitor:  bash monitor_training.sh"
