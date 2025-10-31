#!/bin/bash
# Monitor ensemble training in tmux

echo "🔍 Ensemble Training Monitor"
echo "=============================="
echo

# Check if running
if ps aux | grep -q "[p]ython train_c1_ensemble"; then
    echo "✅ Training is RUNNING"
    ps aux | grep "[p]ython train_c1_ensemble" | awk '{print "   PID:", $2, "CPU:", $3"%, RAM:", $4"%"}'
else
    echo "❌ Training is NOT running"
    exit 1
fi

echo
echo "📊 Latest Output from tmux:"
echo "=============================="
tmux capture-pane -t ensemble_training -p -S -50 | tail -30

echo
echo "=============================="
echo "Commands:"
echo "  Attach:  tmux attach -t ensemble_training"
echo "  Detach:  Ctrl+B then D"
echo "  Kill:    tmux kill-session -t ensemble_training"
echo "  Refresh: ./watch_ensemble.sh"
