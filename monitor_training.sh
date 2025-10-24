#!/bin/bash
# Monitor training progress

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                        🔍 TRAINING MONITOR                                    ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if tmux session exists
if tmux has-session -t eeg_training 2>/dev/null; then
    echo "✅ Tmux session 'eeg_training' is RUNNING"
else
    echo "❌ Tmux session 'eeg_training' is NOT RUNNING"
    exit 1
fi

echo ""
echo "📊 Latest training output:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tail -20 training_tmux.log
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 Commands:"
echo "   Watch live:       tail -f training_tmux.log"
echo "   Attach to session: tmux attach -t eeg_training"
echo "   Check GPU:        rocm-smi"
echo "   Stop training:    tmux kill-session -t eeg_training"
