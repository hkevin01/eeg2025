#!/bin/bash
# Monitor training progress in tmux session

echo "🔍 EEG Training Monitor"
echo "======================="
echo ""

# Check if tmux session exists
if ! tmux has-session -t eeg_training 2>/dev/null; then
    echo "❌ Training session 'eeg_training' not found!"
    echo ""
    echo "Available sessions:"
    tmux list-sessions
    exit 1
fi

echo "✅ Training session 'eeg_training' is running"
echo ""

# Show latest log file
LATEST_LOG=$(ls -t logs/training_subject_aware_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "📄 Latest log: $LATEST_LOG"
    echo ""
    echo "Last 30 lines:"
    echo "----------------------------------------"
    tail -30 "$LATEST_LOG"
    echo "----------------------------------------"
    echo ""
    echo "💡 To attach to session: tmux attach -t eeg_training"
    echo "💡 To detach from session: Press Ctrl+B then D"
    echo "💡 To kill session: tmux kill-session -t eeg_training"
else
    echo "⚠️  No log file found yet"
fi
