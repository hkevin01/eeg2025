#!/bin/bash
# Start Challenge 2 training in tmux (independent of VS Code)

SESSION_NAME="challenge2_training"

# Check if session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "⚠️  Training session '$SESSION_NAME' already exists"
    echo "Options:"
    echo "  1. Attach: tmux attach -t $SESSION_NAME"
    echo "  2. Kill and restart: tmux kill-session -t $SESSION_NAME && $0"
    exit 1
fi

# Create new tmux session
echo "🚀 Starting Challenge 2 training in tmux session: $SESSION_NAME"

tmux new-session -d -s $SESSION_NAME

# Send commands to the session
tmux send-keys -t $SESSION_NAME "cd /home/kevin/Projects/eeg2025" C-m
tmux send-keys -t $SESSION_NAME "echo '🧠 Challenge 2 Training Started: \$(date)'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Session: $SESSION_NAME'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "python3 train_challenge2_fast.py 2>&1 | tee logs/training_challenge2_fast.log" C-m

echo "✅ Training started in tmux session: $SESSION_NAME"
echo ""
echo "📊 Monitor training:"
echo "  • Attach to session: tmux attach -t $SESSION_NAME"
echo "  • View log: tail -f logs/training_challenge2_fast.log"
echo "  • Check database: sqlite3 data/metadata.db 'SELECT * FROM training_runs;'"
echo ""
echo "🔌 Detach from session: Ctrl+B, then D"
echo "📋 List sessions: tmux list-sessions"
