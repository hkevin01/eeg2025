#!/bin/bash
# Script to start training in tmux session

SESSION_NAME="eeg_training"

# Check if tmux session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "⚠️  Tmux session '$SESSION_NAME' already exists!"
    echo "   Options:"
    echo "   1. Attach: tmux attach -t $SESSION_NAME"
    echo "   2. Kill and restart: tmux kill-session -t $SESSION_NAME && $0"
    exit 1
fi

# Create new tmux session
echo "🚀 Creating tmux session: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME

# Send training command to tmux session
echo "📊 Starting training in tmux..."
tmux send-keys -t $SESSION_NAME "cd /home/kevin/Projects/eeg2025" C-m
tmux send-keys -t $SESSION_NAME "python train_challenge1_advanced.py --epochs 100 --batch-size 32 --device cuda --exp-name sam_full_run 2>&1 | tee training_tmux.log" C-m

echo ""
echo "✅ Training started in tmux session: $SESSION_NAME"
echo ""
echo "📋 Useful commands:"
echo "   Attach to session:     tmux attach -t $SESSION_NAME"
echo "   Detach from session:   Ctrl+B then D"
echo "   Kill session:          tmux kill-session -t $SESSION_NAME"
echo "   List sessions:         tmux ls"
echo "   Monitor log:           tail -f training_tmux.log"
echo ""
echo "🔍 The training will continue even if you close VSCode or disconnect!"

