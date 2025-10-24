#!/bin/bash
# Start training in tmux session - crash resistant!

SESSION_NAME="eeg_training"

# Kill existing session if any
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session
tmux new-session -d -s $SESSION_NAME

# Run training in the session
tmux send-keys -t $SESSION_NAME "cd /home/kevin/Projects/eeg2025" C-m
tmux send-keys -t $SESSION_NAME "python train_challenge1_advanced.py --epochs 100 --batch-size 32 --device cuda --exp-name sam_full_run 2>&1 | tee training_tmux.log" C-m

echo "✅ Training started in tmux session: $SESSION_NAME"
echo ""
echo "📋 Useful commands:"
echo "   View session:     tmux attach -t $SESSION_NAME"
echo "   Detach:           Ctrl+B then D"
echo "   Kill session:     tmux kill-session -t $SESSION_NAME"
echo "   Monitor log:      tail -f training_tmux.log"
echo "   Check GPU:        watch -n 2 rocm-smi"
echo ""
echo "🏃 Training is running in background..."
