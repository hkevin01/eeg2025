#!/bin/bash
# Launch Challenge 2 training in tmux session

SESSION="eeg_train_c2"

# Kill existing session if it exists
tmux kill-session -t $SESSION 2>/dev/null

# Create new session
tmux new-session -d -s $SESSION

# Run training
tmux send-keys -t $SESSION "cd /home/kevin/Projects/eeg2025" C-m
tmux send-keys -t $SESSION "python scripts/training/challenge2/train_challenge2_multi_release.py 2>&1 | tee logs/training_comparison/challenge2_improved_$(date +%Y%m%d_%H%M%S).log" C-m

echo "✅ Training started in tmux session: $SESSION"
echo "📋 Attach with: tmux attach -t $SESSION"
echo "🔍 Detach with: Ctrl+b then d"
echo "📊 Monitor with: ./check_training_simple.sh"

chmod +x train_challenge2_tmux.sh
