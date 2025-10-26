#!/bin/bash

# Launch improved Challenge 1 training in tmux

SESSION="c1_improved_r091"

# Kill existing session if it exists
tmux has-session -t $SESSION 2>/dev/null && tmux kill-session -t $SESSION

# Create new tmux session
tmux new-session -d -s $SESSION

# Send commands to tmux
tmux send-keys -t $SESSION "cd /home/kevin/Projects/eeg2025" C-m
tmux send-keys -t $SESSION "source venv_cpu/bin/activate" C-m
tmux send-keys -t $SESSION "export OMP_NUM_THREADS=6" C-m
tmux send-keys -t $SESSION "export MKL_NUM_THREADS=6" C-m
tmux send-keys -t $SESSION "echo '🚀 Starting improved Challenge 1 training...'" C-m
tmux send-keys -t $SESSION "echo 'Target: Pearson r ≥ 0.91'" C-m
tmux send-keys -t $SESSION "echo ''" C-m
tmux send-keys -t $SESSION "python3 training/train_c1_improved_final.py --epochs 50 --batch-size 32 --lr 0.001 --device cpu --exp-name improved_r091 2>&1 | tee logs/c1_improved_$(date +%Y%m%d_%H%M%S).log" C-m

echo "✅ Launched training in tmux session: $SESSION"
echo ""
echo "📊 To monitor:"
echo "   tmux attach -t $SESSION"
echo ""
echo "📁 Logs:"
echo "   tail -f logs/c1_improved_*.log"
echo ""
echo "⏱️  Estimated time: 2-3 hours"

