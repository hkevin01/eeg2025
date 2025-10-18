#!/bin/bash
# Train both challenges sequentially in tmux

SESSION="eeg_both_challenges"

echo "════════════════════════════════════════════════════════════════════"
echo "   🚀 TRAINING BOTH CHALLENGES"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Challenge 1: Response Time (TCN) - Already complete!"
echo "Challenge 2: Externalizing (TCN) - Will train now"
echo ""
echo "Training will run in tmux session: $SESSION"
echo ""

# Kill existing session if it exists
tmux kill-session -t $SESSION 2>/dev/null

# Create new session
tmux new-session -d -s $SESSION

# Send training commands
tmux send-keys -t $SESSION "cd $(pwd)" C-m
tmux send-keys -t $SESSION "echo '🎯 Starting Challenge 2 Training...'" C-m
tmux send-keys -t $SESSION "python3 -u scripts/train_challenge2_tcn.py 2>&1 | tee logs/train_challenge2_tcn_$(date +%Y%m%d_%H%M%S).log" C-m

echo "✅ Training started in tmux session!"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "   📋 COMMANDS"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "View training:"
echo "   tmux attach -t $SESSION"
echo "   (Press Ctrl+B then D to detach)"
echo ""
echo "Check logs:"
echo "   ls -lht logs/train_challenge2*.log | head -3"
echo "   tail -f logs/train_challenge2_tcn_*.log"
echo ""
echo "Stop training:"
echo "   tmux kill-session -t $SESSION"
echo ""
echo "════════════════════════════════════════════════════════════════════"
