#!/bin/bash

# SAM Training for Challenge 1 - CompactCNN
# Baseline: Oct 16 model (score 1.0015)
# Target: < 0.9

echo "🧠 Starting Challenge 1 SAM Training..."
echo "Architecture: CompactCNN"
echo "Baseline: Oct 16 (1.0015)"
echo "Target: < 0.9"
echo ""

# Create tmux session
SESSION_NAME="sam_c1"
LOG_FILE="training_sam_c1.log"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new session
tmux new-session -d -s $SESSION_NAME

# Send commands to session
tmux send-keys -t $SESSION_NAME "cd /home/kevin/Projects/eeg2025" C-m
tmux send-keys -t $SESSION_NAME "source venv_rocm57/bin/activate 2>/dev/null || echo 'No venv'" C-m
tmux send-keys -t $SESSION_NAME "export PYTORCH_ROCM_ARCH=gfx1010" C-m
tmux send-keys -t $SESSION_NAME "export HSA_OVERRIDE_GFX_VERSION=10.1.0" C-m
tmux send-keys -t $SESSION_NAME "python train_c1_sam_compactcnn.py --epochs 100 --batch-size 32 --lr 1e-3 --rho 0.05 --device cuda 2>&1 | tee $LOG_FILE" C-m

echo "✅ Training started in tmux session: $SESSION_NAME"
echo ""
echo "Commands:"
echo "  View training: tmux attach -t $SESSION_NAME"
echo "  Detach: Ctrl+B, then D"
echo "  Check log: tail -f $LOG_FILE"
echo "  Kill training: tmux kill-session -t $SESSION_NAME"
echo ""
