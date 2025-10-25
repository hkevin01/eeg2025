#!/bin/bash

# SAM Training for Challenge 2 - EEGNeX
# Baseline: Oct 24 model (score 1.0087)
# Target: < 0.9

echo "🧠 Starting Challenge 2 SAM Training..."
echo "Architecture: EEGNeX (fallback to SimplifiedTCN)"
echo "Baseline: Oct 24 (1.0087)"
echo "Target: < 0.9"
echo "GPU: Reduced batch size (16) to avoid OOM"
echo ""

# Create tmux session
SESSION_NAME="sam_c2"
LOG_FILE="training_sam_c2.log"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new session with C2 training
tmux new-session -d -s $SESSION_NAME "cd /home/kevin/Projects/eeg2025 && export PYTORCH_ROCM_ARCH=gfx1010 && export HSA_OVERRIDE_GFX_VERSION=10.1.0 && python train_c2_sam_eegnex.py --data-dir data/ds005505-bdf --epochs 50 --batch-size 16 --lr 1e-3 --rho 0.05 --device cuda 2>&1 | tee $LOG_FILE"

echo "✅ Training started in tmux session: $SESSION_NAME"
echo ""
echo "Commands:"
echo "  View training: tmux attach -t $SESSION_NAME"
echo "  Detach: Ctrl+B, then D"
echo "  Check log: tail -f $LOG_FILE"
echo "  Kill training: tmux kill-session -t $SESSION_NAME"
echo ""
echo "Note: Using batch_size=16 (reduced from 32) to share GPU with C1 training"
echo ""
