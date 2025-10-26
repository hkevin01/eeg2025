#!/bin/bash

# Start Enhanced C1 Training in tmux with GPU Safe Loading
SESSION_NAME="c1_enhanced"

echo "================================================================================================"
echo "🚀 Starting Enhanced C1 Training in tmux"
echo "================================================================================================"
echo "Strategy: Load data with GPU hidden, then expose for training"
echo "Session: $SESSION_NAME"
echo "================================================================================================"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session
tmux new-session -d -s $SESSION_NAME

# Setup ROCm SDK environment in tmux
tmux send-keys -t $SESSION_NAME "cd /home/kevin/Projects/eeg2025" C-m
tmux send-keys -t $SESSION_NAME "export ROCM_PATH='/opt/rocm_sdk_612'" C-m
tmux send-keys -t $SESSION_NAME "export LD_LIBRARY_PATH='/opt/rocm_sdk_612/lib:/opt/rocm_sdk_612/lib64:/opt/rocm_sdk_612/lib/python3.11/site-packages/torch/lib:\$LD_LIBRARY_PATH'" C-m
tmux send-keys -t $SESSION_NAME "export PYTHONPATH='/opt/rocm_sdk_612/lib/python3.11/site-packages:\$PYTHONPATH'" C-m
tmux send-keys -t $SESSION_NAME "export PATH='/opt/rocm_sdk_612/bin:\$PATH'" C-m

# ROCm GPU configuration (will be enabled after data loading)
tmux send-keys -t $SESSION_NAME "export HSA_OVERRIDE_GFX_VERSION=10.3.0" C-m
tmux send-keys -t $SESSION_NAME "export PYTORCH_ROCM_ARCH='gfx1030'" C-m
tmux send-keys -t $SESSION_NAME "export HSA_XNACK=0" C-m
tmux send-keys -t $SESSION_NAME "export HSA_FORCE_FINE_GRAIN_PCIE=1" C-m
tmux send-keys -t $SESSION_NAME "export AMD_SERIALIZE_KERNEL=3" C-m

# MNE configuration
tmux send-keys -t $SESSION_NAME "export MNE_USE_CUDA=false" C-m
tmux send-keys -t $SESSION_NAME "export QT_QPA_PLATFORM=offscreen" C-m
tmux send-keys -t $SESSION_NAME "export MPLBACKEND=Agg" C-m

# Start training
tmux send-keys -t $SESSION_NAME "/opt/rocm_sdk_612/bin/python3 train_c1_enhanced_safe.py --max_subjects 30 --batch_size 8 --epochs 30 --early_stopping 15 2>&1 | tee training_c1_enhanced_safe.log" C-m

echo ""
echo "✅ Training started in tmux session '$SESSION_NAME'"
echo ""
echo "Monitor with:"
echo "  tmux attach -t $SESSION_NAME    # Attach to session"
echo "  tail -f training_c1_enhanced_safe.log  # Watch log"
echo "  tmux kill-session -t $SESSION_NAME     # Stop training"
echo ""
echo "================================================================================================"
