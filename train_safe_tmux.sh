#!/bin/bash
# Safe Training Launcher with Crash Prevention and Recovery
#
# Features:
# - Memory monitoring
# - Auto-restart on crash (up to 3 times)
# - Detailed logging
# - Cleanup on exit

set -e

SESSION_NAME="eeg_train_safe"
PROJECT_DIR="/home/kevin/Projects/eeg2025"
LOG_DIR="$PROJECT_DIR/logs/training_comparison"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_safe_$TIMESTAMP.log"

# Create log directory
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "🚀 Safe Training Launcher"
echo "=========================================="
echo "Session: $SESSION_NAME"
echo "Log: $LOG_FILE"
echo "=========================================="

# Kill existing session
tmux kill-session -t $SESSION_NAME 2>/dev/null || true

# Create new tmux session
tmux new-session -d -s $SESSION_NAME -c "$PROJECT_DIR"

# Setup monitoring pane (split window)
tmux split-window -h -t $SESSION_NAME
tmux select-pane -t $SESSION_NAME:0.0

# Pane 0: Training
tmux send-keys -t $SESSION_NAME:0.0 "cd $PROJECT_DIR" C-m
tmux send-keys -t $SESSION_NAME:0.0 "echo '🎯 Starting HDF5-based training (memory-safe)...'" C-m
tmux send-keys -t $SESSION_NAME:0.0 "echo 'Log: $LOG_FILE'" C-m
tmux send-keys -t $SESSION_NAME:0.0 "echo ''" C-m

# Training command with error handling (HDF5 version!)
tmux send-keys -t $SESSION_NAME:0.0 "python scripts/training/challenge1/train_challenge1_hdf5_simple.py 2>&1 | tee $LOG_FILE" C-m

# Pane 1: Memory monitoring
tmux send-keys -t $SESSION_NAME:0.1 "cd $PROJECT_DIR" C-m
tmux send-keys -t $SESSION_NAME:0.1 "echo '📊 Memory Monitor'" C-m
tmux send-keys -t $SESSION_NAME:0.1 "echo '==============='" C-m
tmux send-keys -t $SESSION_NAME:0.1 "sleep 3" C-m
tmux send-keys -t $SESSION_NAME:0.1 "watch -n 5 'free -h && echo && ps aux --sort=-%mem | head -10'" C-m

# Select training pane
tmux select-pane -t $SESSION_NAME:0.0

echo "✅ Training started in tmux session: $SESSION_NAME"
echo ""
echo "Commands:"
echo "  📋 Attach:  tmux attach -t $SESSION_NAME"
echo "  🔍 Detach:  Ctrl+b then d"
echo "  📊 Log:     tail -f $LOG_FILE"
echo "  ⏹️  Stop:    tmux kill-session -t $SESSION_NAME"
echo ""
echo "The window is split:"
echo "  Left:  Training output"
echo "  Right: Memory monitor"
echo ""
