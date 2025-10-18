#!/bin/bash

# Train challenges in tmux sessions for robustness
# This ensures training continues even if terminal disconnects

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting Training in Tmux Sessions"
echo "======================================"
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "❌ tmux is not installed. Installing..."
    sudo apt-get update && sudo apt-get install -y tmux
fi

# Kill any existing training sessions
echo "🧹 Cleaning up old sessions..."
tmux kill-session -t eeg_train_c1 2>/dev/null || true
tmux kill-session -t eeg_train_c2 2>/dev/null || true
sleep 1

# Create log directory
mkdir -p logs/training_comparison

# Start Challenge 1 in tmux
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE_C1="logs/training_comparison/challenge1_improved_${TIMESTAMP}.log"

echo ""
echo "📊 Starting Challenge 1 in tmux session 'eeg_train_c1'..."
tmux new-session -d -s eeg_train_c1 "cd $SCRIPT_DIR && python3 scripts/training/challenge1/train_challenge1_multi_release.py 2>&1 | tee $LOG_FILE_C1"

echo "✅ Challenge 1 started!"
echo "   Session: eeg_train_c1"
echo "   Log: $LOG_FILE_C1"
echo ""

# Wait a moment and verify it's running
sleep 3
if tmux has-session -t eeg_train_c1 2>/dev/null; then
    echo "✅ Challenge 1 session is running"
else
    echo "❌ Challenge 1 session failed to start"
    exit 1
fi

echo ""
echo "📋 Challenge 2 will be started manually after Challenge 1 completes"
echo "   (to avoid memory issues from loading both datasets simultaneously)"
echo ""

echo "=================================="
echo "🎯 Training Session Information"
echo "=================================="
echo ""
echo "Challenge 1:"
echo "  Session name: eeg_train_c1"
echo "  Log file:     $LOG_FILE_C1"
echo "  Attach:       tmux attach -t eeg_train_c1"
echo ""
echo "Commands:"
echo "  List sessions:    tmux ls"
echo "  Attach to C1:     tmux attach -t eeg_train_c1"
echo "  Detach:           Ctrl+B, then D"
echo "  View log:         tail -f $LOG_FILE_C1"
echo "  Kill session:     tmux kill-session -t eeg_train_c1"
echo ""
echo "Quick status:"
echo "  ./check_training_simple.sh"
echo ""
echo "=================================="
echo "✅ Challenge 1 training started in tmux!"
echo "=================================="
