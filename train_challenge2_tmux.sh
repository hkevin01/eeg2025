#!/bin/bash

# Start Challenge 2 training in tmux (run this after Challenge 1 completes)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting Challenge 2 in Tmux"
echo "================================"
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "❌ tmux is not installed"
    exit 1
fi

# Kill any existing Challenge 2 session
echo "🧹 Cleaning up old Challenge 2 session..."
tmux kill-session -t eeg_train_c2 2>/dev/null || true
sleep 1

# Create log directory
mkdir -p logs/training_comparison

# Start Challenge 2 in tmux
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE_C2="logs/training_comparison/challenge2_improved_${TIMESTAMP}.log"

echo ""
echo "📊 Starting Challenge 2 in tmux session 'eeg_train_c2'..."
tmux new-session -d -s eeg_train_c2 "cd $SCRIPT_DIR && python3 scripts/training/challenge2/train_challenge2_multi_release.py 2>&1 | tee $LOG_FILE_C2"

echo "✅ Challenge 2 started!"
echo "   Session: eeg_train_c2"
echo "   Log: $LOG_FILE_C2"
echo ""

# Wait and verify
sleep 3
if tmux has-session -t eeg_train_c2 2>/dev/null; then
    echo "✅ Challenge 2 session is running"
else
    echo "❌ Challenge 2 session failed to start"
    exit 1
fi

echo ""
echo "=================================="
echo "🎯 Challenge 2 Information"
echo "=================================="
echo ""
echo "Session name: eeg_train_c2"
echo "Log file:     $LOG_FILE_C2"
echo "Attach:       tmux attach -t eeg_train_c2"
echo ""
echo "Commands:"
echo "  Attach to C2:     tmux attach -t eeg_train_c2"
echo "  Detach:           Ctrl+B, then D"
echo "  View log:         tail -f $LOG_FILE_C2"
echo "  Kill session:     tmux kill-session -t eeg_train_c2"
echo ""
echo "=================================="
echo "✅ Challenge 2 training started!"
echo "=================================="
