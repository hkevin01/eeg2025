#!/bin/bash
# Independent Training Launcher - Survives VS Code crashes
# Uses tmux for persistent sessions

set -e

echo "════════════════════════════════════════════════════════════════════"
echo "   🚀 INDEPENDENT TRAINING LAUNCHER"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Configuration
SESSION_NAME="eeg_training"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRAIN_SCRIPT="$SCRIPT_DIR/train_tcn_competition_data.py"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_independent_${TIMESTAMP}.log"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "❌ tmux not found! Installing..."
    echo "   Run: sudo apt-get install tmux"
    echo ""
    echo "   Or use nohup fallback:"
    echo "   nohup python3 -u $TRAIN_SCRIPT > $LOG_FILE 2>&1 &"
    exit 1
fi

# Kill existing session if it exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "⚠️  Found existing training session"
    echo "   Killing old session..."
    tmux kill-session -t $SESSION_NAME
    sleep 2
fi

# Create logs directory
mkdir -p "$LOG_DIR"

echo "📝 Configuration:"
echo "   Session: $SESSION_NAME"
echo "   Script: $(basename $TRAIN_SCRIPT)"
echo "   Log: $(basename $LOG_FILE)"
echo ""

# Create new tmux session in detached mode
echo "🚀 Starting training in tmux session..."
tmux new-session -d -s $SESSION_NAME -c "$PROJECT_DIR"

# Send commands to the tmux session
tmux send-keys -t $SESSION_NAME "cd $PROJECT_DIR" C-m
tmux send-keys -t $SESSION_NAME "echo '🧠 Starting Competition Training...'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Session: $SESSION_NAME'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Log: $LOG_FILE'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "python3 -u $TRAIN_SCRIPT 2>&1 | tee $LOG_FILE" C-m

# Wait a moment for startup
sleep 3

# Check if it's running
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "✅ Training started successfully!"
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "   📋 COMMANDS"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
    echo "View live training:"
    echo "   tmux attach -t $SESSION_NAME"
    echo "   (Press Ctrl+B then D to detach without stopping)"
    echo ""
    echo "View log file:"
    echo "   tail -f $LOG_FILE"
    echo ""
    echo "Monitor progress:"
    echo "   ./scripts/monitoring/monitor_training_enhanced.sh"
    echo ""
    echo "Stop training:"
    echo "   tmux kill-session -t $SESSION_NAME"
    echo ""
    echo "Check if running:"
    echo "   tmux ls"
    echo "   ps aux | grep train_tcn"
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
    echo "✨ Training is now independent of VS Code!"
    echo "   You can safely:"
    echo "   - Close VS Code"
    echo "   - Close terminal"
    echo "   - Disconnect SSH"
    echo "   - Restart your computer (survives until shutdown)"
    echo ""
    echo "Training will continue until:"
    echo "   - Model converges (early stopping)"
    echo "   - Max epochs reached (100)"
    echo "   - You manually stop it"
    echo ""
else
    echo "❌ Failed to start training session"
    exit 1
fi
