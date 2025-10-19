#!/bin/bash
# Launch Challenge 2 training in tmux session (survives VS Code crashes)

SESSION_NAME="eeg_challenge2_training"
SCRIPT_NAME="train_challenge2_fast.py"
LOG_FILE="logs/training_challenge2_tmux.log"

echo "════════════════════════════════════════════════════════════════"
echo "🚀 Launching Challenge 2 Training in TMUX"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "❌ tmux not found. Installing..."
    sudo apt update && sudo apt install -y tmux
fi

# Check if session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "⚠️  Session '$SESSION_NAME' already exists!"
    echo ""
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t $SESSION_NAME"
    echo "  2. Kill and restart: tmux kill-session -t $SESSION_NAME && $0"
    echo "  3. List all sessions: tmux ls"
    echo ""
    exit 1
fi

# Check if cache files exist
echo "�� Checking cache files..."
CACHE_FILES=(
    "data/cached/challenge2_R1_windows.h5"
    "data/cached/challenge2_R2_windows.h5"
    "data/cached/challenge2_R3_windows.h5"
    "data/cached/challenge2_R4_windows.h5"
    "data/cached/challenge2_R5_windows.h5"
)

MISSING=0
for file in "${CACHE_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "   ⚠️  Missing: $file"
        MISSING=$((MISSING + 1))
    else
        SIZE=$(du -h "$file" | cut -f1)
        echo "   ✅ $file ($SIZE)"
    fi
done

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "⚠️  Warning: $MISSING cache files missing"
    echo "   Continuing anyway - will use available files"
    echo ""
fi

# Check if training script exists
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "❌ Training script not found: $SCRIPT_NAME"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Create tmux session
echo "Creating tmux session '$SESSION_NAME'..."
tmux new-session -d -s $SESSION_NAME

# Set up the session
tmux send-keys -t $SESSION_NAME "cd $(pwd)" C-m
tmux send-keys -t $SESSION_NAME "clear" C-m
tmux send-keys -t $SESSION_NAME "echo '════════════════════════════════════════════════════════════════'" C-m
tmux send-keys -t $SESSION_NAME "echo '🧠 EEG Challenge 2 Training - TMUX Session'" C-m
tmux send-keys -t $SESSION_NAME "echo '════════════════════════════════════════════════════════════════'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo 'Session: $SESSION_NAME'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Log: $LOG_FILE'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Started: \$(date)'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo 'Commands:'" C-m
tmux send-keys -t $SESSION_NAME "echo '  Ctrl+B, D  - Detach (keep running)'" C-m
tmux send-keys -t $SESSION_NAME "echo '  Ctrl+C     - Stop training'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo '════════════════════════════════════════════════════════════════'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "sleep 2" C-m

# Start training with logging
tmux send-keys -t $SESSION_NAME "python3 $SCRIPT_NAME 2>&1 | tee $LOG_FILE" C-m

echo ""
echo "✅ Training session started!"
echo ""
echo "Session name: $SESSION_NAME"
echo "Log file: $LOG_FILE"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 TMUX COMMANDS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Attach to session (watch live):"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "Detach from session (keep running):"
echo "  Press: Ctrl+B, then D"
echo ""
echo "Check if running:"
echo "  tmux ls"
echo "  ps aux | grep $SCRIPT_NAME"
echo ""
echo "View log:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Kill session:"
echo "  tmux kill-session -t $SESSION_NAME"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 MONITORING:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Database queries:"
echo "  sqlite3 data/metadata.db 'SELECT * FROM training_runs ORDER BY run_id DESC LIMIT 1;'"
echo "  sqlite3 data/metadata.db 'SELECT epoch, train_loss, val_loss FROM epoch_history WHERE run_id = 1;'"
echo ""
echo "System status:"
echo "  nvidia-smi  # GPU usage"
echo "  htop        # CPU/Memory"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "💡 TIP: Training will continue even if VS Code crashes!"
echo ""

