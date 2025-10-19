#!/bin/bash
# Restart Challenge 2 Training After Power Surge
# Created: October 19, 2025

set -e

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║            🔄 RESTARTING CHALLENGE 2 TRAINING AFTER POWER SURGE              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if any training processes are running
echo "📋 Step 1: Checking for existing processes..."
if pgrep -f "train_challenge2" > /dev/null; then
    echo "⚠️  Found existing training process. Stopping it..."
    pkill -f "train_challenge2" || true
    sleep 2
fi

if pgrep -f "watchdog_challenge2" > /dev/null; then
    echo "⚠️  Found existing watchdog process. Stopping it..."
    pkill -f "watchdog_challenge2" || true
    sleep 2
fi

echo "✅ No conflicting processes running"
echo ""

# Create logs directory if needed
echo "📋 Step 2: Preparing log directory..."
mkdir -p logs
echo "✅ Logs directory ready"
echo ""

# Backup old logs
echo "📋 Step 3: Backing up previous logs..."
if [ -f "logs/challenge2_correct_training.log" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    cp logs/challenge2_correct_training.log "logs/challenge2_correct_training_backup_${TIMESTAMP}.log"
    echo "✅ Previous log backed up to: challenge2_correct_training_backup_${TIMESTAMP}.log"
else
    echo "ℹ️  No previous log to backup"
fi
echo ""

# Check Python dependencies
echo "📋 Step 4: Verifying Python environment..."
python3 -c "import torch; import braindecode; import numpy; print('✅ Core dependencies OK')" || {
    echo "❌ Missing dependencies. Please run: pip install -r requirements.txt"
    exit 1
}
echo ""

# Check if data directory exists
echo "📋 Step 5: Verifying data directory..."
if [ ! -d "data" ]; then
    echo "❌ Data directory not found. Please ensure HBN dataset is in ./data/"
    exit 1
fi
echo "✅ Data directory found"
echo ""

# Check if training script exists
echo "📋 Step 6: Verifying training script..."
if [ ! -f "train_challenge2_correct.py" ]; then
    echo "❌ Training script not found: train_challenge2_correct.py"
    exit 1
fi
echo "✅ Training script found"
echo ""

# Start training in background
echo "📋 Step 7: Starting training process..."
nohup python3 train_challenge2_correct.py > logs/challenge2_correct_training.log 2>&1 &
TRAIN_PID=$!
sleep 3

# Verify training started
if ps -p $TRAIN_PID > /dev/null; then
    echo "✅ Training started successfully!"
    echo "   PID: $TRAIN_PID"
else
    echo "❌ Training failed to start. Check logs/challenge2_correct_training.log"
    exit 1
fi
echo ""

# Start watchdog
echo "📋 Step 8: Starting watchdog monitor..."
if [ -f "watchdog_challenge2.sh" ]; then
    nohup ./watchdog_challenge2.sh > logs/watchdog_output.log 2>&1 &
    WATCHDOG_PID=$!
    sleep 2
    
    if ps -p $WATCHDOG_PID > /dev/null; then
        echo "✅ Watchdog started successfully!"
        echo "   PID: $WATCHDOG_PID"
    else
        echo "⚠️  Watchdog failed to start (non-critical)"
    fi
else
    echo "⚠️  Watchdog script not found (training will continue without monitoring)"
fi
echo ""

# Display status
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                        ✅ TRAINING RESTARTED SUCCESSFULLY                     ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Process Information:"
echo "   Training PID: $TRAIN_PID"
echo "   Training Log: logs/challenge2_correct_training.log"
[ ! -z "$WATCHDOG_PID" ] && echo "   Watchdog PID: $WATCHDOG_PID"
echo ""
echo "📋 Monitoring Commands:"
echo "   Quick status:  ./quick_training_status.sh"
echo "   Full monitor:  ./monitor_challenge2.sh"
echo "   Live tail:     tail -f logs/challenge2_correct_training.log"
echo "   Watchdog:      ./manage_watchdog.sh status"
echo ""
echo "⏳ Training will take several hours/days. Watchdog will alert on issues."
echo ""

