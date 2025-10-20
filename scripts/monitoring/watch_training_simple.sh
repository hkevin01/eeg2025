#!/bin/bash
# Simple training monitor - shows key metrics

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║            📊 Training Monitor - Live View                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Find latest log file
LATEST_LOG=$(ls -t logs/training_live_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ No training log found"
    exit 1
fi

echo "📝 Monitoring: $LATEST_LOG"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Show GPU status
if command -v rocm-smi &> /dev/null; then
    echo "🎮 GPU Status:"
    rocm-smi --showuse | grep -A 1 "GPU\|%" | head -5
    echo ""
fi

# Show training progress  (last 30 lines)
echo "📈 Training Progress (last 30 lines):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tail -30 "$LATEST_LOG"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check for errors
if grep -i "error\|exception\|failed" "$LATEST_LOG" | tail -5 > /dev/null 2>&1; then
    echo "⚠️  Recent errors/warnings:"
    grep -i "error\|exception\|failed" "$LATEST_LOG" | tail -5
fi

# Show process info
TRAIN_PID=$(ps aux | grep "python.*train_challenge2" | grep -v grep | awk '{print $2}')
if [ -n "$TRAIN_PID" ]; then
    echo "✅ Training process running (PID: $TRAIN_PID)"
else
    echo "❌ No training process found"
fi
