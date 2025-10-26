#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║          🧠 EEG Challenge Training Monitor                           ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if training is running
PID=$(ps aux | grep "train_c1_sam_simple" | grep -v grep | awk '{print $2}')

if [ -z "$PID" ]; then
    echo "❌ No training process found!"
    echo ""
    echo "Last 20 lines of log:"
    tail -20 logs/training_20251026/c1_training_v2.log
    exit 1
fi

echo "✅ Training is RUNNING (PID: $PID)"
echo ""

# Show CPU usage
echo "�� CPU Usage:"
ps -p $PID -o %cpu,%mem,etime,cmd | tail -n +2
echo ""

# Show recent log lines
echo "📊 Recent Training Output:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tail -15 logs/training_20251026/c1_training_v2.log
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📝 To follow training in real-time:"
echo "   tail -f logs/training_20251026/c1_training_v2.log"
echo ""
echo "⏹  To stop training:"
echo "   kill $PID"
echo ""
