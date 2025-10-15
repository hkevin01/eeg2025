#!/bin/bash
# Simple training monitor

cd /home/kevin/Projects/eeg2025

echo "📊 Training Monitor"
echo "===================="

# Check if training is running
if ps aux | grep -q "[t]rain_foundation_cpu"; then
    echo "✅ Training is RUNNING"
    PID=$(ps aux | grep "[t]rain_foundation_cpu" | awk '{print $2}')
    echo "   PID: $PID"
else
    echo "⭕ Training is NOT running"
fi

echo ""
echo "📁 Recent Logs:"
echo "----------------"
ls -lt logs/foundation_cpu_*.log 2>/dev/null | head -3

echo ""
echo "📝 Latest Log (last 40 lines):"
echo "--------------------------------"
tail -40 logs/foundation_cpu_*.log 2>/dev/null | tail -40

echo ""
echo "💾 Checkpoints:"
echo "---------------"
ls -lht checkpoints/foundation_*.pth 2>/dev/null | head -5 || echo "No checkpoints yet"
