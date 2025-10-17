#!/bin/bash
# Start training completely independent of terminal/VSCode
# Will continue even if terminal closes or VSCode crashes

cd /home/kevin/Projects/eeg2025
source venv/bin/activate

# Kill any existing
pkill -9 -f "train_challenge" 2>/dev/null
sleep 2

echo "🚀 Starting INDEPENDENT training (immune to terminal/VSCode crashes)"
echo "======================================================================"

# Start Challenge 1 - completely detached
nohup python -u scripts/train_challenge1_robust_gpu.py \
  > logs/train_c1_robust_hybrid.log 2>&1 &
C1_PID=$!
disown $C1_PID

sleep 2

# Start Challenge 2 - completely detached  
nohup python -u scripts/train_challenge2_robust_gpu.py \
  > logs/train_c2_robust_hybrid.log 2>&1 &
C2_PID=$!
disown $C2_PID

sleep 3

echo "✅ Training started in background (independent mode)"
echo "   Challenge 1 PID: $C1_PID"
echo "   Challenge 2 PID: $C2_PID"
echo ""
echo "These processes will continue even if:"
echo "   - Terminal closes"
echo "   - VSCode crashes"
echo "   - SSH disconnects"
echo ""
echo "Check status: ./check_training_status.sh"
echo "Monitor: tail -f logs/train_c1_robust_hybrid.log"
echo "======================================================================"

# Verify they're running
sleep 2
if pgrep -f "train_challenge1" > /dev/null; then
    echo "✅ Challenge 1 confirmed running"
else
    echo "❌ Challenge 1 failed to start"
fi

if pgrep -f "train_challenge2" > /dev/null; then
    echo "✅ Challenge 2 confirmed running"
else
    echo "❌ Challenge 2 failed to start"
fi
