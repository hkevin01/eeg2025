#!/bin/bash

echo "=========================================="
echo "🧠 EEG Training Quick Status"
echo "=========================================="
echo ""

# Check processes
echo "📊 Running Processes:"
ps aux | grep "[p]ython3 scripts/train_challenge" | awk '{print "  ✓ PID " $2 " - CPU: " $3 "% - " ($NF ~ /challenge1/ ? "Challenge 1" : "Challenge 2")}'

echo ""
echo "🎮 GPU Status:"
if command -v rocm-smi &> /dev/null; then
    gpu_name=$(rocm-smi --showproductname 2>/dev/null | grep "Card Series" | cut -d':' -f3 | sed 's/^[[:space:]]*//' | head -1)
    echo "  ✓ AMD GPU: $gpu_name"
    echo "  ✓ ROCm: Enabled"
else
    echo "  ℹ CPU Only"
fi

echo ""
echo "📈 Latest Training Results:"
echo ""
echo "Challenge 1 (Response Time):"
if [ -f logs/challenge1_fresh_start.log ]; then
    tail -100 logs/challenge1_fresh_start.log | grep -E "Epoch [0-9]+/50" | tail -1
    tail -100 logs/challenge1_fresh_start.log | grep "Train NRMSE:" | tail -1 | awk '{print "  Train: " $3}'
    tail -100 logs/challenge1_fresh_start.log | grep "Val NRMSE:" | tail -1 | awk '{print "  Val:   " $3}'
    tail -100 logs/challenge1_fresh_start.log | grep "Best model saved" | tail -1 | awk '{print "  Best:  " $5 ")"}'
else
    echo "  ❌ No log file"
fi

echo ""
echo "Challenge 2 (Externalizing):"
if [ -f logs/challenge2_fresh_start.log ]; then
    tail -100 logs/challenge2_fresh_start.log | grep -E "Epoch [0-9]+/50" | tail -1
    tail -100 logs/challenge2_fresh_start.log | grep "Train NRMSE:" | tail -1 | awk '{print "  Train: " $3}'
    tail -100 logs/challenge2_fresh_start.log | grep "Val NRMSE:" | tail -1 | awk '{print "  Val:   " $3}'
    tail -100 logs/challenge2_fresh_start.log | grep "Best model saved" | tail -1 | awk '{print "  Best:  " $5 ")"}'
else
    echo "  ❌ No log file"
fi

echo ""
echo "=========================================="
