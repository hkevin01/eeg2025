#!/bin/bash

echo "🧠 Challenge 2 SAM Training Monitor"
echo "="*60

echo ""
echo "📊 Tmux Session Status:"
tmux ls | grep sam_c2 || echo "   ❌ No sam_c2 session found"

echo ""
echo "📁 Latest Training Output (last 40 lines):"
if [ -f training_sam_c2.log ]; then
    tail -40 training_sam_c2.log
else
    echo "   ⚠️  Log file not found yet"
fi

echo ""
echo "📈 Key Metrics (if available):"
if [ -f training_sam_c2.log ]; then
    echo "   Best Val NRMSE:"
    grep -i "best" training_sam_c2.log | tail -5 || echo "   (not yet available)"
    
    echo ""
    echo "   Current Epoch:"
    grep "Epoch [0-9]*/[0-9]*" training_sam_c2.log | tail -1 || echo "   (not yet started)"
fi

echo ""
echo "Commands:"
echo "   Watch training: tmux attach -t sam_c2"
echo "   Detach: Ctrl+B, then D"
echo "   Live log: tail -f training_sam_c2.log"
echo "   Kill training: tmux kill-session -t sam_c2"
