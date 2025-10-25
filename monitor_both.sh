#!/bin/bash

echo "🔥 SAM Training - Both Challenges"
echo "="*70
echo ""

echo "📊 Tmux Sessions:"
tmux ls 2>/dev/null || echo "No sessions"
echo ""

echo "🧠 Challenge 1 (GPU - Real Data):"
echo "-"*70
if [ -f training_sam_c1.log ]; then
    tail -12 training_sam_c1.log
    echo ""
    echo "Best metrics:"
    grep -i "best\|val nrmse" training_sam_c1.log | tail -3 || echo "(training not started yet)"
else
    echo "Log not found"
fi

echo ""
echo "🧠 Challenge 2 (CPU - Dummy Data ⚠️ ):"
echo "-"*70
if [ -f training_sam_c2.log ]; then
    tail -8 training_sam_c2.log
else
    echo "Log not found"
fi

echo ""
echo "="*70
echo "Commands:"
echo "  C1: tmux attach -t sam_c1"
echo "  C2: tmux attach -t sam_c2"
echo "  Logs: tail -f training_sam_c1.log"
echo "  Kill: tmux kill-session -t sam_c1"
