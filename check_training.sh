#!/bin/bash

clear
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║          🧠 EEG Challenge Training Status                            ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check tmux sessions
echo "📺 TMUX Sessions:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if tmux ls 2>/dev/null | grep -q "eeg_"; then
    tmux ls | grep "eeg_" | while read line; do
        echo "  ✅ $line"
    done
else
    echo "  ❌ No training sessions found"
fi
echo ""

# Check C1 progress
echo "📊 Challenge 1 Progress:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "logs/training_20251026/c1_tmux.log" ]; then
    tail -10 logs/training_20251026/c1_tmux.log | sed 's/^/  /'
else
    echo "  ⏳ Log file not found yet..."
fi
echo ""

# Check C2 progress
echo "📊 Challenge 2 Progress:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "logs/training_20251026/c2_tmux.log" ]; then
    tail -10 logs/training_20251026/c2_tmux.log | sed 's/^/  /'
else
    echo "  ⏳ Log file not found yet..."
fi
echo ""

# Quick commands
echo "🔧 Quick Commands:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Attach C1:     tmux attach -t eeg_c1_train"
echo "  Attach C2:     tmux attach -t eeg_c2_train"
echo "  Watch C1 log:  tail -f logs/training_20251026/c1_tmux.log"
echo "  Watch C2 log:  tail -f logs/training_20251026/c2_tmux.log"
echo "  Re-check:      ./check_training.sh"
echo ""
