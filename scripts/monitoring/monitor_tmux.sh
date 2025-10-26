#!/bin/bash

clear
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║          🧠 EEG Challenge Training Monitor (TMUX)                    ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
date
echo ""

# Check tmux sessions
echo "📺 TMUX Sessions:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if tmux ls 2>/dev/null | grep -q "eeg_c"; then
    tmux ls | grep "eeg_c" | while read line; do
        echo "  ✅ $line"
    done
else
    echo "  ❌ No training sessions found"
    echo ""
    echo "  💡 Restart with: ./launch_training_tmux.sh"
    exit 1
fi
echo ""

# Challenge 1 Status
echo "📊 Challenge 1 Status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f logs/training_20251026/c1_final.log ]; then
    # Check for epoch lines
    EPOCHS=$(grep -o "Epoch.*/" logs/training_20251026/c1_final.log 2>/dev/null | tail -1)
    if [ -n "$EPOCHS" ]; then
        echo "  🎯 $EPOCHS"
        # Get latest validation NRMSE
        NRMSE=$(grep "Val NRMSE:" logs/training_20251026/c1_final.log 2>/dev/null | tail -1 | grep -o "Val NRMSE: [0-9.]*" | awk '{print $3}')
        if [ -n "$NRMSE" ]; then
            echo "  📈 Latest Val NRMSE: $NRMSE"
        fi
        # Check for BEST markers
        BEST=$(grep "BEST" logs/training_20251026/c1_final.log 2>/dev/null | tail -1)
        if [ -n "$BEST" ]; then
            echo "  ✨ $BEST"
        fi
    else
        # Still loading data
        LOADING=$(tail -5 logs/training_20251026/c1_final.log 2>/dev/null | grep -E "subjects|Loading" | tail -1)
        echo "  ⏳ $LOADING"
    fi
    
    # Last update
    LAST_LINE=$(tail -1 logs/training_20251026/c1_final.log 2>/dev/null)
    if [ -n "$LAST_LINE" ]; then
        echo "  💬 Last: ${LAST_LINE:0:70}..."
    fi
else
    echo "  ⏳ Initializing..."
fi
echo ""

# Challenge 2 Status
echo "📊 Challenge 2 Status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f logs/training_20251026/c2_final.log ]; then
    # Check for epoch lines
    EPOCHS=$(grep -o "Epoch.*/" logs/training_20251026/c2_final.log 2>/dev/null | tail -1)
    if [ -n "$EPOCHS" ]; then
        echo "  🎯 $EPOCHS"
        # Get latest validation NRMSE
        NRMSE=$(grep "Val NRMSE:" logs/training_20251026/c2_final.log 2>/dev/null | tail -1 | grep -o "Val NRMSE: [0-9.]*" | awk '{print $3}')
        if [ -n "$NRMSE" ]; then
            echo "  📈 Latest Val NRMSE: $NRMSE"
        fi
        # Check for BEST markers
        BEST=$(grep "BEST" logs/training_20251026/c2_final.log 2>/dev/null | tail -1)
        if [ -n "$BEST" ]; then
            echo "  ✨ $BEST"
        fi
    else
        # Still loading data or installing
        LOADING=$(tail -5 logs/training_20251026/c2_final.log 2>/dev/null | grep -E "subjects|Loading|Installing|eegdash" | tail -1)
        echo "  ⏳ $LOADING"
    fi
    
    # Last update
    LAST_LINE=$(tail -1 logs/training_20251026/c2_final.log 2>/dev/null)
    if [ -n "$LAST_LINE" ]; then
        echo "  💬 Last: ${LAST_LINE:0:70}..."
    fi
else
    echo "  ⏳ Initializing..."
fi
echo ""

# System resources
echo "💻 System Resources:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# CPU usage
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
echo "  CPU: ${CPU_USAGE}% used"
# Memory
MEM_INFO=$(free -h | grep "Mem:" | awk '{print "  RAM: "$3" / "$2" used"}')
echo "$MEM_INFO"
# Training processes
TRAIN_PROCS=$(ps aux | grep -E "train_c[12]_sam" | grep -v grep | wc -l)
echo "  Training processes: $TRAIN_PROCS"
echo ""

# Quick commands
echo "🔧 Quick Commands:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Watch C1:  tail -f logs/training_20251026/c1_final.log"
echo "  Watch C2:  tail -f logs/training_20251026/c2_final.log"
echo "  Attach C1: tmux attach -t eeg_c1_train"
echo "  Attach C2: tmux attach -t eeg_c2_train"
echo "  Re-check:  ./monitor_tmux.sh"
echo ""
