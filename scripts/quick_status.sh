#!/bin/bash
# Quick one-time status check (no loop)

C1_LOG="logs/challenge1_training_v7_R4val_fixed.log"
C2_LOG="logs/challenge2_training_v9_R4val_fixed.log"

echo "════════════════════════════════════════════════════════════════"
echo "🎯 QUICK TRAINING STATUS - $(date '+%H:%M:%S')"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Process count
echo "📊 ACTIVE PROCESSES:"
ps aux | grep "[p]ython3 scripts/train_challenge" | awk '{printf "   %-50s PID: %-7s CPU: %5s%%\n", $NF, $2, $3}'
echo ""

# Challenge 1
echo "📊 CHALLENGE 1:"
if [ -f "$C1_LOG" ]; then
    if grep -q "Epoch" "$C1_LOG" 2>/dev/null; then
        EPOCH=$(grep -oP "Epoch \K\d+(?=/50)" "$C1_LOG" | tail -1)
        TRAIN=$(grep "Train NRMSE:" "$C1_LOG" | tail -1 | awk '{print $NF}')
        VAL=$(grep "Val NRMSE:" "$C1_LOG" | tail -1 | awk '{print $NF}')
        BEST=$(grep "Best Val NRMSE" "$C1_LOG" | tail -1 | grep -oP '\d+\.\d+')
        
        echo "   Status: ✅ TRAINING"
        echo "   Epoch: $EPOCH/50"
        [ ! -z "$TRAIN" ] && echo "   Train NRMSE: $TRAIN"
        [ ! -z "$VAL" ] && echo "   Val NRMSE: $VAL"
        [ ! -z "$BEST" ] && echo "   Best Val: $BEST ⭐"
    else
        echo "   Status: 🔄 Loading data"
        tail -3 "$C1_LOG" | grep -E "Loading|Creating|Windows" | tail -1 | sed 's/^[[:space:]]*/   /'
    fi
else
    echo "   Status: ❌ Log not found"
fi

echo ""

# Challenge 2
echo "📊 CHALLENGE 2:"
if [ -f "$C2_LOG" ]; then
    if grep -q "Epoch" "$C2_LOG" 2>/dev/null; then
        EPOCH=$(grep -oP "Epoch \K\d+(?=/50)" "$C2_LOG" | tail -1)
        TRAIN=$(grep "Train NRMSE:" "$C2_LOG" | tail -1 | awk '{print $NF}')
        VAL=$(grep "Val NRMSE:" "$C2_LOG" | tail -1 | awk '{print $NF}')
        BEST=$(grep "Best Val NRMSE" "$C2_LOG" | tail -1 | grep -oP '\d+\.\d+')
        
        echo "   Status: ✅ TRAINING"
        echo "   Epoch: $EPOCH/50"
        [ ! -z "$TRAIN" ] && echo "   Train NRMSE: $TRAIN"
        [ ! -z "$VAL" ] && echo "   Val NRMSE: $VAL"
        [ ! -z "$BEST" ] && echo "   Best Val: $BEST ⭐"
        
        # Check validation std
        VAL_STD=$(grep "Validation.*Std:" "$C2_LOG" | tail -1 | grep -oP 'Std: \K\d+\.\d+')
        if [ ! -z "$VAL_STD" ]; then
            echo "   Val Std: $VAL_STD ✓"
        fi
    else
        echo "   Status: 🔄 Loading data"
        tail -5 "$C2_LOG" | grep -E "Loading|Creating|Windows|Std:" | tail -2 | sed 's/^[[:space:]]*/   /'
    fi
else
    echo "   Status: ❌ Log not found"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Commands:"
echo "  ./monitor_training_enhanced.sh  - Live monitor (refreshes every 10s)"
echo "  ./quick_status.sh               - One-time status check"
echo "  tail -f logs/challenge1_training_v7_R4val_fixed.log | grep NRMSE"
echo "  tail -f logs/challenge2_training_v9_R4val_fixed.log | grep NRMSE"
echo "════════════════════════════════════════════════════════════════"
