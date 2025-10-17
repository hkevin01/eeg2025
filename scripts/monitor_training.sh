#!/bin/bash
# Monitor training progress for both challenges

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║              📊 PHASE 1 TRAINING MONITOR                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check if processes are running
echo "🔍 Checking running processes..."
echo ""

C1_PID=$(ps aux | grep "train_challenge1_robust_gpu.py" | grep -v grep | awk '{print $2}')
C2_PID=$(ps aux | grep "train_challenge2_robust_gpu.py" | grep -v grep | awk '{print $2}')

if [ -n "$C1_PID" ]; then
    echo "✅ Challenge 1: Running (PID: $C1_PID)"
else
    echo "❌ Challenge 1: Not running"
fi

if [ -n "$C2_PID" ]; then
    echo "✅ Challenge 2: Running (PID: $C2_PID)"
else
    echo "❌ Challenge 2: Not running"
fi

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "CHALLENGE 1: Response Time Prediction"
echo "══════════════════════════════════════════════════════════════════"
echo ""

if [ -f "logs/train_c1_robust_final.log" ]; then
    # Check for epoch progress
    EPOCH_LINE=$(grep -E "Epoch [0-9]+/[0-9]+" logs/train_c1_robust_final.log | tail -1)
    
    if [ -n "$EPOCH_LINE" ]; then
        echo "📈 Latest Progress:"
        echo "$EPOCH_LINE"
        echo ""
        
        # Show last few epochs
        echo "Recent Epochs:"
        grep -E "Epoch [0-9]+/[0-9]+" logs/train_c1_robust_final.log | tail -5
    else
        # Still loading data
        echo "⏳ Status: Loading datasets..."
        echo ""
        echo "Last 10 lines:"
        tail -10 logs/train_c1_robust_final.log
    fi
else
    echo "❌ Log file not found: logs/train_c1_robust_final.log"
fi

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "CHALLENGE 2: Externalizing Behavior Prediction"
echo "══════════════════════════════════════════════════════════════════"
echo ""

if [ -f "logs/train_c2_robust_final.log" ]; then
    # Check for epoch progress
    EPOCH_LINE=$(grep -E "Epoch [0-9]+/[0-9]+" logs/train_c2_robust_final.log | tail -1)
    
    if [ -n "$EPOCH_LINE" ]; then
        echo "📈 Latest Progress:"
        echo "$EPOCH_LINE"
        echo ""
        
        # Show last few epochs
        echo "Recent Epochs:"
        grep -E "Epoch [0-9]+/[0-9]+" logs/train_c2_robust_final.log | tail -5
    else
        # Still loading data
        echo "⏳ Status: Loading datasets..."
        echo ""
        echo "Last 10 lines:"
        tail -10 logs/train_c2_robust_final.log
    fi
else
    echo "❌ Log file not found: logs/train_c2_robust_final.log"
fi

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "Commands:"
echo "  Monitor C1: tail -f logs/train_c1_robust_final.log"
echo "  Monitor C2: tail -f logs/train_c2_robust_final.log"
echo "  Re-run this: bash scripts/monitor_training.sh"
echo "══════════════════════════════════════════════════════════════════"
