#!/bin/bash
# Monitor training progress

echo "🔍 Monitoring Training Progress"
echo "================================"
echo ""

# Find most recent training logs
LATEST_C1_LOG=$(ls -t logs/training_comparison/challenge1_improved_*.log 2>/dev/null | head -1)
LATEST_C2_LOG=$(ls -t logs/training_comparison/challenge2_improved_*.log 2>/dev/null | head -1)

if [ -n "$LATEST_C1_LOG" ]; then
    echo "📊 Challenge 1 Progress:"
    echo "------------------------"
    echo "Log: $LATEST_C1_LOG"
    
    # Show last epoch info
    echo ""
    echo "Recent progress:"
    tail -30 "$LATEST_C1_LOG" | grep -E "(Epoch|Train NRMSE|Val NRMSE|Best)"
    
    echo ""
    echo "---"
fi

if [ -n "$LATEST_C2_LOG" ]; then
    echo ""
    echo "📊 Challenge 2 Progress:"
    echo "------------------------"
    echo "Log: $LATEST_C2_LOG"
    
    # Show last epoch info
    echo ""
    echo "Recent progress:"
    tail -30 "$LATEST_C2_LOG" | grep -E "(Epoch|Train NRMSE|Val NRMSE|Best)"
    
    echo ""
    echo "---"
fi

# Show running processes
echo ""
echo "🔄 Running Training Processes:"
echo "------------------------------"
ps aux | grep -E "train_challenge[12]" | grep -v grep || echo "No training processes running"

echo ""
echo "💾 Model Weights:"
echo "----------------"
ls -lh weights_challenge_*.pt 2>/dev/null || echo "No weights found yet"

echo ""
echo "📈 To watch live:"
echo "  tail -f $LATEST_C1_LOG"
echo "  tail -f $LATEST_C2_LOG"
