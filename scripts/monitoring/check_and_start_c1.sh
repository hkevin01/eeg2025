#!/bin/bash
# Check and start Challenge 1 training

cd /home/kevin/Projects/eeg2025

echo "🔍 Checking for running Challenge 1 training..."
if pgrep -f "train_challenge1_enhanced.py" > /dev/null; then
    echo "✅ Training already running!"
    echo "PID: $(pgrep -f 'train_challenge1_enhanced.py')"
    echo ""
    echo "📊 Last 20 lines of log:"
    tail -20 training_challenge1_real.log
else
    echo "❌ Not running. Starting now..."
    echo ""
    
    # Activate SDK and start training
    source activate_sdk.sh > /dev/null 2>&1
    
    # Run in background
    nohup python train_challenge1_enhanced.py > training_challenge1_real.log 2>&1 &
    NEW_PID=$!
    
    echo "✅ Started with PID: $NEW_PID"
    echo ""
    echo "⏳ Waiting 10 seconds for initialization..."
    sleep 10
    
    if ps -p $NEW_PID > /dev/null; then
        echo "✅ Process running!"
        echo ""
        echo "📊 Initial log output:"
        head -50 training_challenge1_real.log
    else
        echo "❌ Process died. Check log:"
        cat training_challenge1_real.log
    fi
fi
