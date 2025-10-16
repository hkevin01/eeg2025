#!/bin/bash
# Monitor training progress and detect when downloading phase completes

echo "================================================================"
echo "🔍 Training Progress Monitor"
echo "================================================================"
echo "Started: $(date)"
echo ""

LAST_C1_LINE=""
LAST_C2_LINE=""
C1_TRAINING_STARTED=0
C2_TRAINING_STARTED=0

while true; do
    # Check if processes are still running
    C1_PID=$(ps aux | grep "train_challenge1_multi_release" | grep -v grep | awk '{print $2}')
    C2_PID=$(ps aux | grep "train_challenge2_multi_release" | grep -v grep | awk '{print $2}')
    
    if [ -z "$C1_PID" ] && [ -z "$C2_PID" ]; then
        echo ""
        echo "⚠️  Both processes have stopped!"
        break
    fi
    
    clear
    echo "================================================================"
    echo "🔍 Training Progress Monitor - $(date +%H:%M:%S)"
    echo "================================================================"
    echo ""
    
    # Challenge 1 Status
    echo "📊 CHALLENGE 1 (Response Time Prediction)"
    echo "   PID: ${C1_PID:-Stopped}"
    if [ -n "$C1_PID" ]; then
        # Check for training start indicators
        if grep -q "Training Multi-Release Model" logs/challenge1_training.log 2>/dev/null; then
            if [ $C1_TRAINING_STARTED -eq 0 ]; then
                echo "   🎉 TRAINING STARTED!"
                C1_TRAINING_STARTED=1
            fi
            # Show latest epoch info
            EPOCH_LINE=$(grep -E "Epoch [0-9]+/[0-9]+" logs/challenge1_training.log 2>/dev/null | tail -1)
            NRMSE_LINE=$(grep "Val NRMSE:" logs/challenge1_training.log 2>/dev/null | tail -1)
            echo "   Status: TRAINING"
            [ -n "$EPOCH_LINE" ] && echo "   $EPOCH_LINE"
            [ -n "$NRMSE_LINE" ] && echo "   $NRMSE_LINE"
        else
            # Still downloading
            CURRENT_LINE=$(tail -1 logs/challenge1_training.log 2>/dev/null)
            if [[ "$CURRENT_LINE" == *"Downloading"* ]]; then
                echo "   Status: 📥 Downloading data..."
                echo "   Latest: $(echo $CURRENT_LINE | grep -oP 'sub-[A-Z0-9]+' | head -1)"
            elif [[ "$CURRENT_LINE" == *"Loading"* ]] || [[ "$CURRENT_LINE" == *"Windows"* ]]; then
                echo "   Status: 🔄 Preprocessing data..."
            else
                echo "   Status: 🔄 Initializing..."
            fi
        fi
    fi
    echo ""
    
    # Challenge 2 Status
    echo "📊 CHALLENGE 2 (Externalizing Prediction)"
    echo "   PID: ${C2_PID:-Stopped}"
    if [ -n "$C2_PID" ]; then
        # Check for training start indicators
        if grep -q "Training Multi-Release Model" logs/challenge2_training_retry.log 2>/dev/null; then
            if [ $C2_TRAINING_STARTED -eq 0 ]; then
                echo "   🎉 TRAINING STARTED!"
                C2_TRAINING_STARTED=1
            fi
            # Show latest epoch info
            EPOCH_LINE=$(grep -E "Epoch [0-9]+/[0-9]+" logs/challenge2_training_retry.log 2>/dev/null | tail -1)
            NRMSE_LINE=$(grep "Val NRMSE:" logs/challenge2_training_retry.log 2>/dev/null | tail -1)
            echo "   Status: TRAINING"
            [ -n "$EPOCH_LINE" ] && echo "   $EPOCH_LINE"
            [ -n "$NRMSE_LINE" ] && echo "   $NRMSE_LINE"
        else
            # Still downloading
            CURRENT_LINE=$(tail -1 logs/challenge2_training_retry.log 2>/dev/null)
            if [[ "$CURRENT_LINE" == *"Downloading"* ]]; then
                echo "   Status: �� Downloading data..."
                echo "   Latest: $(echo $CURRENT_LINE | grep -oP 'sub-[A-Z0-9]+' | head -1)"
            elif [[ "$CURRENT_LINE" == *"Loading"* ]] || [[ "$CURRENT_LINE" == *"Windows"* ]]; then
                echo "   Status: �� Preprocessing data..."
            else
                echo "   Status: 🔄 Initializing..."
            fi
        fi
    fi
    echo ""
    echo "================================================================"
    
    # Check if both have started training
    if [ $C1_TRAINING_STARTED -eq 1 ] && [ $C2_TRAINING_STARTED -eq 1 ]; then
        echo ""
        echo "✅ DOWNLOAD PHASE COMPLETE!"
        echo "🎯 Both challenges are now TRAINING"
        echo ""
        echo "Next steps:"
        echo "  1. Let training run overnight (~14 hours)"
        echo "  2. Check back tomorrow for results"
        echo "  3. Validation scores will be in the logs"
        echo ""
        echo "Monitor commands:"
        echo "  tail -f logs/challenge1_training.log"
        echo "  tail -f logs/challenge2_training_retry.log"
        echo ""
        break
    fi
    
    echo "Refreshing in 30 seconds... (Ctrl+C to stop)"
    sleep 30
done

echo ""
echo "Monitor stopped at: $(date)"
