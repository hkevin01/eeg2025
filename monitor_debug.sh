#!/bin/bash

echo "Monitoring training for debug output..."
echo "Will check every 30 seconds until Epoch 1 completes"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    echo "==================================================================="
    echo "Training Debug Monitor - $(date +'%H:%M:%S')"
    echo "==================================================================="
    
    echo ""
    echo "CHALLENGE 1 STATUS:"
    if grep -q "DEBUG" logs/challenge1_training_v6_debug.log 2>/dev/null; then
        echo "✅ Epoch 1 complete - showing debug output:"
        tail -100 logs/challenge1_training_v6_debug.log | grep -B2 -A15 "DEBUG" | head -40
        break
    else
        tail -5 logs/challenge1_training_v6_debug.log 2>/dev/null | grep -E "Loading|Checking|Creating|Epoch" | tail -2 || echo "Still loading..."
    fi
    
    echo ""
    echo "CHALLENGE 2 STATUS:"
    if grep -q "DEBUG" logs/challenge2_training_v7_debug.log 2>/dev/null; then
        echo "✅ Epoch 1 complete - showing debug output:"
        tail -100 logs/challenge2_training_v7_debug.log | grep -B2 -A15 "DEBUG" | head -40
        break
    else
        tail -5 logs/challenge2_training_v7_debug.log 2>/dev/null | grep -E "Loading|Checking|Creating|Epoch" | tail -2 || echo "Still loading..."
    fi
    
    echo ""
    echo "-------------------------------------------------------------------"
    echo "Next check in 30 seconds..."
    sleep 30
done

echo ""
echo "==================================================================="
echo "✅ Debug output available! Review above for target/prediction stats"
echo "==================================================================="
