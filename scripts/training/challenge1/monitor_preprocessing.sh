#!/bin/bash
# Monitor feature preprocessing progress

while true; do
    clear
    echo "=============================================="
    echo "Feature Preprocessing Monitor"
    echo "Time: $(date '+%H:%M:%S')"
    echo "=============================================="
    echo
    
    # Check if process is running
    if ps aux | grep -q "[a]dd_neuro_features"; then
        ps aux | grep "[a]dd_neuro_features" | awk '{printf "Process: PID %s, CPU %s%%, MEM %s%%, Time %s\n", $2, $3, $4, $10}'
        echo
    else
        echo "Process completed or not running"
        echo
        python3 /tmp/check_h5_progress.py
        break
    fi
    
    # Check progress
    python3 /tmp/check_h5_progress.py
    
    echo
    echo "Refreshing in 60 seconds... (Ctrl+C to stop)"
    sleep 60
done

echo
echo "✅ Preprocessing complete! Ready for training."
