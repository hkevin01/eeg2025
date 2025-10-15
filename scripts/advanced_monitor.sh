#!/bin/bash
# Advanced Training Monitor
# Monitors training without overwhelming VS Code

clear
echo "=========================================="
echo "  Advanced Training Monitor"
echo "=========================================="
echo ""

while true; do
    # Clear screen
    tput cup 5 0
    
    # Check for Python training processes
    echo "📊 Training Process Status:"
    echo "==========================================
"
    
    PROCESS=$(ps aux | grep python | grep -E "(train_foundation|challenge)" | grep -v grep)
    
    if [ -z "$PROCESS" ]; then
        echo "❌ No training process found"
        echo ""
        echo "Start training with:"
        echo "  nohup python3 scripts/train_foundation_v2.py > logs/training.log 2>&1 &"
    else
        echo "$PROCESS" | awk '{printf "✅ PID: %-8s CPU: %-6s MEM: %-6s TIME: %s\n", $2, $3"%", $4"%", $10}'
        
        PID=$(echo "$PROCESS" | awk '{print $2}')
        
        echo ""
        echo "📈 Resource Usage:"
        echo "=========================================="
        
        # CPU and Memory
        CPU=$(echo "$PROCESS" | awk '{print $3}')
        MEM=$(echo "$PROCESS" | awk '{print $4}')
        RSS=$(echo "$PROCESS" | awk '{print $6}')
        MEM_MB=$(echo "scale=1; $RSS / 1024" | bc 2>/dev/null || echo "N/A")
        
        echo "  CPU Usage: ${CPU}%"
        echo "  Memory:    ${MEM}% (${MEM_MB} MB)"
        
        # Elapsed time
        ELAPSED=$(ps -p $PID -o etime= 2>/dev/null | tr -d ' ')
        echo "  Elapsed:   $ELAPSED"
        
        # File descriptors
        FDS=$(lsof -p $PID 2>/dev/null | wc -l)
        echo "  File Descriptors: $FDS"
    fi
    
    echo ""
    echo "📝 Latest Log Output:"
    echo "=========================================="
    
    # Find latest log file
    LATEST_LOG=$(ls -t logs/foundation_*.log 2>/dev/null | head -1)
    
    if [ -z "$LATEST_LOG" ]; then
        echo "❌ No log file found"
    else
        echo "Log: $LATEST_LOG"
        echo ""
        tail -n 15 "$LATEST_LOG" 2>/dev/null || echo "Unable to read log"
    fi
    
    echo ""
    echo "=========================================="
    echo "🔄 Refreshing in 5 seconds... (Ctrl+C to exit)"
    echo "=========================================="
    
    sleep 5
done
