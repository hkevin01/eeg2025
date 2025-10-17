#!/bin/bash
while true; do
    clear
    echo "🌙 Overnight Training Monitor - $(date)"
    echo "========================================"
    
    # Check processes
    if pgrep -f "train_challenge1" > /dev/null; then
        c1_pid=$(pgrep -f "train_challenge1")
        c1_info=$(ps -p $c1_pid -o pid,pcpu,pmem,etime --no-headers)
        echo "✅ Challenge 1: RUNNING"
        echo "   $c1_info"
    else
        echo "❌ Challenge 1: STOPPED"
    fi
    
    if pgrep -f "train_challenge2" > /dev/null; then
        c2_pid=$(pgrep -f "train_challenge2")
        c2_info=$(ps -p $c2_pid -o pid,pcpu,pmem,etime --no-headers)
        echo "✅ Challenge 2: RUNNING"
        echo "   $c2_info"
    else
        echo "❌ Challenge 2: STOPPED"
    fi
    
    echo ""
    echo "📊 System Resources:"
    free -h | grep "Mem:" | awk '{print "   Memory: "$3" / "$2" ("int($3/$2*100)"%)"}'
    df -h / | tail -1 | awk '{print "   Disk: "$3" / "$2" ("$5")"}'
    
    echo ""
    echo "📝 Latest Progress:"
    tail -5 logs/train_c1_robust_hybrid.log 2>/dev/null | grep -E "(Epoch|Loading R|Total:)" | tail -2
    tail -5 logs/train_c2_robust_hybrid.log 2>/dev/null | grep -E "(Epoch|Loading R|Total:)" | tail -2
    
    sleep 30
done
