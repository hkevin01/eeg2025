#!/bin/bash
echo "🔍 Training Status Check - $(date)"
echo "=========================================="

if pgrep -f "train_challenge1" > /dev/null; then
    c1_pid=$(pgrep -f "train_challenge1")
    c1_info=$(ps -p $c1_pid -o pid,pcpu,pmem,etime --no-headers)
    echo "✅ Challenge 1: RUNNING"
    echo "   PID: $c1_pid | $c1_info"
    echo "   Latest: $(tail -3 logs/train_c1_robust_hybrid.log 2>/dev/null | tail -1 | cut -c1-80)"
else
    echo "❌ Challenge 1: STOPPED"
    echo "   Last log: $(tail -1 logs/train_c1_robust_hybrid.log 2>/dev/null)"
fi

echo ""

if pgrep -f "train_challenge2" > /dev/null; then
    c2_pid=$(pgrep -f "train_challenge2")
    c2_info=$(ps -p $c2_pid -o pid,pcpu,pmem,etime --no-headers)
    echo "✅ Challenge 2: RUNNING"
    echo "   PID: $c2_pid | $c2_info"
    echo "   Latest: $(tail -3 logs/train_c2_robust_hybrid.log 2>/dev/null | tail -1 | cut -c1-80)"
else
    echo "❌ Challenge 2: STOPPED"
    echo "   Last log: $(tail -1 logs/train_c2_robust_hybrid.log 2>/dev/null)"
fi

echo ""
echo "📊 Resources:"
free -h | grep "Mem:" | awk '{print "   Memory: "$3" / "$2" ("int($3/$2*100)"% used)"}'
df -h / | tail -1 | awk '{print "   Disk: "$3" / "$2" ("$5" used)"}'
echo "=========================================="
