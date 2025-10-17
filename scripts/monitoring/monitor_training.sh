#!/bin/bash
echo "=== Challenge 2 Training Monitor ==="
echo ""
echo "Process Status:"
if ps aux | grep -q "[t]rain_challenge2_multi_release.py"; then
    echo "  ✅ Training is RUNNING"
    ps aux | grep "[t]rain_challenge2_multi_release.py" | awk '{printf "  PID: %s, CPU: %s%%, MEM: %s%%\n", $2, $3, $4}'
else
    echo "  ⚠️  Training has STOPPED or COMPLETED"
fi

echo ""
echo "Log File Size:"
ls -lh logs/challenge2_r234_final.log 2>/dev/null | awk '{print "  "$9": "$5}'

echo ""
echo "Latest Progress (last 20 lines with metrics):"
tail -100 logs/challenge2_r234_final.log | grep -E "Epoch|NRMSE|Best|Complete|Windows|Total|Checked|Train |Val " | tail -20

echo ""
echo "=== To watch live: tail -f logs/challenge2_r234_final.log ==="
