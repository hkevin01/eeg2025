#!/bin/bash

echo "🌙 Starting Overnight Training - $(date)"
echo "============================================"

cd /home/kevin/Projects/eeg2025
source venv/bin/activate

# Kill any existing training
pkill -9 -f "train_challenge" 2>/dev/null
sleep 3

# Clear old logs
mkdir -p logs/archive
mv logs/train_c*.log logs/archive/ 2>/dev/null

# Create monitoring script
cat > monitor_overnight.sh << 'MONITOR'
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
MONITOR
chmod +x monitor_overnight.sh

# Start Challenge 1
echo "Starting Challenge 1..."
nohup python scripts/train_challenge1_robust_gpu.py > logs/train_c1_robust_hybrid.log 2>&1 &
C1_PID=$!
echo "   PID: $C1_PID"

# Start Challenge 2
echo "Starting Challenge 2..."
nohup python scripts/train_challenge2_robust_gpu.py > logs/train_c2_robust_hybrid.log 2>&1 &
C2_PID=$!
echo "   PID: $C2_PID"

sleep 5

# Verify processes started
if ps -p $C1_PID > /dev/null 2>&1; then
    echo "✅ Challenge 1 started successfully"
else
    echo "❌ Challenge 1 failed to start"
    tail -20 logs/train_c1_robust_hybrid.log
    exit 1
fi

if ps -p $C2_PID > /dev/null 2>&1; then
    echo "✅ Challenge 2 started successfully"
else
    echo "❌ Challenge 2 failed to start"
    tail -20 logs/train_c2_robust_hybrid.log
    exit 1
fi

echo ""
echo "🌙 Training started successfully!"
echo "============================================"
echo "PIDs: Challenge1=$C1_PID, Challenge2=$C2_PID"
echo ""
echo "Monitor with:"
echo "   ./monitor_overnight.sh"
echo "   tail -f logs/train_c1_robust_hybrid.log"
echo "   tail -f logs/train_c2_robust_hybrid.log"
echo ""
echo "Stop with:"
echo "   pkill -f train_challenge"
echo ""
echo "Expected completion: $(date -d '+6 hours' '+%Y-%m-%d %H:%M')"
echo "============================================"
