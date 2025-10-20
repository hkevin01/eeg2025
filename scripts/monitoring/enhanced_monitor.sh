#!/bin/bash
# Enhanced training monitor with comprehensive debugging

echo "🔍 Enhanced Training Monitor with Full Debug Info"
echo "================================================="
echo ""

while true; do
    clear
    echo "🔍 ENHANCED TRAINING MONITOR - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
    echo ""

    # Check if training process is running
    echo "📊 PROCESS STATUS:"
    if pgrep -f "train_challenge2_r1r2.py" > /dev/null; then
        TRAIN_PID=$(pgrep -f "train_challenge2_r1r2.py")
        echo "✅ Training process running (PID: $TRAIN_PID)"
        ps -p $TRAIN_PID -o pid,ppid,pcpu,pmem,etime,cmd --no-headers 2>/dev/null || echo "   Process details unavailable"
    else
        echo "❌ No training process found"
    fi
    echo ""

    # Show GPU status with more details
    if command -v rocm-smi &> /dev/null; then
        echo "📊 GPU Status (ROCm):"
        rocm-smi --showuse --showmeminfo vram --showtemp --showpower 2>/dev/null || echo "   ROCm-SMI unavailable"
        echo ""
        echo "📊 GPU Memory Details:"
        rocm-smi --showmeminfo all 2>/dev/null | head -10 || echo "   Memory details unavailable"
        echo ""
    elif command -v nvidia-smi &> /dev/null; then
        echo "📊 GPU Status (NVIDIA):"
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits
        echo ""
    else
        echo "❌ No GPU monitoring available"
        echo ""
    fi

    # System resources
    echo "💻 SYSTEM RESOURCES:"
    echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)% | Memory: $(free -h | awk '/^Mem:/ {printf "%s/%s", $3, $2}') | Load: $(uptime | awk -F'load average:' '{print $2}')"
    echo ""

    # Show latest checkpoints
    echo "💾 Latest Checkpoints:"
    ls -lth checkpoints/challenge2_r1r2/*.pth 2>/dev/null | head -5 || echo "   No checkpoints yet"
    echo ""

    # Show training database entries
    echo "📊 Recent Training Runs:"
    sqlite3 data/metadata.db "SELECT run_id, started_at, max_epochs, note FROM training_runs ORDER BY run_id DESC LIMIT 3;" 2>/dev/null || echo "   Database not available"
    echo ""
    echo "📊 Latest Epochs:"
    sqlite3 data/metadata.db "SELECT run_id, epoch, train_loss, val_loss, completed_at FROM epoch_results ORDER BY run_id DESC, epoch DESC LIMIT 5;" 2>/dev/null || echo "   No epoch data yet"
    echo ""

    # Show recent log output
    echo "📝 Recent Training Output:"
    LATEST_LOG=$(ls -t logs/training_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "From: $(basename $LATEST_LOG)"
        if grep -q "CPU fallback ready" "$LATEST_LOG"; then
            echo "⚠️  ROCm failure detected earlier — training resumed on CPU."
        fi
        tail -3 "$LATEST_LOG" 2>/dev/null || echo "   Cannot read log"
    else
        echo "   No training logs found"
    fi
    echo ""

    # Python processes
    echo "🐍 Python Processes:"
    ps aux | grep -E "(python.*train|train.*python)" | grep -v grep | head -3 || echo "   No Python training processes"
    echo ""

    # HDF5 file access check
    echo "📁 HDF5 File Status:"
    if [ -d "data/cached" ]; then
        echo "Cache files: $(ls data/cached/*.h5 2>/dev/null | wc -l) files"
        lsof data/cached/*.h5 2>/dev/null | head -3 || echo "   No active HDF5 file locks"
    else
        echo "   Cache directory not found"
    fi
    echo ""

    echo "================================================================"
    echo "Auto-refresh in 3 seconds... (Ctrl+C to stop)"
    sleep 3
done
