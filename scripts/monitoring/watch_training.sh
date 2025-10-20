#!/bin/bash
# Simple training monitor - watches GPU and shows latest training output

echo "🔍 Training Monitor"
echo "=================="
echo ""

while true; do
    clear
    echo "🔍 Training Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
    echo ""

    # Show GPU status
    if command -v rocm-smi &> /dev/null; then
        echo "📊 GPU Status:"
        rocm-smi --showuse --showmeminfo vram | head -20
        echo ""
    fi

    # Show latest training log snippet and detect CPU fallback
    latest_log=$(ls -t /home/kevin/Projects/eeg2025/logs/training_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        if grep -q "CPU fallback ready" "$latest_log"; then
            echo "⚠️ CPU fallback active (ROCm error detected)."
        fi
        echo "📝 Recent Training Log:"
        tail -5 "$latest_log"
        echo ""
    fi

    # Show latest checkpoints
    echo "💾 Latest Checkpoints:"
    ls -lth /home/kevin/Projects/eeg2025/checkpoints/challenge2_r1r2/*.pth 2>/dev/null | head -5 || echo "   No checkpoints yet"
    echo ""

    # Show training database entries
    echo "📊 Recent Training Runs:"
    sqlite3 /home/kevin/Projects/eeg2025/data/metadata.db "SELECT id, model_name, status, datetime(start_time, 'localtime') as start, best_val_loss FROM training_runs ORDER BY id DESC LIMIT 3;" 2>/dev/null || echo "   Database not available"
    echo ""

    echo "Press Ctrl+C to stop monitoring"
    echo "================================================================"

    sleep 5
done
