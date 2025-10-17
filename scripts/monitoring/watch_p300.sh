#!/bin/bash
# Continuous monitoring of P300 extraction

LOG_FILE="logs/p300_extraction.log"

while true; do
    clear
    echo "========================================"
    echo "🧠 P300 EXTRACTION LIVE MONITOR"
    echo "========================================"
    date
    echo ""
    
    # Process status
    if pgrep -f "extract_p300_features.py" > /dev/null; then
        PID=$(pgrep -f "extract_p300_features.py")
        CPU=$(ps aux | grep $PID | grep -v grep | awk '{print $3}')
        MEM=$(ps aux | grep $PID | grep -v grep | awk '{print $4}')
        TIME=$(ps aux | grep $PID | grep -v grep | awk '{print $10}')
        echo "✅ Process RUNNING (PID: $PID)"
        echo "   CPU: ${CPU}% | Memory: ${MEM}% | Runtime: ${TIME}"
    else
        echo "⚠️  Process NOT running"
        echo ""
        echo "Last 50 lines of log:"
        tail -50 "$LOG_FILE"
        exit 0
    fi
    
    echo ""
    echo "📁 Cache Files:"
    ls -lh data/processed/p300_cache/ 2>/dev/null | grep -v "^total" || echo "   (none yet)"
    
    echo ""
    echo "📊 Progress Indicators:"
    grep -c "Processing R" "$LOG_FILE" 2>/dev/null | xargs -I {} echo "   Releases started: {}"
    grep -c "Extracted.*trial features" "$LOG_FILE" 2>/dev/null | xargs -I {} echo "   Releases completed: {}"
    
    echo ""
    echo "📜 Recent Activity (last 15 lines):"
    echo "----------------------------------------"
    tail -15 "$LOG_FILE" | grep -E "Processing|Loading|Checking|Creating|Windows|Preprocessing|Extracting|Extracted|Saved|SUMMARY|P300|Correlation" --color=never || tail -15 "$LOG_FILE"
    echo "----------------------------------------"
    
    echo ""
    echo "🔄 Auto-refresh every 5 seconds (Ctrl+C to exit)"
    sleep 5
done
