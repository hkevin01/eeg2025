#!/bin/bash
# Monitor P300 feature extraction progress

LOG_FILE="logs/p300_extraction.log"

echo "========================================"
echo "📊 P300 FEATURE EXTRACTION MONITOR"
echo "========================================"
echo ""

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    exit 1
fi

# Check if process is running
if pgrep -f "extract_p300_features.py" > /dev/null; then
    echo "✅ Extraction process is RUNNING"
else
    echo "⚠️  Extraction process NOT running"
fi

echo ""
echo "📁 Cache Directory:"
ls -lh data/processed/p300_cache/ 2>/dev/null || echo "   (not created yet)"

echo ""
echo "📜 Last 30 lines of log:"
echo "----------------------------------------"
tail -30 "$LOG_FILE"
echo "----------------------------------------"

echo ""
echo "🔍 Statistics:"
grep -c "Processing R" "$LOG_FILE" 2>/dev/null && echo "   Releases processed so far" || true
grep -c "Extracted.*trial features" "$LOG_FILE" 2>/dev/null && echo "   Trial extractions completed" || true
grep "Extracted.*trial features" "$LOG_FILE" 2>/dev/null | tail -3 || true

echo ""
echo "Press Ctrl+C to exit. Process continues in background."
