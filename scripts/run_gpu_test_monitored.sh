#!/bin/bash
# Monitored GPU test runner with external timeout
# This will kill the process if it hangs

TEST_SCRIPT="scripts/gpu_ultra_safe_test.py"
MAX_TIME=120  # 2 minutes total max
LOG_FILE="logs/gpu_safe_test_$(date +%Y%m%d_%H%M%S).log"

echo "🛡️  Monitored GPU Test Runner"
echo "========================================================================"
echo "Script: $TEST_SCRIPT"
echo "Max time: ${MAX_TIME}s"
echo "Log file: $LOG_FILE"
echo "========================================================================"
echo ""
echo "⚠️  SAFETY FEATURES:"
echo "   - External timeout: ${MAX_TIME}s"
echo "   - Can press Ctrl+C to abort"
echo "   - Process will be killed if it hangs"
echo ""
echo "Press Ctrl+C NOW if you want to abort..."
sleep 3
echo ""

# Create logs directory
mkdir -p logs

# Run with timeout
echo "🚀 Starting GPU test..."
echo ""

timeout ${MAX_TIME} python3 $TEST_SCRIPT 2>&1 | tee "$LOG_FILE"
EXIT_CODE=$?

echo ""
echo "========================================================================"

if [ $EXIT_CODE -eq 124 ]; then
    echo "❌ TEST TIMED OUT (exceeded ${MAX_TIME}s)"
    echo "   GPU operations are hanging - NOT SAFE"
    echo "   ✅ CPU training is the safe choice"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TEST COMPLETED SUCCESSFULLY"
    echo "   Check output above for detailed results"
else
    echo "⚠️  TEST EXITED WITH CODE: $EXIT_CODE"
    echo "   Check log file: $LOG_FILE"
fi

echo "========================================================================"
echo "Log saved to: $LOG_FILE"
echo ""
