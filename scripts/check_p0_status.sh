#!/bin/bash
# Quick P0 status checker

echo ""
echo "🔴 P0 CRITICAL TASKS STATUS"
echo "============================"
echo ""

# Check data
if [ -d "data/raw/hbn" ] && [ "$(ls -A data/raw/hbn 2>/dev/null)" ]; then
    SUBJECTS=$(ls data/raw/hbn/sub-* 2>/dev/null | wc -l)
    echo "✅ Task 1: Data Acquired ($SUBJECTS subjects)"
else
    echo "⭕ Task 1: Data Acquired (0 subjects)"
    echo "   → Run: python scripts/download_hbn_data.py --subjects 2"
fi

# Check tests
TESTS=$(find tests/ -name "test_*.py" 2>/dev/null | wc -l)
if [ "$TESTS" -ge 15 ]; then
    echo "✅ Task 2: Core Tests ($TESTS tests)"
else
    echo "⭕ Task 2: Core Tests ($TESTS/15 tests)"
    echo "   → Need: $((15 - TESTS)) more tests"
fi

# Check validation
if [ -f "scripts/verify_data_structure.py" ]; then
    echo "✅ Task 3: Validation Scripts"
else
    echo "⭕ Task 3: Validation Scripts"
    echo "   → Scripts already created, ready to run"
fi

# Check benchmark
if [ -f "tests/test_inference_speed.py" ]; then
    echo "✅ Task 4: Inference Benchmark"
else
    echo "⭕ Task 4: Inference Benchmark"
    echo "   → Need to create test_inference_speed.py"
fi

echo ""
echo "📊 Progress Summary"
echo "-------------------"

# Calculate progress
COMPLETED=0
[ -d "data/raw/hbn" ] && [ "$(ls -A data/raw/hbn 2>/dev/null)" ] && COMPLETED=$((COMPLETED + 1))
[ "$TESTS" -ge 15 ] && COMPLETED=$((COMPLETED + 1))
[ -f "scripts/verify_data_structure.py" ] && COMPLETED=$((COMPLETED + 1))
[ -f "tests/test_inference_speed.py" ] && COMPLETED=$((COMPLETED + 1))

PERCENT=$((COMPLETED * 100 / 4))
echo "Completed: $COMPLETED/4 tasks ($PERCENT%)"

# Progress bar
BAR=""
for i in {1..20}; do
    if [ $i -le $((COMPLETED * 5)) ]; then
        BAR="${BAR}█"
    else
        BAR="${BAR}░"
    fi
done
echo "Progress: [$BAR]"

echo ""

if [ $COMPLETED -eq 4 ]; then
    echo "�� ALL P0 TASKS COMPLETE - Ready to train!"
else
    echo "⚠️  $((4 - COMPLETED)) task(s) remaining"
    echo ""
    echo "📖 Next: cat START_HERE_P0.md"
fi

echo ""
