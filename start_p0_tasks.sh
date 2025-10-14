#!/bin/bash

# P0 Critical Tasks - Daily Action Script
# Run this each day to see what needs to be done

echo ""
echo "🔴 ================================================"
echo "   P0 CRITICAL TASKS - Daily Action Plan"
echo "   $(date '+%A, %B %d, %Y')"
echo "================================================"
echo ""

# Function to check if something exists
check_status() {
    if [ $1 -eq 0 ]; then
        echo "✅"
    else
        echo "⭕"
    fi
}

# Check current status
echo "📊 CURRENT STATUS"
echo "----------------"

# Task 1: Data
[ -d "data/raw/hbn/sub-NDARAA536PTU" ]
STATUS_DATA=$?
echo "$(check_status $STATUS_DATA) Task 1: Data Acquired"

# Task 2: Tests
TEST_COUNT=$(find tests/ -name "test_*.py" 2>/dev/null | wc -l)
if [ "$TEST_COUNT" -ge 15 ]; then
    echo "✅ Task 2: Core Tests ($TEST_COUNT/15+)"
else
    echo "⭕ Task 2: Core Tests ($TEST_COUNT/15)"
fi

# Task 3: Validation Scripts
[ -f "scripts/verify_data_structure.py" ]
STATUS_VAL=$?
echo "$(check_status $STATUS_VAL) Task 3: Validation Scripts"

# Task 4: Benchmark
[ -f "tests/test_inference_speed.py" ]
STATUS_BENCH=$?
echo "$(check_status $STATUS_BENCH) Task 4: Inference Benchmark"

echo ""
echo "🎯 TODAY'S RECOMMENDED ACTIONS"
echo "------------------------------"

# Determine what to do today
if [ $STATUS_DATA -ne 0 ]; then
    echo ""
    echo "▶️  PRIORITY 1: Get Data (Critical!)"
    echo ""
    echo "Run these commands:"
    echo "  1. pip install mne mne-bids boto3 requests tqdm"
    echo "  2. mkdir -p data/raw/hbn data/processed data/cache"
    echo "  3. python scripts/download_hbn_data.py --subjects 2 --verify"
    echo ""
    echo "Time: 30-60 minutes"
    echo "Why: Cannot do anything without data!"
    echo ""
    
elif [ "$TEST_COUNT" -lt 15 ]; then
    echo ""
    echo "▶️  PRIORITY 1: Write Tests"
    echo ""
    echo "Today's goal: Add 5 more tests"
    echo ""
    echo "Quick start:"
    echo "  cp tests/conftest.py tests/test_data_loading.py"
    echo "  # Edit test_data_loading.py and add tests"
    echo "  pytest tests/test_data_loading.py -v"
    echo ""
    echo "Time: 2-3 hours"
    echo ""
    
elif [ $STATUS_VAL -ne 0 ]; then
    echo ""
    echo "▶️  PRIORITY 1: Validation"
    echo ""
    echo "Run:"
    echo "  python scripts/verify_data_structure.py --data-dir data/raw/hbn"
    echo "  python scripts/validate_data_statistics.py --data-dir data/raw/hbn"
    echo ""
    echo "Time: 1-2 hours"
    echo ""
    
elif [ $STATUS_BENCH -ne 0 ]; then
    echo ""
    echo "▶️  PRIORITY 1: Benchmark Inference"
    echo ""
    echo "Run:"
    echo "  pytest tests/test_inference_speed.py -v -s"
    echo ""
    echo "Time: 1 hour"
    echo ""
    
else
    echo ""
    echo "🎉 ALL P0 TASKS COMPLETE!"
    echo ""
    echo "You're ready to start training:"
    echo "  python src/training/train_cross_task.py --config configs/challenge1_baseline.yaml"
    echo ""
fi

echo "📚 RESOURCES"
echo "------------"
echo "  📖 Full Plan: cat CRITICAL_TASKS_P0.md"
echo "  📖 Data Guide: cat docs/DATA_ACQUISITION_GUIDE.md"
echo "  📖 Quick Start: cat GETTING_STARTED_WITH_DATA.md"
echo ""

echo "💡 QUICK TIPS"
echo "-------------"
echo "  • Work in 90-minute focused blocks"
echo "  • Run this script daily to track progress"
echo "  • Ask for help if stuck >30 minutes"
echo "  • Commit working code frequently"
echo ""

echo "🚀 Ready to start? Pick your priority above!"
echo ""
