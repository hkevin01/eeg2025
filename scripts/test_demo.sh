#!/bin/bash
# Demo Test Runner
# ================
#
# Simple script to run demo tests with better error handling and guidance.

echo "🚀 GPU EEG Demo - Integration Test"
echo "=================================="
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+ to continue."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "backend/demo_server.py" ]; then
    echo "❌ Please run this script from the project root directory."
    echo "   Expected files: backend/demo_server.py, web/demo.html"
    exit 1
fi

echo "📋 Running demo integration tests..."
echo

# Run the improved test script
if [ -f "tests/test_demo_integration_improved.py" ]; then
    python3 tests/test_demo_integration_improved.py
else
    echo "⚠️  Improved test script not found, trying original..."
    if [ -f "tests/test_demo_integration.py" ]; then
        python3 tests/test_demo_integration.py
    else
        echo "❌ No test scripts found in tests/ directory"
        exit 1
    fi
fi

echo
echo "🔧 Demo Management Commands:"
echo "  Start demo:    ./scripts/demo.sh start"
echo "  Stop demo:     ./scripts/demo.sh stop"
echo "  View logs:     ./scripts/demo.sh logs"
echo "  Setup deps:    ./scripts/setup_demo.sh"
echo
echo "📖 Documentation: web/README.md"
echo "🌐 Demo URL: http://localhost:8000 (when running)"
