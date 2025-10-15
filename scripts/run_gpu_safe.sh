#!/bin/bash
# Safe wrapper for running GPU Python scripts

# Source GPU environment
source "$(dirname "$0")/setup_gpu_env.sh"

# Check if script argument provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <python_script.py> [args...]"
    exit 1
fi

# Run with timeout and cleanup
echo "🚀 Running: python3 $@"
echo "   (with 60s timeout for safety)"
echo ""

timeout 60s python3 "$@"
EXIT_CODE=$?

# Cleanup
if [ $EXIT_CODE -eq 124 ]; then
    echo ""
    echo "⚠️  Script timed out after 60s - killed for safety"
    python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null
elif [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ Script exited with error code: $EXIT_CODE"
    python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null
else
    echo ""
    echo "✅ Script completed successfully"
fi

exit $EXIT_CODE
