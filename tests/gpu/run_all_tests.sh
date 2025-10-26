#!/bin/bash

# Master GPU Test Runner
# Tests all GPU functionality and HSA aperture violation detection

echo "=========================================================================="
echo "🧪 EEG2025 GPU Test Suite"
echo "=========================================================================="
echo ""

# Check if environment argument provided
ENV=$1
if [ -z "$ENV" ]; then
    echo "Usage: ./run_all_tests.sh <environment>"
    echo ""
    echo "Available environments:"
    echo "  rocm_sdk    - ROCm SDK 6.1.2 (gfx1010 build)"
    echo "  rocm622     - venv_rocm622 (ROCm 6.2.2)"
    echo "  rocm57      - venv_rocm57"
    echo ""
    exit 1
fi

# Set up environment
cd /home/kevin/Projects/eeg2025

case $ENV in
    rocm_sdk)
        echo "🔧 Environment: ROCm SDK 6.1.2"
        export ROCM_PATH="/opt/rocm_sdk_612"
        export LD_LIBRARY_PATH="/opt/rocm_sdk_612/lib:/opt/rocm_sdk_612/lib64:/opt/rocm_sdk_612/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"
        export PYTHONPATH="/opt/rocm_sdk_612/lib/python3.11/site-packages:$PYTHONPATH"
        export PATH="/opt/rocm_sdk_612/bin:$PATH"
        PYTHON="/opt/rocm_sdk_612/bin/python3"
        ;;
    rocm622)
        echo "🔧 Environment: venv_rocm622 (ROCm 6.2.2)"
        unset PYTHONPATH
        unset LD_LIBRARY_PATH
        unset ROCM_PATH
        source venv_rocm622/bin/activate
        export ROCM_PATH="/opt/rocm"
        export LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64"
        PYTHON="python"
        ;;
    rocm57)
        echo "🔧 Environment: venv_rocm57"
        source venv_rocm57/bin/activate
        PYTHON="python"
        ;;
    *)
        echo "❌ Unknown environment: $ENV"
        exit 1
        ;;
esac

# Common ROCm settings
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_XNACK=0
export HSA_FORCE_FINE_GRAIN_PCIE=1

echo "=========================================================================="
echo ""

# Show environment info
echo "📊 Environment Information:"
$PYTHON -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'HIP: {torch.version.hip}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'Arch: {torch.cuda.get_device_properties(0).gcnArchName if torch.cuda.is_available() else \"N/A\"}')"
echo ""
echo "=========================================================================="
echo ""

# Run tests
TEST_DIR="/home/kevin/Projects/eeg2025/tests/gpu"
TESTS=("test_01_basic_operations.py" "test_02_convolutions.py" "test_03_training_loop.py" "test_04_memory_stress.py")
PASSED=0
FAILED=0
FAILED_TESTS=()

for test in "${TESTS[@]}"; do
    echo ""
    echo "Running: $test"
    echo "--------------------------------------------------------------------------"
    
    if timeout 300 $PYTHON "$TEST_DIR/$test"; then
        ((PASSED++))
        echo "✅ $test PASSED"
    else
        ((FAILED++))
        FAILED_TESTS+=("$test")
        echo "❌ $test FAILED"
    fi
    
    echo "--------------------------------------------------------------------------"
    sleep 2
done

# Summary
echo ""
echo "=========================================================================="
echo "📊 Test Summary for $ENV"
echo "=========================================================================="
echo "Total tests: $((PASSED + FAILED))"
echo "Passed: $PASSED"
echo "Failed: $FAILED"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  ❌ $test"
    done
    echo ""
    echo "=========================================================================="
    exit 1
else
    echo ""
    echo "🎉 ALL TESTS PASSED!"
    echo "✅ Environment $ENV is FULLY FUNCTIONAL"
    echo "=========================================================================="
    exit 0
fi
