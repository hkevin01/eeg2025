#!/bin/bash
# Run GPU tests on ROCm 7.0.2

# Clear environment
for var in $(env | grep -i rocm | cut -d= -f1); do
    unset $var
done
unset LD_LIBRARY_PATH
unset PYTHONPATH

# Set ROCm 7.0.2 environment
export PATH=/opt/rocm-7.0.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export ROCM_PATH=/opt/rocm-7.0.2
export LD_LIBRARY_PATH=/opt/rocm-7.0.2/lib
export PYTORCH_ROCM_ARCH=gfx1030
export HIP_VISIBLE_DEVICES=0

# Activate venv
cd /home/kevin/Projects/eeg2025
source venv_rocm702/bin/activate

echo "========================================="
echo "  ROCm 7.0.2 GPU Test Suite"
echo "========================================="
echo ""

# Test 1: Basic Operations (~5s)
echo "🧪 Test 1: Basic GPU Operations (5s)"
timeout 30 python tests/gpu/test_01_basic_operations.py
TEST1=$?
echo ""

# Test 2: Convolutions (~10s) - CRITICAL TEST (ROCm 6.2.2 FAILS HERE)
echo "🧪 Test 2: Convolution Operations (10s) - **CRITICAL**"
timeout 60 python tests/gpu/test_02_convolutions.py
TEST2=$?
echo ""

# Test 3: Training Loop (~15s)
echo "🧪 Test 3: Training Loop (15s)"
timeout 60 python tests/gpu/test_03_training_loop.py
TEST3=$?
echo ""

# Test 4: Memory Stress (~120s) - CRITICAL TEST (ROCm SDK FAILS HERE)
echo "🧪 Test 4: Memory Stress Test (120s) - **CRITICAL**"
timeout 180 python tests/gpu/test_04_memory_stress.py
TEST4=$?
echo ""

echo "========================================="
echo "  Test Results Summary"
echo "========================================="
echo "Test 1 (Basic Ops):     $([ $TEST1 -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo "Test 2 (Convolutions):  $([ $TEST2 -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL') - Critical for EEG"
echo "Test 3 (Training):      $([ $TEST3 -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo "Test 4 (Memory Stress): $([ $TEST4 -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL') - Aperture test"
echo "========================================="

# Overall result
if [ $TEST1 -eq 0 ] && [ $TEST2 -eq 0 ] && [ $TEST3 -eq 0 ] && [ $TEST4 -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED! ROCm 7.0.2 is PRODUCTION READY!"
    exit 0
else
    echo "⚠️  Some tests failed. See details above."
    exit 1
fi
