#!/bin/bash
# Check if ROCm convolution fixes are active

echo "══════════════════════════════════════════════════════════════"
echo "🔧 ROCm RDNA1 Convolution Fix - Environment Check"
echo "══════════════════════════════════════════════════════════════"
echo ""

# Source bashrc to get latest environment
source ~/.bashrc 2>/dev/null

# Check critical environment variables
echo "📋 Checking Environment Variables:"
echo "──────────────────────────────────────────────────────────────"

check_var() {
    local var_name=$1
    local expected=$2
    local value="${!var_name}"
    
    if [ "$value" = "$expected" ]; then
        echo "✅ $var_name=$value"
        return 0
    else
        echo "❌ $var_name=$value (expected: $expected)"
        return 1
    fi
}

all_good=true

check_var "MIOPEN_DEBUG_CONV_GEMM" "0" || all_good=false
check_var "MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2" "0" || all_good=false  
check_var "MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW53" "0" || all_good=false
check_var "HSA_OVERRIDE_GFX_VERSION" "10.3.0" || all_good=false

echo ""
echo "📦 Additional Variables:"
echo "──────────────────────────────────────────────────────────────"
echo "   GPU_DEVICE_ORDINAL=$GPU_DEVICE_ORDINAL"
echo "   HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo "   MIOPEN_FIND_MODE=$MIOPEN_FIND_MODE"
echo "   MIOPEN_LOG_LEVEL=$MIOPEN_LOG_LEVEL"

echo ""
echo "══════════════════════════════════════════════════════════════"

if $all_good; then
    echo "✅ ALL CONVOLUTION FIXES ARE ACTIVE!"
    echo ""
    echo "🚀 Ready to test convolutions. Run:"
    echo "   python3 test_rocm_convolution_fixes.py"
    echo ""
    echo "📖 Documentation:"
    echo "   • CONVOLUTION_FIX_SUMMARY.md - Quick reference"
    echo "   • ROCM_CONVOLUTION_FIX.md - Complete guide"
else
    echo "⚠️  CONVOLUTION FIXES NOT FULLY ACTIVE"
    echo ""
    echo "🔧 To fix, run:"
    echo "   source ~/.bashrc"
    echo "   # Or open a new terminal"
    echo ""
    echo "📖 For help, see: CONVOLUTION_FIX_SUMMARY.md"
fi

echo "══════════════════════════════════════════════════════════════"
