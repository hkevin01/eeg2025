#!/bin/bash
# ROCm Environment Setup for RX 5600 XT (gfx1010)
# Using gfx1030 as fallback for rocBLAS compatibility

echo "🔧 Configuring ROCm environment for AMD Radeon RX 5600 XT..."

# GPU Architecture - Use gfx1030 for rocBLAS compatibility (gfx1010 not in PyTorch library)
export PYTORCH_ROCM_ARCH=gfx1030
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# GPU Selection
export HIP_VISIBLE_DEVICES=0

# Workspace Configuration (disable for stability)
export HIPBLAS_WORKSPACE_CONFIG=:0:0

# Memory Optimization
# Set to 1 to disable caching if memory issues occur
export PYTORCH_NO_HIP_MEMORY_CACHING=0

# Debug Level (3-7, higher = more verbose)
# Set to 3 for errors only, 6 for detailed info
export HSAKMT_DEBUG_LEVEL=3

# ROCm Debugging Flags (optional, uncomment if needed)
# export HSA_ENABLE_SDMA=0
# export HSA_ENABLE_INTERRUPT=0
# export HSA_SVM_GUARD_PAGES=0
# export HSA_DISABLE_CACHE=1

# ROCm Memory Management
export HSA_TOOLS_LIB=""
export HSA_TOOLS_REPORT_LOAD_FAILURE=0

# Performance Tuning
export AMD_LOG_LEVEL=0  # Reduce logging overhead
export GPU_MAX_HW_QUEUES=1

echo "✅ Environment configured:"
echo "   PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH"
echo "   HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo "   HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo "   HIPBLAS_WORKSPACE_CONFIG=$HIPBLAS_WORKSPACE_CONFIG"
echo "   HSAKMT_DEBUG_LEVEL=$HSAKMT_DEBUG_LEVEL"
echo ""
echo "🚀 Ready for training!"
echo "ℹ️  Note: Using gfx1030 for rocBLAS compatibility (RX 5600 XT is gfx1010)"
