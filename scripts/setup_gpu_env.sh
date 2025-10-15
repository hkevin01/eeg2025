#!/bin/bash
# Setup safe GPU environment for AMD ROCm

# AMD ROCm Environment Variables
export HSA_OVERRIDE_GFX_VERSION="10.3.0"
export HIP_VISIBLE_DEVICES="0"
export ROCR_VISIBLE_DEVICES="0"
export GPU_MAX_HEAP_SIZE="100"
export GPU_MAX_ALLOC_PERCENT="100"

# PyTorch Memory Management
export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:128"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Disable problematic features
export HSA_ENABLE_SDMA="0"

# Python optimizations
export PYTHONUNBUFFERED="1"

echo "✅ GPU Environment configured for AMD ROCm"
echo "   HSA_OVERRIDE_GFX_VERSION: $HSA_OVERRIDE_GFX_VERSION"
echo "   HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
echo "   Memory limits: max_split_size_mb=128"
