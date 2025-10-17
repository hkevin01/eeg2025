#!/bin/bash

# Source the GPU function
source monitor_training_enhanced.sh

# Test GPU detection
echo "Testing GPU detection..."
gpu_info=$(get_gpu_info)
echo "Result: $gpu_info"
