#!/bin/bash

# Cross-task transfer training script
# Usage: ./scripts/run_cross_task.sh [--config CONFIG_PATH] [OVERRIDES...]

set -e

# Default configuration
CONFIG_PATH="configs/train_cross_task.yaml"
PYTHON_CMD="python"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--config CONFIG_PATH] [HYDRA_OVERRIDES...]"
            echo ""
            echo "Options:"
            echo "  --config PATH    Configuration file path (default: configs/train_cross_task.yaml)"
            echo "  --help          Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --config configs/train_cross_task.yaml training.batch_size=64"
            exit 0
            ;;
        *)
            # Pass remaining arguments as hydra overrides
            break
            ;;
    esac
done

# Ensure we're in the project root
if [[ ! -f "pyproject.toml" ]]; then
    echo "Error: Must run from project root directory"
    exit 1
fi

# Check if config file exists
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: Configuration file not found: $CONFIG_PATH"
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Log start
echo "Starting cross-task transfer training..."
echo "Configuration: $CONFIG_PATH"
echo "Python path: $PYTHONPATH"
echo "Hydra overrides: $@"
echo ""

# Run training
$PYTHON_CMD src/training/train_cross_task.py \
    --config-path="../$CONFIG_PATH" \
    "$@"

echo ""
echo "Cross-task training completed!"
