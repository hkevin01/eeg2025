#!/bin/bash

# Pretrain SSL model script
# Usage: ./scripts/run_pretrain.sh [--config CONFIG_PATH] [OVERRIDES...]

set -e

# Default configuration
CONFIG_PATH="configs/pretrain.yaml"
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
            echo "  --config PATH    Configuration file path (default: configs/pretrain.yaml)"
            echo "  --help          Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --config configs/pretrain.yaml model.backbone.d_model=256"
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
echo "Starting SSL pretraining..."
echo "Configuration: $CONFIG_PATH"
echo "Python path: $PYTHONPATH"
echo "Hydra overrides: $@"
echo ""

# Run training
$PYTHON_CMD src/training/pretrain_ssl.py \
    --config-path="../$CONFIG_PATH" \
    "$@"

echo ""
echo "SSL pretraining completed!"
